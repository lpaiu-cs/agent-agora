"""토론 관련 REST + WebSocket 엔드포인트.

phase-6: 인메모리 매니저 레지스트리(`_managers`)를 제거 — 라우터는 DB 에서
상태를 로드하고 무상태 `Orchestrator` 의 이벤트 진입점을 호출한다.
phase-8: 인프라 객체(LLM 풀·오케스트레이터·소켓 레지스트리)를 모듈 전역에서
FastAPI `app.state` 로 격상하고, HTTP 라우터는 `Depends` 로 주입받는다 —
모듈 임포트 순서 의존을 끊어 결합도를 낮춘다 (구조 검토 ③ 교정).
"""

from __future__ import annotations

import logging
import uuid

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)

from .. import database
from ..manager import Orchestrator, PipelineEvent
from ..schemas import (
    CreateDiscussionRequest,
    CreateDiscussionResponse,
    DiscussionState,
    DiscussionStatus,
    ManualResponseRequest,
    ModelProvider,
    RefinePersonaRequest,
    RefinePersonaResponse,
    UserIntervention,
    WSMessage,
    WSMessageType,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/discussions", tags=["discussion"])

#: 특정 토론에 속하지 않는 보조 도구 엔드포인트 (페르소나 윤문 등) — prefix 없음.
tools_router = APIRouter(tags=["tools"])


class SocketRegistry:
    """discussion_id 별 WebSocket 연결 집합 — 브로드캐스트 전용 인메모리 레지스트리.

    WebSocket 은 살아있는 TCP 연결이라 본질적으로 인메모리다. 토론 *상태* 는
    DB 에 있고, 여기에는 연결 핸들만 둔다.
    """

    def __init__(self) -> None:
        self._sockets: dict[str, list[WebSocket]] = {}

    def register(self, discussion_id: str, ws: WebSocket) -> None:
        self._sockets.setdefault(discussion_id, []).append(ws)

    def unregister(self, discussion_id: str, ws: WebSocket) -> None:
        socks = self._sockets.get(discussion_id, [])
        if ws in socks:
            socks.remove(ws)

    async def broadcast(self, discussion_id: str, message: WSMessage) -> None:
        """해당 토론의 모든 소켓으로 전송한다. 실패한 소켓은 등록 해제."""
        encoded = message.model_dump(mode="json")
        dead: list[WebSocket] = []
        for ws in list(self._sockets.get(discussion_id, [])):
            try:
                await ws.send_json(encoded)
            except Exception:  # noqa: BLE001 - 끊긴 소켓 정리
                dead.append(ws)
        for ws in dead:
            self.unregister(discussion_id, ws)


# ---------------------------------------------------------------------------
# 의존성 — app.state 에 바인딩된 인프라 객체를 라우터로 주입한다.
# 풀/오케스트레이터/소켓 레지스트리의 생명주기는 main.py lifespan 이 소유하며,
# 라우터는 모듈 전역 변수에 의존하지 않는다 (결합도↓ — 구조 검토 ③ 교정).
# ---------------------------------------------------------------------------
def get_orchestrator(request: Request) -> Orchestrator:
    """HTTP 요청 컨텍스트에서 app.state 의 Orchestrator 를 주입한다."""
    return request.app.state.orchestrator


async def _load_or_404(discussion_id: str) -> DiscussionState:
    """DB 에서 상태를 로드한다. 없으면 HTTP 404."""
    state = await database.load_state(discussion_id)
    if state is None:
        raise HTTPException(status_code=404, detail="토론을 찾을 수 없습니다.")
    return state


# ---------------------------------------------------------------------------
# REST 엔드포인트
# ---------------------------------------------------------------------------
@router.post("", response_model=CreateDiscussionResponse, status_code=201)
async def create_discussion(
    req: CreateDiscussionRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> CreateDiscussionResponse:
    """새 토론을 DB 에 영속화하고 START 이벤트로 파이프라인을 기동한다."""
    discussion_id = uuid.uuid4().hex
    state = DiscussionState(
        discussion_id=discussion_id,
        topic=req.topic,
        agents=req.agents,
        force_consensus=req.force_consensus,
    )
    await database.insert_state(state)
    orchestrator.trigger(discussion_id, PipelineEvent.START)
    return CreateDiscussionResponse(
        discussion_id=discussion_id,
        status=state.status,
        current_phase=state.current_phase,
    )


@router.get("/{discussion_id}", response_model=DiscussionState)
async def get_discussion(discussion_id: str) -> DiscussionState:
    """토론의 전체 상태 스냅샷을 DB 에서 조회해 반환한다."""
    return await _load_or_404(discussion_id)


@router.post("/{discussion_id}/advance", status_code=202)
async def advance_discussion(
    discussion_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> dict[str, str]:
    """다음 단계 진입을 승인한다. WAITING_FOR_USER 가 아니면 HTTP 409."""
    state = await _load_or_404(discussion_id)
    if state.status is not DiscussionStatus.WAITING_FOR_USER:
        raise HTTPException(
            status_code=409,
            detail=f"advance 는 'waiting_for_user' 상태에서만 가능합니다 "
                   f"(현재: {state.status.value}).",
        )
    orchestrator.trigger(discussion_id, PipelineEvent.ADVANCE)
    return {"status": "advance_requested", "discussion_id": discussion_id}


@router.post("/{discussion_id}/interventions", status_code=201)
async def add_intervention(
    discussion_id: str,
    intervention: UserIntervention,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> dict[str, str]:
    """유저 개입을 주입한다 (낙관적 락으로 동시 갱신 충돌을 흡수)."""
    await _load_or_404(discussion_id)
    await orchestrator.add_intervention(discussion_id, intervention)
    return {"status": "intervention_recorded", "discussion_id": discussion_id}


@router.post("/{discussion_id}/manual-response", status_code=202)
async def submit_manual_response(
    discussion_id: str,
    req: ManualResponseRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> dict[str, str]:
    """수동(manual) 에이전트의 응답을 주입해 MANUAL_RESPONSE 이벤트를 트리거한다.

    PENDING_MANUAL_INPUT 상태가 아니면 HTTP 409. 새 HTTP 워커가 DB 에서 상태를
    읽어 다음 상태 전이 함수를 직접 트리거하므로, 서버 재시작 후에도 동작한다.
    """
    state = await _load_or_404(discussion_id)
    if state.status is not DiscussionStatus.PENDING_MANUAL_INPUT:
        raise HTTPException(
            status_code=409,
            detail=f"수동 입력은 'pending_manual_input' 상태에서만 가능합니다 "
                   f"(현재: {state.status.value}).",
        )
    orchestrator.trigger(
        discussion_id, PipelineEvent.MANUAL_RESPONSE,
        {"agent_id": req.agent_id, "phase": req.phase.value,
         "content": req.content},
    )
    return {"status": "manual_response_accepted", "discussion_id": discussion_id}


# ---------------------------------------------------------------------------
# 보조 도구 엔드포인트 (페르소나 윤문)
# ---------------------------------------------------------------------------
@tools_router.post("/personas/refine", response_model=RefinePersonaResponse)
async def refine_persona(
    req: RefinePersonaRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> RefinePersonaResponse:
    """페르소나 초안을 토론 주제에 맞춰 윤문한다.

    윤문은 요청에 담긴 provider/model(보통 해당 에이전트 슬롯 설정)로 수행한다.
    manual 공급자는 호출할 API 가 없으므로 거부한다(400).
    """
    if req.provider is ModelProvider.MANUAL:
        raise HTTPException(
            status_code=400,
            detail="manual 슬롯은 윤문할 수 없습니다. "
                   "OpenAI/Anthropic/Ollama 슬롯에서 시도하세요.",
        )
    try:
        refined = await orchestrator.refine_persona(
            topic=req.topic,
            draft=req.draft,
            provider=req.provider,
            model=req.model,
            name=req.name,
            persona_role=req.persona_type.value if req.persona_type else "",
        )
    except Exception as exc:  # noqa: BLE001 - LLM 호출 실패를 502 로 변환
        raise HTTPException(status_code=502, detail=f"윤문 실패: {exc}") from exc
    if not refined:
        raise HTTPException(status_code=502, detail="윤문 결과가 비어 있습니다.")
    return RefinePersonaResponse(refined=refined)


# ---------------------------------------------------------------------------
# WebSocket 엔드포인트
# ---------------------------------------------------------------------------
@router.websocket("/{discussion_id}/ws")
async def discussion_ws(websocket: WebSocket, discussion_id: str) -> None:
    """토론 진행 상황 실시간 스트림 + 유저 개입/진행 수신 채널.

    인프라 객체는 `websocket.app.state` 에서 직접 얻는다 (WS 경로는 HTTP 와
    의존성 주입 결이 달라, app.state 직접 접근이 더 단순·확실하다).
    """
    await websocket.accept()
    orchestrator: Orchestrator = websocket.app.state.orchestrator
    sockets: SocketRegistry = websocket.app.state.sockets

    state = await database.load_state(discussion_id)
    if state is None:
        await websocket.send_json(
            WSMessage(
                type=WSMessageType.ERROR,
                payload={"message": "토론을 찾을 수 없습니다."},
            ).model_dump(mode="json")
        )
        await websocket.close(code=4004)
        return

    sockets.register(discussion_id, websocket)
    # 접속 직후 DB 상태 스냅샷 1회 전송.
    await websocket.send_json(
        WSMessage(
            type=WSMessageType.STATE_SNAPSHOT,
            payload={"state": state.model_dump(mode="json")},
        ).model_dump(mode="json")
    )
    # 수동 대기 '2중 방어선' — PENDING_MANUAL_INPUT 이면 복붙 페이로드를 이 소켓에
    # 재전송한다. 새로고침·재연결로 복붙 터널 패널이 증발해도 복구할 수 있다.
    await orchestrator.emit_manual_input_required_for_socket(discussion_id, websocket)

    try:
        while True:
            raw = await websocket.receive_json()
            await _handle_client_message(discussion_id, websocket, raw, orchestrator)
    except WebSocketDisconnect:
        sockets.unregister(discussion_id, websocket)
    except Exception:  # noqa: BLE001 - 비정상 소켓 정리
        logger.exception("WS 처리 오류 (discussion=%s)", discussion_id)
        sockets.unregister(discussion_id, websocket)


async def _handle_client_message(
    discussion_id: str, websocket: WebSocket, raw: dict, orchestrator: Orchestrator
) -> None:
    """클라이언트 -> 서버 WS 메시지를 처리한다."""
    msg_type = raw.get("type")
    payload = raw.get("payload") or {}

    if msg_type == WSMessageType.USER_INTERVENTION.value:
        intervention = UserIntervention.model_validate(payload)
        await orchestrator.add_intervention(discussion_id, intervention)
    elif msg_type == WSMessageType.ADVANCE_PHASE.value:
        state = await database.load_state(discussion_id)
        if state is not None and state.status is DiscussionStatus.WAITING_FOR_USER:
            orchestrator.trigger(discussion_id, PipelineEvent.ADVANCE)
        else:
            await websocket.send_json(
                WSMessage(
                    type=WSMessageType.ERROR,
                    payload={"message": "지금은 다음 단계로 진행할 수 없습니다."},
                ).model_dump(mode="json")
            )
    else:
        logger.warning("알 수 없는 WS 메시지 타입: %r", msg_type)
