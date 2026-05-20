"""토론 관련 REST + WebSocket 엔드포인트.

phase-6: 인메모리 매니저 레지스트리(`_managers`)를 완전히 제거했다. 모든 라우터는
DB(`database`)에서 discussion_id 로 상태를 로드하고, 무상태 `Orchestrator` 의
이벤트 진입점(`trigger` → `process_event`)을 호출한다. WebSocket 연결만은 본질적
으로 인메모리이므로 `SocketRegistry` 가 브로드캐스트용으로 보관한다.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .. import database
from ..manager import LLMClientPool, Orchestrator, PipelineEvent
from ..schemas import (
    CreateDiscussionRequest,
    CreateDiscussionResponse,
    DiscussionState,
    DiscussionStatus,
    ManualResponseRequest,
    UserIntervention,
    WSMessage,
    WSMessageType,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/discussions", tags=["discussion"])


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


#: 모듈 수준 단일 인스턴스. `pool`/`orchestrator` 는 main.py lifespan 이 사용.
sockets = SocketRegistry()
pool = LLMClientPool()
orchestrator = Orchestrator(pool, sockets.broadcast)


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
async def create_discussion(req: CreateDiscussionRequest) -> CreateDiscussionResponse:
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
async def advance_discussion(discussion_id: str) -> dict[str, str]:
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
    discussion_id: str, intervention: UserIntervention
) -> dict[str, str]:
    """유저 개입을 주입한다 (낙관적 락으로 동시 갱신 충돌을 흡수)."""
    await _load_or_404(discussion_id)
    await orchestrator.add_intervention(discussion_id, intervention)
    return {"status": "intervention_recorded", "discussion_id": discussion_id}


@router.post("/{discussion_id}/manual-response", status_code=202)
async def submit_manual_response(
    discussion_id: str, req: ManualResponseRequest
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
        {"agent_id": req.agent_id, "phase": req.phase.value, "content": req.content},
    )
    return {"status": "manual_response_accepted", "discussion_id": discussion_id}


# ---------------------------------------------------------------------------
# WebSocket 엔드포인트
# ---------------------------------------------------------------------------
@router.websocket("/{discussion_id}/ws")
async def discussion_ws(websocket: WebSocket, discussion_id: str) -> None:
    """토론 진행 상황 실시간 스트림 + 유저 개입/진행 수신 채널."""
    await websocket.accept()

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

    try:
        while True:
            raw = await websocket.receive_json()
            await _handle_client_message(discussion_id, websocket, raw)
    except WebSocketDisconnect:
        sockets.unregister(discussion_id, websocket)
    except Exception:  # noqa: BLE001 - 비정상 소켓 정리
        logger.exception("WS 처리 오류 (discussion=%s)", discussion_id)
        sockets.unregister(discussion_id, websocket)


async def _handle_client_message(
    discussion_id: str, websocket: WebSocket, raw: dict
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
