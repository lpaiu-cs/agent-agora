"""토론 관련 REST + WebSocket 엔드포인트."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .. import database
from ..manager import DiscussionManager, InvalidStateTransition, LLMClientPool
from ..schemas import (
    CreateDiscussionRequest,
    CreateDiscussionResponse,
    DiscussionState,
    ManualResponseRequest,
    UserIntervention,
    WSMessage,
    WSMessageType,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/discussions", tags=["discussion"])


class DiscussionRegistry:
    """토론 세션 레지스트리 — 상태는 SQLite 에 영속, 활성 매니저만 메모리 보유.

    설계:
      * 토론 상태(`DiscussionState`)의 단일 진실 공급원은 SQLite DB(`database`).
        조회는 ``load_state``(DB), 저장은 ``_persist`` 콜백(DB)이 담당한다.
      * 진행 중 세션을 오케스트레이션하는 `DiscussionManager` 는 asyncio
        프리미티브를 들고 있어 본질적으로 인메모리 — `_managers` 에 보관한다.
      * `pool` 은 앱 레벨 글로벌 LLM 클라이언트 풀(서버 종료 시 lifespan 이 폐쇄).
    """

    def __init__(self) -> None:
        self._managers: dict[str, DiscussionManager] = {}
        self._sockets: dict[str, list[WebSocket]] = {}
        self.pool = LLMClientPool()

    @staticmethod
    async def _persist(state: DiscussionState) -> None:
        """상태를 DB 에 저장한다 — 동기 DB I/O 는 스레드로 오프로드해 루프 비차단."""
        await asyncio.to_thread(database.save_state, state)

    @staticmethod
    async def load_state(discussion_id: str) -> Optional[DiscussionState]:
        """DB 에서 토론 상태를 조회한다 (종료된 세션 포함). 없으면 None."""
        return await asyncio.to_thread(database.load_state, discussion_id)

    def create(self, req: CreateDiscussionRequest) -> DiscussionManager:
        """새 토론을 생성한다 — 매니저 구성 + 초기 상태(CREATED)를 DB 에 저장."""
        discussion_id = uuid.uuid4().hex
        state = DiscussionState(
            discussion_id=discussion_id,
            topic=req.topic,
            agents=req.agents,
            force_consensus=req.force_consensus,
        )
        manager = DiscussionManager(
            state,
            broadcast=lambda msg: self.broadcast(discussion_id, msg),
            pool=self.pool,
            persist=self._persist,  # 단계 체크포인트마다 DB 에 영속화
        )
        self._managers[discussion_id] = manager
        self._sockets.setdefault(discussion_id, [])
        database.save_state(state)  # 초기 상태 영속화
        return manager

    def get(self, discussion_id: str) -> DiscussionManager:
        """활성 매니저를 조회한다. 없으면 HTTP 404 (제어 엔드포인트용)."""
        manager = self._managers.get(discussion_id)
        if manager is None:
            raise HTTPException(
                status_code=404,
                detail="활성 토론 세션을 찾을 수 없습니다 (미존재 또는 서버 재시작).",
            )
        return manager

    def get_or_none(self, discussion_id: str) -> Optional[DiscussionManager]:
        """활성 매니저를 조회한다. 없으면 None (WebSocket 경로용)."""
        return self._managers.get(discussion_id)

    def register_socket(self, discussion_id: str, ws: WebSocket) -> None:
        self._sockets.setdefault(discussion_id, []).append(ws)

    def unregister_socket(self, discussion_id: str, ws: WebSocket) -> None:
        sockets = self._sockets.get(discussion_id, [])
        if ws in sockets:
            sockets.remove(ws)

    async def broadcast(self, discussion_id: str, message: WSMessage) -> None:
        """해당 토론에 연결된 모든 소켓으로 메시지를 전송한다.

        전송에 실패한 소켓은 등록 해제한다.
        """
        encoded = message.model_dump(mode="json")
        dead: list[WebSocket] = []
        for ws in list(self._sockets.get(discussion_id, [])):
            try:
                await ws.send_json(encoded)
            except Exception:  # noqa: BLE001 - 끊긴 소켓 정리
                dead.append(ws)
        for ws in dead:
            self.unregister_socket(discussion_id, ws)


#: 모듈 수준 단일 레지스트리.
registry = DiscussionRegistry()


# ---------------------------------------------------------------------------
# REST 엔드포인트
# ---------------------------------------------------------------------------
@router.post("", response_model=CreateDiscussionResponse, status_code=201)
async def create_discussion(req: CreateDiscussionRequest) -> CreateDiscussionResponse:
    """새 토론 세션을 생성하고 5단계 파이프라인을 백그라운드로 기동한다."""
    manager = registry.create(req)
    manager.start()
    return CreateDiscussionResponse(
        discussion_id=manager.state.discussion_id,
        status=manager.state.status,
        current_phase=manager.state.current_phase,
    )


@router.get("/{discussion_id}", response_model=DiscussionState)
async def get_discussion(discussion_id: str) -> DiscussionState:
    """토론의 전체 상태 스냅샷을 DB 에서 조회해 반환한다."""
    state = await registry.load_state(discussion_id)
    if state is None:
        raise HTTPException(status_code=404, detail="토론을 찾을 수 없습니다.")
    return state


@router.post("/{discussion_id}/advance", status_code=202)
async def advance_discussion(discussion_id: str) -> dict[str, str]:
    """게이트 락 상태의 파이프라인에 다음 단계 진입을 승인한다.

    WAITING_FOR_USER 가 아닌 상태에서의 호출은 HTTP 409 로 거부된다.
    """
    manager = registry.get(discussion_id)
    try:
        manager.request_advance()
    except InvalidStateTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "advance_requested", "discussion_id": discussion_id}


@router.post("/{discussion_id}/interventions", status_code=201)
async def add_intervention(
    discussion_id: str, intervention: UserIntervention
) -> dict[str, str]:
    """단계 사이 게이트 락 구간에 유저 개입을 주입한다."""
    manager = registry.get(discussion_id)
    await manager.submit_user_intervention(intervention)
    return {"status": "intervention_recorded", "discussion_id": discussion_id}


@router.post("/{discussion_id}/manual-response", status_code=202)
async def submit_manual_response(
    discussion_id: str, req: ManualResponseRequest
) -> dict[str, str]:
    """수동(manual) 에이전트의 응답을 주입해 대기 중인 파이프라인을 재구동한다.

    대기 중인 수동 입력 요청이 없으면 HTTP 409 로 거부된다.
    """
    manager = registry.get(discussion_id)
    try:
        manager.submit_manual_response(req.agent_id, req.phase, req.content)
    except InvalidStateTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "manual_response_submitted", "discussion_id": discussion_id}


# ---------------------------------------------------------------------------
# WebSocket 엔드포인트
# ---------------------------------------------------------------------------
@router.websocket("/{discussion_id}/ws")
async def discussion_ws(websocket: WebSocket, discussion_id: str) -> None:
    """토론 진행 상황 실시간 스트림 + 유저 개입 수신 채널.

    접속 직후 현재 상태 스냅샷을 1회 전송하고, 이후 매니저의 브로드캐스트를
    중계한다. 클라이언트는 개입/단계진행 메시지를 보낼 수 있다.
    """
    await websocket.accept()

    manager = registry.get_or_none(discussion_id)
    if manager is None:
        await websocket.send_json(
            WSMessage(
                type=WSMessageType.ERROR,
                payload={"message": "활성 토론 세션을 찾을 수 없습니다."},
            ).model_dump(mode="json")
        )
        await websocket.close(code=4004)
        return

    registry.register_socket(discussion_id, websocket)
    # 접속 직후 현재(라이브) 상태 스냅샷 1회 전송.
    await websocket.send_json(
        WSMessage(
            type=WSMessageType.STATE_SNAPSHOT,
            payload={"state": manager.state.model_dump(mode="json")},
        ).model_dump(mode="json")
    )

    try:
        while True:
            raw = await websocket.receive_json()
            await _handle_client_message(manager, websocket, raw)
    except WebSocketDisconnect:
        registry.unregister_socket(discussion_id, websocket)
    except Exception:  # noqa: BLE001 - 비정상 소켓 정리
        logger.exception("WS 처리 오류 (discussion=%s)", discussion_id)
        registry.unregister_socket(discussion_id, websocket)


async def _handle_client_message(
    manager: DiscussionManager, websocket: WebSocket, raw: dict
) -> None:
    """클라이언트 -> 서버 WS 메시지를 처리한다."""
    msg_type = raw.get("type")
    payload = raw.get("payload") or {}

    if msg_type == WSMessageType.USER_INTERVENTION.value:
        intervention = UserIntervention.model_validate(payload)
        await manager.submit_user_intervention(intervention)
    elif msg_type == WSMessageType.ADVANCE_PHASE.value:
        try:
            manager.request_advance()
        except InvalidStateTransition as exc:
            await websocket.send_json(
                WSMessage(
                    type=WSMessageType.ERROR, payload={"message": str(exc)}
                ).model_dump(mode="json")
            )
    else:
        logger.warning("알 수 없는 WS 메시지 타입: %r", msg_type)
