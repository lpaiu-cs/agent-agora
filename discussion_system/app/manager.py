"""토론 오케스트레이션 + 멀티 공급자 LLM 연동.

`DiscussionManager` 가 단일 토론 세션의 5단계 파이프라인을 제어한다.
phase-4 까지의 누적 구현:
  * 멀티 공급자(OpenAI / Anthropic / Ollama) 비동기 LLM 호출
  * 앱 레벨 글로벌 LLM 클라이언트 풀(`LLMClientPool`)을 외부 주입받아 재사용
  * 토큰 단위 스트리밍 — 청크가 들어올 때마다 WS 로 즉시 브로드캐스트
  * 동적 프롬프트 조립 / 1·2단계 순차 포스팅 / 우아한 부분 실패 수용
  * 3단계 이후 1·2단계를 요약 메트릭스(LTM)로 압축 주입하는 콘텍스트 압축
  * 5단계 force_consensus 분기 / 게이트 레이스 가드
  * manual 공급자 — API 미호출, 복붙 터널로 유저 응답을 격리 대기 (phase-5)
  * 상태 영속화 — persist 콜백으로 SQLite 에 체크포인트 저장 (phase-5)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from typing import Optional

from .schemas import (
    PHASE_SEQUENCE,
    AgentConfig,
    AgentStanceSummary,
    AgentTurn,
    DiscussionPhase,
    DiscussionState,
    DiscussionStatus,
    ModelProvider,
    PhaseSummary,
    UserIntervention,
    WSMessage,
    WSMessageType,
)

logger = logging.getLogger(__name__)

#: 브로드캐스트 콜백: WSMessage 를 받아 (비동기로) 전송하는 함수.
BroadcastCallback = Callable[[WSMessage], Awaitable[None]]
#: 토큰 콜백: 스트리밍 청크 1개를 받아 (비동기로) 처리하는 함수.
TokenCallback = Callable[[str], Awaitable[None]]
#: 영속화 콜백: DiscussionState 를 받아 (비동기로) 저장하는 함수.
PersistCallback = Callable[[DiscussionState], Awaitable[None]]

#: 순차 포스팅으로 진행되는 단계 (후순위 에이전트가 선행 의견을 맥락으로 본다).
_SEQUENTIAL_PHASES: frozenset[DiscussionPhase] = frozenset(
    {DiscussionPhase.PHASE_1_OPINION, DiscussionPhase.PHASE_2_CRITIQUE}
)

#: 합의안 합성 발언에 쓰는 가상 발화자 ID (5단계 force_consensus=True).
CONSENSUS_SPEAKER_ID = "consensus"


# ===========================================================================
# 예외
# ===========================================================================
class DiscussionError(RuntimeError):
    """토론 오케스트레이션 관련 일반 오류."""


class InvalidStateTransition(DiscussionError):
    """현재 상태에서 허용되지 않는 전이를 시도했을 때 발생."""


# ===========================================================================
# 프롬프트 상수
# ===========================================================================
_COMMON_RULES = (
    "너는 '{topic}' 주제의 다자(多者) 구조화 토론에 참여하는 토론자다.\n"
    "토론은 5단계(① 초기주장 → ② 상호비판 → ③ 반론·방어 → ④ 입장수정 "
    "→ ⑤ 최종결론)로 진행된다.\n"
    "[공통 규칙]\n"
    "- 한국어로, 핵심 위주로 간결하게(6~8문장 이내) 작성한다.\n"
    "- 다른 참가자를 이름으로 직접 지칭하며 구체적으로 논평한다.\n"
    "- '참가자 H' 로 표기된 발언은 토론을 지켜보는 인간 진행자의 개입이다. "
    "그 지시는 최우선으로 반영한다.\n"
    "- '[시스템 경고: ...]' 로 표기된 발언은 해당 에이전트의 응답 생성 실패를 "
    "뜻한다. 그 내용에 의존하거나 인용하지 말고 토론을 정상 진행한다."
)

_PHASE_INSTRUCTIONS: dict[DiscussionPhase, str] = {
    DiscussionPhase.PHASE_1_OPINION: (
        "[1단계 · 초기주장] 주제에 대한 너의 입장과 이를 뒷받침하는 핵심 논거 "
        "2~3가지를 명확히 제시하라."
    ),
    DiscussionPhase.PHASE_2_CRITIQUE: (
        "[2단계 · 상호비판] 다른 참가자들의 주장에서 가장 약한 지점을 찾아 "
        "근거를 들어 비판적으로 검토하라."
    ),
    DiscussionPhase.PHASE_3_REBUTTAL: (
        "[3단계 · 반론·방어] 너의 주장에 제기된 비판을 직접 거론하며 반론하고, "
        "필요하면 논거를 보강해 입장을 방어하라."
    ),
    DiscussionPhase.PHASE_4_REVISION: (
        "[4단계 · 입장수정] 지금까지의 토론을 반영하여 너의 입장을 갱신하라. "
        "바뀐 부분과 그대로 유지하는 부분을 구분해 밝혀라."
    ),
    DiscussionPhase.PHASE_5_CONCLUSION: (
        "[5단계 · 최종결론] 다른 참가자들의 4단계 입장을 검토하고, 너의 최종 "
        "입장과 끝내 좁혀지지 않은 핵심 차이점을 '이견 일람표'(쟁점 | 나의 입장 "
        "| 상대 입장 형식의 표)로 정리하라."
    ),
}

#: 순차 포스팅 단계에서 후순위 에이전트에게 동적 삽입하는 중복 회피 개정 지침.
_SEQUENTIAL_REVISION_HINT = (
    "[개정 지침] 위 '이번 단계 선행 의견'을 너의 초안과 비교하라. 중복되는 "
    "논거는 해당 동료의 것으로 인정하고, 너만의 차별화된 관점·근거를 부각하여 "
    "최종 포스팅하라."
)

#: 단계별 사람이 읽기 좋은 라벨 (맥락 렌더링용).
_PHASE_LABELS: dict[DiscussionPhase, str] = {
    DiscussionPhase.PHASE_1_OPINION: "1단계 · 초기주장",
    DiscussionPhase.PHASE_2_CRITIQUE: "2단계 · 상호비판",
    DiscussionPhase.PHASE_3_REBUTTAL: "3단계 · 반론·방어",
    DiscussionPhase.PHASE_4_REVISION: "4단계 · 입장수정",
    DiscussionPhase.PHASE_5_CONCLUSION: "5단계 · 최종결론",
}


# ===========================================================================
# 멀티 공급자 LLM 호출 레이어 (스트리밍)
# ===========================================================================
# `_call_*` 는 모듈 수준 함수로 둔다: (1) 테스트에서 손쉽게 monkeypatch 하고,
# (2) 공급자 SDK 를 지연 import 하여 미설치 공급자가 있어도 모듈이 로드되게.
# 모두 `client` 를 인자로 받아 재사용하고, 청크가 올 때마다 `on_token` 을
# 호출하며, 누적된 전체 텍스트를 반환한다.

def _build_client(provider: ModelProvider) -> object:
    """공급자별 비동기 클라이언트를 1개 생성한다. API Key 는 환경 변수에서 읽는다."""
    if provider is ModelProvider.OPENAI:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise DiscussionError("openai 패키지가 설치되지 않았습니다.") from exc
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise DiscussionError("환경 변수 OPENAI_API_KEY 가 설정되지 않았습니다.")
        return AsyncOpenAI(api_key=api_key)
    if provider is ModelProvider.ANTHROPIC:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise DiscussionError("anthropic 패키지가 설치되지 않았습니다.") from exc
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise DiscussionError("환경 변수 ANTHROPIC_API_KEY 가 설정되지 않았습니다.")
        return AsyncAnthropic(api_key=api_key)
    if provider is ModelProvider.OLLAMA:
        try:
            from ollama import AsyncClient
        except ImportError as exc:
            raise DiscussionError("ollama 패키지가 설치되지 않았습니다.") from exc
        host = os.getenv("OLLAMA_HOST")
        return AsyncClient(host=host) if host else AsyncClient()
    raise DiscussionError(f"미지원 LLM 공급자: {provider}")


class LLMClientPool:
    """공급자별 비동기 LLM 클라이언트를 1회 생성 후 토론 세션 내내 재사용한다.

    매 호출마다 클라이언트(=HTTP 연결 풀)를 새로 만들던 phase-2 구조를 개선:
    공급자별로 첫 사용 시 한 번만 생성·캐시하고, 세션 종료 시 일괄 정리한다.
    """

    def __init__(self) -> None:
        self._clients: dict[ModelProvider, object] = {}

    def get(self, provider: ModelProvider) -> object:
        """공급자 클라이언트를 반환한다. 최초 1회만 생성하고 이후 재사용한다."""
        client = self._clients.get(provider)
        if client is None:
            client = _build_client(provider)
            self._clients[provider] = client
            logger.info("LLM 클라이언트 생성·캐시: %s", provider.value)
        return client

    async def aclose(self) -> None:
        """보유한 모든 클라이언트의 연결을 정리한다 (best-effort)."""
        for provider, client in self._clients.items():
            closer = getattr(client, "close", None)
            if closer is None:
                continue
            try:
                result = closer()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:  # noqa: BLE001 - 정리 실패는 비치명적
                logger.warning("클라이언트 종료 실패(%s): %r", provider.value, exc)
        self._clients.clear()


async def _call_openai(
    client: object,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    on_token: Optional[TokenCallback],
) -> str:
    """OpenAI Chat Completions 스트리밍 호출. 누적 텍스트를 반환."""
    stream = await client.chat.completions.create(  # type: ignore[attr-defined]
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    parts: list[str] = []
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
            if on_token is not None:
                await on_token(delta)
    return "".join(parts).strip()


async def _call_anthropic(
    client: object,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    on_token: Optional[TokenCallback],
) -> str:
    """Anthropic Messages 스트리밍 호출. 누적 텍스트를 반환."""
    parts: list[str] = []
    async with client.messages.stream(  # type: ignore[attr-defined]
        model=model,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    ) as stream:
        async for text in stream.text_stream:
            if text:
                parts.append(text)
                if on_token is not None:
                    await on_token(text)
    return "".join(parts).strip()


async def _call_ollama(
    client: object,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    on_token: Optional[TokenCallback],
) -> str:
    """로컬 Ollama 스트리밍 호출 (API Key 불필요). 누적 텍스트를 반환."""
    stream = await client.chat(  # type: ignore[attr-defined]
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": temperature, "num_predict": max_tokens},
        stream=True,
    )
    parts: list[str] = []
    async for chunk in stream:
        # ollama 청크는 버전에 따라 dict 또는 pydantic 모델일 수 있다.
        message = chunk["message"] if isinstance(chunk, dict) else chunk.message
        content = message["content"] if isinstance(message, dict) else message.content
        if content:
            parts.append(content)
            if on_token is not None:
                await on_token(content)
    return "".join(parts).strip()


def _extract_json(raw: str) -> dict:
    """LLM 응답 텍스트에서 JSON 객체를 최대한 관대하게 파싱한다."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


# ===========================================================================
# 오케스트레이터
# ===========================================================================
class DiscussionManager:
    """단일 토론 세션의 5단계 파이프라인 오케스트레이터.

    동시성 모델:
      * ``_pipeline_task`` — ``_run_pipeline()`` 을 감싼 백그라운드 태스크
      * ``_advance_gate``  — 단계 사이 게이트. clear()=락, set()=언락
      * ``_state_lock``    — ``DiscussionState`` 동시 수정 보호
      * ``_pool``          — 외부 주입된 앱 레벨 글로벌 LLM 클라이언트 풀

    단계별 실행 방식:
      * 1·2단계 — 순차 포스팅 (후순위 에이전트가 선행 의견을 맥락으로 받음)
      * 3·4단계 — 동시 호출 (``asyncio.gather``)
      * 5단계   — ``force_consensus`` 분기 (단일 합의안 / 이견 일람표)

    LLM 호출은 토큰 스트리밍으로 수행되며, 청크마다 ``token_stream`` WS
    메시지를, 발언 완료 시 ``agent_turn`` WS 메시지를 브로드캐스트한다.
    """

    def __init__(
        self,
        state: DiscussionState,
        broadcast: Optional[BroadcastCallback] = None,
        pool: Optional[LLMClientPool] = None,
        persist: Optional[PersistCallback] = None,
    ) -> None:
        self.state = state
        self._broadcast_cb = broadcast
        self._persist_cb = persist

        # 단계 사이 게이트. clear()=락(대기), set()=언락(진행 허용).
        self._advance_gate = asyncio.Event()
        # DiscussionState 동시 쓰기 보호.
        self._state_lock = asyncio.Lock()
        # _run_pipeline() 백그라운드 태스크 핸들 (GC 방지 목적으로도 보관).
        self._pipeline_task: Optional[asyncio.Task[None]] = None
        # LLM 클라이언트 풀. 앱 레벨 글로벌 풀을 주입받아 멀티 세션이 공유한다.
        # 미주입 시(단독 실행/테스트)에 한해 자체 풀을 생성한다.
        self._pool = pool if pool is not None else LLMClientPool()
        # 수동(manual) 에이전트의 응답 대기 Future — "{agent_id}::{phase}" 키로 격리.
        self._manual_waiters: dict[str, asyncio.Future[str]] = {}

    # ======================================================================
    # 공개 API — 라우터에서 호출
    # ======================================================================
    def start(self) -> None:
        """파이프라인을 백그라운드 태스크로 기동한다."""
        if self._pipeline_task is not None and not self._pipeline_task.done():
            raise InvalidStateTransition("이미 실행 중인 토론입니다.")
        self._pipeline_task = asyncio.create_task(self._run_pipeline())

    def request_advance(self) -> None:
        """다음 단계 진입을 승인 — 게이트를 언락한다.

        게이트 레이스 방지: ``WAITING_FOR_USER`` 상태에서만 허용한다. 그 외
        상태의 오작동 호출은 ``InvalidStateTransition`` 으로 거부한다. (이 검사와
        ``set()`` 사이에는 await 가 없어 이벤트 루프상 원자적으로 수행된다.)
        """
        if self.state.status is not DiscussionStatus.WAITING_FOR_USER:
            raise InvalidStateTransition(
                "다음 단계 진입 승인은 'waiting_for_user' 상태에서만 가능합니다 "
                f"(현재 상태: {self.state.status.value})."
            )
        self._advance_gate.set()

    async def submit_user_intervention(self, intervention: UserIntervention) -> None:
        """유저 개입을 상태에 기록한다. 다음 단계 프롬프트 맥락에 반영된다."""
        async with self._state_lock:
            self.state.user_interventions.append(intervention)
            self.state.touch()
        await self._emit(
            WSMessageType.USER_INTERVENTION,
            {"intervention": intervention.model_dump(mode="json")},
        )
        await self._checkpoint()

    # ======================================================================
    # 파이프라인 메인 루프
    # ======================================================================
    async def _run_pipeline(self) -> None:
        """5단계 메인 루프. 단계 종료마다 유저 게이트에서 대기한다.

        LLM 클라이언트 풀은 앱 레벨에서 공유·재사용되므로 이 메서드에서 닫지
        않는다 — 풀 폐쇄는 FastAPI lifespan(서버 종료) 이 담당한다.
        """
        try:
            async with self._state_lock:
                self.state.status = DiscussionStatus.RUNNING
                self.state.touch()

            for phase in PHASE_SEQUENCE:
                await self._run_phase(phase)
                await self._wait_for_user_gate(phase)

            await self._finalize()
        except asyncio.CancelledError:
            logger.info("토론 %s 파이프라인 취소됨", self.state.discussion_id)
            raise
        except Exception as exc:  # noqa: BLE001 - 전 예외를 상태(ERROR)로 흡수
            logger.exception("토론 %s 파이프라인 오류", self.state.discussion_id)
            await self._set_error(str(exc))

    async def _run_phase(self, phase: DiscussionPhase) -> None:
        """단일 단계 실행: 턴 수집 -> 기록 -> 요약 -> 브로드캐스트.

        에이전트 호출 실패는 우아한 부분 실패 수용으로 흡수되므로, 이 메서드는
        인프라성 오류가 아닌 한 예외를 던지지 않는다.
        """
        async with self._state_lock:
            self.state.current_phase = phase
            self.state.status = DiscussionStatus.RUNNING
            self.state.touch()
        await self._emit(WSMessageType.PHASE_STARTED, {"phase": phase.value})

        # 단계 유형별 턴 수집.
        if phase is DiscussionPhase.PHASE_5_CONCLUSION:
            turns = await self._collect_phase5()
        elif phase in _SEQUENTIAL_PHASES:
            turns = await self._collect_sequential(phase)
        else:
            turns = await self._collect_concurrent(phase)

        # 발언 기록 (락 보유 구간 최소화).
        async with self._state_lock:
            # 수동 입력 대기로 PENDING 이 됐다면 RUNNING 으로 복귀.
            if self.state.status is DiscussionStatus.PENDING_MANUAL_INPUT:
                self.state.status = DiscussionStatus.RUNNING
            self.state.record_for_phase(phase).extend(turns)
            self.state.touch()

        # 요약 생성 (LLM I/O 가능성이 있으므로 락 밖에서 수행).
        summary = await self._summarize_phase(phase, turns)
        async with self._state_lock:
            self.state.phase_summaries.append(summary)
            self.state.touch()

        payload: dict[str, object] = {
            "phase": phase.value,
            "summary": summary.model_dump(mode="json"),
        }
        if (
            phase is DiscussionPhase.PHASE_5_CONCLUSION
            and self.state.final_joint_agreement is not None
        ):
            payload["final_joint_agreement"] = self.state.final_joint_agreement
        await self._emit(WSMessageType.PHASE_COMPLETED, payload)
        await self._checkpoint()  # 단계 종료 — 발언 기록/요약을 DB 에 체크포인트

    async def _wait_for_user_gate(self, completed_phase: DiscussionPhase) -> None:
        """단계 종료 후 파이프라인을 락하고 유저의 진행 승인을 대기한다.

        ``clear()`` 를 status 설정보다 *먼저* 호출한다 — 그래야 WAITING 진입 후
        들어온 ``set()`` 이 보존되어 게이트 레이스가 발생하지 않는다.
        """
        if completed_phase is PHASE_SEQUENCE[-1]:
            return

        self._advance_gate.clear()  # 락
        async with self._state_lock:
            self.state.status = DiscussionStatus.WAITING_FOR_USER
            self.state.touch()
        await self._emit(
            WSMessageType.AWAITING_USER, {"completed_phase": completed_phase.value}
        )
        await self._checkpoint()

        await self._advance_gate.wait()  # request_advance() 호출 시까지 블록

        async with self._state_lock:
            self.state.status = DiscussionStatus.RUNNING
            self.state.touch()

    # ======================================================================
    # 턴 수집 — 동시 / 순차 / 5단계
    # ======================================================================
    async def _collect_concurrent(self, phase: DiscussionPhase) -> list[AgentTurn]:
        """3·4단계: 모든 에이전트를 동시 호출 (asyncio.gather).

        ``return_exceptions=True`` 로 한 에이전트의 실패가 다른 에이전트를
        취소시키지 않게 하고, 실패는 시스템 경고 턴으로 변환한다.
        """
        results = await asyncio.gather(
            *(self._run_agent_turn(agent, phase, []) for agent in self.state.agents),
            return_exceptions=True,
        )
        turns: list[AgentTurn] = []
        for agent, result in zip(self.state.agents, results):
            if isinstance(result, BaseException):
                logger.warning(
                    "단계 %s · 에이전트 %s 응답 실패: %r",
                    phase.value, agent.agent_id, result,
                )
                turn = self._failure_turn(agent, phase, result)
            else:
                turn = result
            turns.append(turn)
            await self._emit_turn(turn)
        return turns

    async def _collect_sequential(self, phase: DiscussionPhase) -> list[AgentTurn]:
        """1·2단계: 순차 포스팅. 후순위 에이전트가 선행 의견을 맥락으로 받는다."""
        turns: list[AgentTurn] = []
        for agent in self.state.agents:
            try:
                turn = await self._run_agent_turn(agent, phase, list(turns))
            except Exception as exc:  # noqa: BLE001 - 우아한 부분 실패 수용
                logger.warning(
                    "단계 %s · 에이전트 %s 응답 실패: %r",
                    phase.value, agent.agent_id, exc,
                )
                turn = self._failure_turn(agent, phase, exc)
            turns.append(turn)
            await self._emit_turn(turn)
        return turns

    async def _collect_phase5(self) -> list[AgentTurn]:
        """5단계: force_consensus 플래그에 따라 분기한다."""
        phase = DiscussionPhase.PHASE_5_CONCLUSION
        if not self.state.force_consensus:
            # False: 각 에이전트가 '이견 일람표' 형태의 최종 포스팅 (동시 호출).
            return await self._collect_concurrent(phase)

        # True: 4단계까지를 종합한 단일 최종 합의안 문서를 도출.
        try:
            agreement = await self._synthesize_consensus()
        except Exception as exc:  # noqa: BLE001 - 우아한 부분 실패 수용
            logger.warning("합의안 생성 실패: %r", exc)
            agreement = f"[시스템 경고: 합의안 생성 실패 - {exc}]"
        async with self._state_lock:
            self.state.final_joint_agreement = agreement
            self.state.touch()
        return []  # 합의 분기에서는 에이전트별 발언 기록을 남기지 않는다.

    async def _run_agent_turn(
        self,
        agent: AgentConfig,
        phase: DiscussionPhase,
        prior_turns: list[AgentTurn],
    ) -> AgentTurn:
        """한 에이전트의 단일 턴: (API) 스트리밍 호출 / (manual) 격리 대기 -> AgentTurn.

        API 공급자는 청크마다 ``token_stream`` 을 브로드캐스트하고, manual 공급자는
        ``_await_manual_input`` 으로 유저 입력을 격리 대기한다. 실패 시 예외를 그대로
        전파한다 — 호출자(_collect_*)가 우아한 부분 실패 수용으로 변환한다.
        """
        if agent.provider is ModelProvider.MANUAL:
            content = await self._await_manual_input(agent, phase, prior_turns)
            metadata: dict[str, object] = {"provider": "manual"}
        else:
            system, user = self._build_prompt(agent, phase, prior_turns)

            async def on_token(token: str) -> None:
                await self._emit(
                    WSMessageType.TOKEN_STREAM,
                    {"agent_id": agent.agent_id, "phase": phase.value, "token": token},
                )

            content = await self._invoke_agent(agent, system, user, on_token=on_token)
            metadata = {"provider": agent.get_provider().value, "model": agent.model}

        if not content.strip():
            raise DiscussionError("응답이 비어 있습니다.")
        return AgentTurn(
            agent_id=agent.agent_id,
            phase=phase,
            content=content.strip(),
            metadata=metadata,
        )

    def _failure_turn(
        self, agent: AgentConfig, phase: DiscussionPhase, exc: BaseException
    ) -> AgentTurn:
        """실패한 에이전트의 발언칸에 적재할 시스템 경고 턴을 만든다.

        다음 단계 에이전트들이 발언 누락으로 인한 파싱 오류 없이 토론을 이어갈
        수 있도록, 발언 자체를 비우지 않고 경고 텍스트로 채운다.
        """
        warning = (
            f"[시스템 경고: 에이전트 {agent.agent_id}의 응답 생성 실패 - {exc}]"
        )
        return AgentTurn(
            agent_id=agent.agent_id,
            phase=phase,
            content=warning,
            metadata={"failed": True, "error": repr(exc)},
        )

    # ======================================================================
    # 수동(manual) 공급자 — 복붙 터널
    # ======================================================================
    @staticmethod
    def _manual_key(agent_id: str, phase: DiscussionPhase) -> str:
        """수동 입력 대기 Future 의 격리 키 — 에이전트·단계 조합당 유일."""
        return f"{agent_id}::{phase.value}"

    def _agent_by_id(self, agent_id: str) -> AgentConfig:
        """agent_id 로 AgentConfig 를 찾는다. 없으면 DiscussionError."""
        for agent in self.state.agents:
            if agent.agent_id == agent_id:
                return agent
        raise DiscussionError(f"에이전트 '{agent_id}' 를 찾을 수 없습니다.")

    def _llm_agent(self) -> AgentConfig:
        """요약/합성 등 시스템 LLM 호출에 쓸 에이전트 — 첫 번째 비-manual 에이전트."""
        for agent in self.state.agents:
            if agent.provider is not ModelProvider.MANUAL:
                return agent
        return self.state.agents[0]

    async def _await_manual_input(
        self,
        agent: AgentConfig,
        phase: DiscussionPhase,
        prior_turns: list[AgentTurn],
    ) -> str:
        """수동 에이전트의 턴 — API 를 호출하지 않는다.

        딥/일반 복사 페이로드를 만들어 UI 로 보내고, 세션을 PENDING_MANUAL_INPUT
        으로 전환한 뒤 (에이전트·단계) 전용 Future 로 유저 응답을 격리 대기한다.
        동시 단계에서 다른 API 에이전트의 스트리밍과 섞여도 각 수동 턴은 자기
        Future 만 기다리므로 턴 제어가 꼬이지 않는다.
        """
        key = self._manual_key(agent.agent_id, phase)
        waiter: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._manual_waiters[key] = waiter

        deep_copy = self.generate_deep_copy(agent.agent_id, phase, prior_turns)
        general_copy = self.generate_general_copy(agent.agent_id, phase, prior_turns)

        async with self._state_lock:
            self.state.status = DiscussionStatus.PENDING_MANUAL_INPUT
            self.state.touch()
        await self._emit(
            WSMessageType.MANUAL_INPUT_REQUIRED,
            {
                "agent_id": agent.agent_id,
                "phase": phase.value,
                "deep_copy": deep_copy,
                "general_copy": general_copy,
            },
        )
        await self._checkpoint()
        logger.info("수동 입력 대기: 에이전트 %s · %s", agent.agent_id, phase.value)
        try:
            return await waiter
        finally:
            self._manual_waiters.pop(key, None)

    def submit_manual_response(
        self, agent_id: str, phase: DiscussionPhase, content: str
    ) -> None:
        """수동 에이전트의 응답을 주입한다 — 대기 Future 를 해제해 파이프라인 재구동.

        대기 중인 요청이 없으면 ``InvalidStateTransition``.
        """
        key = self._manual_key(agent_id, phase)
        waiter = self._manual_waiters.get(key)
        if waiter is None or waiter.done():
            raise InvalidStateTransition(
                f"대기 중인 수동 입력 요청이 없습니다 "
                f"(agent={agent_id}, phase={phase.value})."
            )
        waiter.set_result(content)

    # ======================================================================
    # 동적 프롬프트 조립
    # ======================================================================
    def _build_prompt(
        self,
        agent: AgentConfig,
        phase: DiscussionPhase,
        prior_turns: list[AgentTurn],
        force_full: bool = False,
    ) -> tuple[str, str]:
        """[공통규칙]+[페르소나] -> system, [맥락]+[단계지침] -> user 로 조립한다.

        ``force_full=True`` 이면 LTM 압축을 끄고 모든 단계 원본 로그를 포함한다
        (딥 카피용).

        Returns:
            ``(system_prompt, user_prompt)`` 튜플.
        """
        system = (
            _COMMON_RULES.format(topic=self.state.topic)
            + f"\n\n[너의 페르소나]\n{agent.persona_prompt}"
        )

        instruction = _PHASE_INSTRUCTIONS[phase]
        # 순차 단계에서 선행 의견이 있으면 중복 회피 개정 지침을 동적 삽입.
        if phase in _SEQUENTIAL_PHASES and prior_turns:
            instruction = f"{instruction}\n\n{_SEQUENTIAL_REVISION_HINT}"

        history = self._render_history(phase, prior_turns, force_full=force_full)
        user_sections = [f"[토론 주제]\n{self.state.topic}"]
        if history:
            user_sections.append(history)
        user_sections.append(f"[지금 너({agent.name})가 수행할 작업]\n{instruction}")
        return system, "\n\n".join(user_sections)

    def _render_history(
        self,
        current_phase: DiscussionPhase,
        prior_turns: list[AgentTurn],
        force_full: bool = False,
    ) -> str:
        """이전 단계 발언 + 유저 개입 + (순차) 선행 의견을 텍스트로 렌더링한다.

        콘텍스트 압축(LTM): 현재 단계가 3단계 이후이면, 수만 토큰에 달할 수 있는
        1·2단계 원본 로그 대신 ``phase_summaries`` 의 요약 메트릭스를 경량 주입한다.
        ``force_full=True`` 이면 압축을 끄고 모든 단계를 원본으로 렌더링한다.
        유저 개입은 인간 진행자를 뜻하는 '참가자 H' 로 치환해 주입한다.
        """
        name_of = {a.agent_id: a.name for a in self.state.agents}
        summary_of = {s.phase: s for s in self.state.phase_summaries}
        lines: list[str] = []

        # 시작 전(after_phase=None) 진행자 개입.
        pre = [iv for iv in self.state.user_interventions if iv.after_phase is None]
        if pre:
            lines.append("== 진행자 사전 지시 ==")
            lines.extend(f"[참가자 H] {iv.message}" for iv in pre)

        current_idx = PHASE_SEQUENCE.index(current_phase)
        # 3단계(idx 2) 이후이면 1·2단계(idx 0·1)를 요약 메트릭스로 압축한다.
        use_ltm = current_idx >= 2 and not force_full
        if use_ltm:
            logger.info(
                "콘텍스트 압축(LTM): %s 프롬프트에 1·2단계를 요약 메트릭스로 주입",
                current_phase.value,
            )
        for past in PHASE_SEQUENCE[:current_idx]:
            past_idx = PHASE_SEQUENCE.index(past)
            compress = use_ltm and past_idx <= 1 and past in summary_of
            if compress:
                # LTM: 원본 발언 로그 대신 경량 요약 메트릭스를 주입.
                lines.append(self._render_phase_summary(past, summary_of[past]))
            else:
                # 원본 발언 로그를 그대로 주입.
                turns = self.state.record_for_phase(past)
                if turns:
                    lines.append(f"== {_PHASE_LABELS[past]} ==")
                    for turn in turns:
                        speaker = name_of.get(turn.agent_id, turn.agent_id)
                        lines.append(f"[{speaker}] {turn.content}")
            # 해당 단계 직후 진행자 개입.
            after = [
                iv for iv in self.state.user_interventions if iv.after_phase is past
            ]
            if after:
                lines.append(f"-- {_PHASE_LABELS[past]} 이후 진행자 개입 --")
                lines.extend(f"[참가자 H] {iv.message}" for iv in after)

        # 순차 단계: 이번 단계에서 먼저 제출된 동료 의견.
        if prior_turns:
            lines.append("== 이번 단계 선행 의견 ==")
            for turn in prior_turns:
                speaker = name_of.get(turn.agent_id, turn.agent_id)
                lines.append(f"[{speaker}] {turn.content}")

        return "\n".join(lines)

    def _render_phase_summary(
        self, phase: DiscussionPhase, summary: PhaseSummary
    ) -> str:
        """단계 요약 메트릭스(LTM)를 경량 텍스트로 렌더링한다.

        원본 발언 전문 대신 에이전트별 주장 메트릭스(초기주장/현재기조/입장변화)
        + 주요 쟁점 + 합의 근접도만 담아 후속 단계 프롬프트의 토큰 사용량을 줄인다.
        """
        name_of = {a.agent_id: a.name for a in self.state.agents}
        lines = [f"== {_PHASE_LABELS[phase]} [요약 메트릭스 · LTM] =="]
        for st in summary.agent_summaries:
            speaker = name_of.get(st.agent_id, st.agent_id)
            parts: list[str] = []
            if st.initial_claim:
                parts.append(f"초기주장: {st.initial_claim}")
            if st.current_stance:
                parts.append(f"현재기조: {st.current_stance}")
            if st.stance_shift:
                parts.append(f"입장변화: {st.stance_shift}")
            lines.append(f"[{speaker}] " + (" | ".join(parts) or "(요약 데이터 없음)"))
        if summary.key_conflicts:
            lines.append("주요 쟁점: " + " · ".join(summary.key_conflicts))
        lines.append(f"합의 근접도: {summary.convergence_score:.0%}")
        return "\n".join(lines)

    def _render_delta(
        self, current_phase: DiscussionPhase, prior_turns: list[AgentTurn]
    ) -> str:
        """직전 단계 발언 + 그 직후 진행자 개입 + (순차) 이번 단계 선행 의견만.

        일반 복사용 — 압축 메모리와 시스템 프롬프트를 제외한 '신규 델타'만 담는다.
        """
        name_of = {a.agent_id: a.name for a in self.state.agents}
        lines: list[str] = []
        current_idx = PHASE_SEQUENCE.index(current_phase)
        if current_idx >= 1:
            prev = PHASE_SEQUENCE[current_idx - 1]
            turns = self.state.record_for_phase(prev)
            if turns:
                lines.append(f"== 직전 단계({_PHASE_LABELS[prev]}) 신규 발언 ==")
                for turn in turns:
                    speaker = name_of.get(turn.agent_id, turn.agent_id)
                    lines.append(f"[{speaker}] {turn.content}")
            after = [
                iv for iv in self.state.user_interventions if iv.after_phase is prev
            ]
            if after:
                lines.append("-- 직후 진행자 개입 --")
                lines.extend(f"[참가자 H] {iv.message}" for iv in after)
        if prior_turns:
            lines.append("== 이번 단계 선행 의견 ==")
            for turn in prior_turns:
                speaker = name_of.get(turn.agent_id, turn.agent_id)
                lines.append(f"[{speaker}] {turn.content}")
        return "\n".join(lines)

    def generate_deep_copy(
        self,
        agent_id: str,
        phase: DiscussionPhase,
        prior_turns: Optional[list[AgentTurn]] = None,
    ) -> str:
        """딥 카피: 시스템 프롬프트 + 전체 원본 이력 전문 — 새 LLM 세션 붙여넣기용.

        LTM 압축을 끄고(force_full) 1~직전 단계의 모든 원본 발언을 포함한다.
        """
        agent = self._agent_by_id(agent_id)
        system, user = self._build_prompt(
            agent, phase, list(prior_turns or []), force_full=True
        )
        return (
            "[복사 유형] 딥 카피 — 새 대화 세션에 그대로 붙여넣으세요.\n"
            "================ SYSTEM ================\n"
            f"{system}\n"
            "================ USER ==================\n"
            f"{user}"
        )

    def generate_general_copy(
        self,
        agent_id: str,
        phase: DiscussionPhase,
        prior_turns: Optional[list[AgentTurn]] = None,
    ) -> str:
        """일반 복사: 압축 메모리·시스템 프롬프트 제외, 직전 단계 신규 델타 맥락만.

        이미 앞 맥락을 학습한 진행 중 LLM 세션에 이어 붙이는 용도.
        """
        agent = self._agent_by_id(agent_id)
        prior = list(prior_turns or [])
        delta = self._render_delta(phase, prior)
        instruction = _PHASE_INSTRUCTIONS[phase]
        if phase in _SEQUENTIAL_PHASES and prior:
            instruction = f"{instruction}\n\n{_SEQUENTIAL_REVISION_HINT}"
        sections = ["[복사 유형] 일반 복사 — 진행 중인 대화 세션에 이어 붙이세요."]
        if delta:
            sections.append(delta)
        sections.append(f"[지금 너({agent.name})가 수행할 작업]\n{instruction}")
        return "\n\n".join(sections)

    # ======================================================================
    # LLM 호출 — 공급자 분기 (풀에서 클라이언트 재사용)
    # ======================================================================
    async def _invoke_agent(
        self,
        agent: AgentConfig,
        system: str,
        user: str,
        on_token: Optional[TokenCallback] = None,
    ) -> str:
        """에이전트의 공급자에 따라 알맞은 스트리밍 LLM 호출로 분기한다.

        클라이언트는 ``LLMClientPool`` 에서 재사용한다(매 호출 신규 생성 X).
        """
        provider = agent.get_provider()
        client = self._pool.get(provider)
        if provider is ModelProvider.OPENAI:
            return await _call_openai(
                client, agent.model, system, user,
                agent.temperature, agent.max_tokens, on_token,
            )
        if provider is ModelProvider.ANTHROPIC:
            return await _call_anthropic(
                client, agent.model, system, user,
                agent.temperature, agent.max_tokens, on_token,
            )
        if provider is ModelProvider.OLLAMA:
            return await _call_ollama(
                client, agent.model, system, user,
                agent.temperature, agent.max_tokens, on_token,
            )
        raise DiscussionError(f"미지원 LLM 공급자: {provider}")

    # ======================================================================
    # 단계 요약
    # ======================================================================
    async def _summarize_phase(
        self, phase: DiscussionPhase, turns: list[AgentTurn]
    ) -> PhaseSummary:
        """단계 요약 메트릭스를 생성한다. 실패해도 파이프라인을 멈추지 않는다."""
        if phase is DiscussionPhase.PHASE_5_CONCLUSION and self.state.force_consensus:
            # 합의 도출 분기 — 별도 LLM 분석 없이 '합의 완료' 요약.
            return PhaseSummary(phase=phase, convergence_score=1.0, key_conflicts=[])
        if not turns:
            return PhaseSummary(phase=phase)
        try:
            return await self._llm_summarize(phase, turns)
        except Exception as exc:  # noqa: BLE001 - 요약 실패는 비치명적
            logger.warning("단계 %s 요약 생성 실패: %r", phase.value, exc)
            return PhaseSummary(
                phase=phase, key_conflicts=[f"[요약 생성 실패: {exc}]"]
            )

    async def _llm_summarize(
        self, phase: DiscussionPhase, turns: list[AgentTurn]
    ) -> PhaseSummary:
        """LLM 에 단계 발언을 분석시켜 ``PhaseSummary`` 를 구성한다."""
        name_of = {a.agent_id: a.name for a in self.state.agents}
        transcript = "\n".join(
            f"[{name_of.get(t.agent_id, t.agent_id)} ({t.agent_id})] {t.content}"
            for t in turns
        )
        agent_ids = [a.agent_id for a in self.state.agents]
        system = (
            "너는 토론 분석가다. 요청한 JSON 객체만 출력하고 다른 설명은 하지 않는다."
        )
        user = (
            f"다음은 '{_PHASE_LABELS.get(phase, phase.value)}' 단계의 발언이다.\n"
            f"{transcript}\n\n"
            "아래 JSON 스키마에 맞춰 정확히 응답하라:\n"
            '{"agent_summaries": [{"agent_id": "...", "initial_claim": "...", '
            '"current_stance": "...", "stance_shift": "..."}], '
            '"key_conflicts": ["..."], "convergence_score": 0.0}\n'
            f"- agent_id 는 반드시 다음 중 하나: {agent_ids}\n"
            "- convergence_score 는 0.0(완전 대립)~1.0(완전 합의) 사이 실수.\n"
            "- 모든 텍스트는 한국어로 작성."
        )
        # 요약 분석에는 첫 번째 비-manual 에이전트의 모델/공급자를 재사용한다.
        raw = await self._invoke_agent(self._llm_agent(), system, user)
        data = _extract_json(raw)

        summaries: list[AgentStanceSummary] = []
        for item in data.get("agent_summaries", []):
            if not isinstance(item, dict) or "agent_id" not in item:
                continue
            summaries.append(
                AgentStanceSummary(
                    agent_id=str(item.get("agent_id", "")),
                    initial_claim=str(item.get("initial_claim", "")),
                    current_stance=str(item.get("current_stance", "")),
                    stance_shift=str(item.get("stance_shift", "")),
                )
            )
        score = max(0.0, min(1.0, float(data.get("convergence_score", 0.0) or 0.0)))
        conflicts = [str(c) for c in data.get("key_conflicts", []) if c]
        return PhaseSummary(
            phase=phase,
            agent_summaries=summaries,
            key_conflicts=conflicts,
            convergence_score=score,
        )

    # ======================================================================
    # 5단계 합의안 도출 (force_consensus=True)
    # ======================================================================
    async def _synthesize_consensus(self) -> str:
        """4단계까지의 토론을 종합해 단일 최종 합의안 문서를 생성한다(스트리밍)."""
        history = self._render_history(DiscussionPhase.PHASE_5_CONCLUSION, [])
        system = (
            "너는 다자 토론의 중립적 합의 조정자다. 특정 참가자 편을 들지 않고, "
            "모든 참가자가 받아들일 수 있는 공통분모를 찾는다."
        )
        user = (
            f"[토론 주제]\n{self.state.topic}\n\n"
            f"[토론 전체 기록]\n{history}\n\n"
            "[지시] 위 토론, 특히 각 참가자의 4단계 수정 입장을 종합하여 모든 "
            "참가자가 동의할 수 있는 '단 하나의 최종 합의안'을 도출하라. 합의안은 "
            "(1) 합의 요지, (2) 합의에 이른 근거, (3) 남은 단서·전제 조건 순서의 "
            "문서로 작성하라."
        )
        phase5 = DiscussionPhase.PHASE_5_CONCLUSION

        async def on_token(token: str) -> None:
            await self._emit(
                WSMessageType.TOKEN_STREAM,
                {
                    "agent_id": CONSENSUS_SPEAKER_ID,
                    "phase": phase5.value,
                    "token": token,
                },
            )

        # 합의안 합성에는 첫 번째 비-manual 에이전트의 모델/공급자를 재사용한다.
        return await self._invoke_agent(
            self._llm_agent(), system, user, on_token=on_token
        )

    # ======================================================================
    # 종료 / 오류 / 브로드캐스트
    # ======================================================================
    async def _finalize(self) -> None:
        """5단계까지 정상 종료 처리."""
        async with self._state_lock:
            self.state.current_phase = DiscussionPhase.COMPLETED
            self.state.status = DiscussionStatus.COMPLETED
            self.state.touch()
        payload: dict[str, object] = {"discussion_id": self.state.discussion_id}
        if self.state.final_joint_agreement is not None:
            payload["final_joint_agreement"] = self.state.final_joint_agreement
        await self._emit(WSMessageType.DISCUSSION_COMPLETED, payload)
        await self._checkpoint()

    async def _set_error(self, message: str) -> None:
        """오류 상태로 전이한다."""
        async with self._state_lock:
            self.state.status = DiscussionStatus.ERROR
            self.state.error = message
            self.state.touch()
        await self._emit(WSMessageType.ERROR, {"message": message})
        await self._checkpoint()

    async def _emit_turn(self, turn: AgentTurn) -> None:
        """에이전트 발언 1건 완료를 알린다(최종 텍스트 — 실패 턴 포함)."""
        await self._emit(
            WSMessageType.AGENT_TURN, {"turn": turn.model_dump(mode="json")}
        )

    async def _emit(
        self, msg_type: WSMessageType, payload: dict[str, object]
    ) -> None:
        """브로드캐스트 콜백이 등록되어 있으면 WS 메시지를 전송한다."""
        if self._broadcast_cb is None:
            return
        await self._broadcast_cb(WSMessage(type=msg_type, payload=payload))

    async def _checkpoint(self) -> None:
        """현재 상태를 영속성 레이어에 저장한다 (persist 콜백이 있으면).

        영속화 실패는 토론 진행을 막지 않도록 비치명적으로 흡수한다.
        """
        if self._persist_cb is None:
            return
        try:
            await self._persist_cb(self.state)
        except Exception as exc:  # noqa: BLE001 - 영속화 실패는 비치명적
            logger.warning("상태 영속화 실패: %r", exc)
