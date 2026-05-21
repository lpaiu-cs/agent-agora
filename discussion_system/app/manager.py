"""이벤트 구동형 무상태(stateless) 토론 오케스트레이션 + 멀티 공급자 LLM 연동.

phase-6: 토론 전체를 점유하던 거대한 ``_run_pipeline`` while/for 루프를 제거하고,
단일 진입점 ``Orchestrator.process_event(discussion_id, event, payload)`` 로 재편했다.

  * 토론 상태는 100% DB(`database`)에 있다. process_event 는 매 호출마다
    DB 에서 상태를 로드 -> 전이 -> 저장 -> 즉시 종료한다.
  * 단계 사이(유저 게이트)·수동 입력 대기 중에는 어떤 코루틴/Future 도
    메모리에 점유되지 않는다 — 그 대기는 DB status(`WAITING_FOR_USER` /
    `PENDING_MANUAL_INPUT`)로만 표현되고, 후속 이벤트(HTTP 요청·복구)가 재개한다.
  * 상태 저장은 `version` 컬럼 낙관적 락으로 보호되며, 충돌 시 재시도한다.
  * `recover()` 가 서버 기동 시 DB 를 스캔해 중단된 세션을 안전 재기동한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Optional

from . import database
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

#: 브로드캐스트 콜백: (discussion_id, WSMessage) 를 받아 (비동기로) 전송한다.
BroadcastCallback = Callable[[str, WSMessage], Awaitable[None]]
#: 토큰 콜백: 스트리밍 청크 1개를 받아 (비동기로) 처리하는 함수.
TokenCallback = Callable[[str], Awaitable[None]]

#: 순차 포스팅으로 진행되는 단계 (후순위 에이전트가 선행 의견을 맥락으로 본다).
_SEQUENTIAL_PHASES: frozenset[DiscussionPhase] = frozenset(
    {DiscussionPhase.PHASE_1_OPINION, DiscussionPhase.PHASE_2_CRITIQUE}
)

#: 합의안 합성 발언에 쓰는 가상 발화자 ID (5단계 force_consensus=True).
CONSENSUS_SPEAKER_ID = "consensus"

#: 낙관적 락 충돌 시 commit 재시도 횟수.
_COMMIT_RETRIES = 6


def _positive_int_env(name: str, default: int) -> int:
    """양의 정수 환경변수를 읽는다 — 미설정·형식 오류·0 이하이면 ``default`` 로 흡수."""
    try:
        parsed = int(os.getenv(name, ""))
    except ValueError:
        return default
    return parsed if parsed >= 1 else default


#: 동시에 진행할 수 있는 무거운 LLM 호출(외부 API·추론)의 최대 개수.
#: 토론 수백 개가 한꺼번에 이벤트를 발사해도, 이 수치를 넘는 LLM 호출은 세마포어
#: 에서 대기한다 — 동시 코루틴 폭주로 인한 CPU 스파이크와 낙관적 락 경합 폭증을
#: 막는 백프레셔 장치다. ``AGORA_MAX_CONCURRENT_LLM`` 환경변수로 조정한다.
_MAX_CONCURRENT_LLM = _positive_int_env("AGORA_MAX_CONCURRENT_LLM", 8)


# ===========================================================================
# 예외 / 이벤트
# ===========================================================================
class DiscussionError(RuntimeError):
    """토론 오케스트레이션 관련 일반 오류."""


class InvalidStateTransition(DiscussionError):
    """현재 상태에서 허용되지 않는 전이를 시도했을 때 발생 (엔드포인트가 409 로 변환)."""


class PipelineEvent(str, Enum):
    """`process_event` 의 단일 진입점이 받는 이벤트 종류."""

    START = "start"                      # 토론 생성 직후 — 1단계 기동
    ADVANCE = "advance"                  # 유저가 다음 단계 진입 승인
    MANUAL_RESPONSE = "manual_response"   # 수동 에이전트 응답 수신
    RECOVER = "recover"                   # 서버 재기동 후 크래시 복구


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


def _next_phase(phase: DiscussionPhase) -> Optional[DiscussionPhase]:
    """다음 단계를 반환한다. 마지막(5)단계이면 None."""
    idx = PHASE_SEQUENCE.index(phase)
    return PHASE_SEQUENCE[idx + 1] if idx + 1 < len(PHASE_SEQUENCE) else None


# ===========================================================================
# 멀티 공급자 LLM 호출 레이어 (스트리밍)
# ===========================================================================
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
    """공급자별 비동기 LLM 클라이언트를 1회 생성 후 재사용하는 앱 레벨 풀."""

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
    client: object, model: str, system: str, user: str,
    temperature: float, max_tokens: int, on_token: Optional[TokenCallback],
) -> str:
    """OpenAI Chat Completions 스트리밍 호출. 누적 텍스트를 반환."""
    stream = await client.chat.completions.create(  # type: ignore[attr-defined]
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature, max_tokens=max_tokens, stream=True,
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
    client: object, model: str, system: str, user: str,
    temperature: float, max_tokens: int, on_token: Optional[TokenCallback],
) -> str:
    """Anthropic Messages 스트리밍 호출. 누적 텍스트를 반환."""
    parts: list[str] = []
    async with client.messages.stream(  # type: ignore[attr-defined]
        model=model, system=system,
        messages=[{"role": "user", "content": user}],
        temperature=temperature, max_tokens=max_tokens,
    ) as stream:
        async for text in stream.text_stream:
            if text:
                parts.append(text)
                if on_token is not None:
                    await on_token(text)
    return "".join(parts).strip()


async def _call_ollama(
    client: object, model: str, system: str, user: str,
    temperature: float, max_tokens: int, on_token: Optional[TokenCallback],
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
# 순수 헬퍼 — 상태(state)를 인자로 받는 무상태 함수들
# ===========================================================================
def _agent_by_id(state: DiscussionState, agent_id: str) -> AgentConfig:
    """agent_id 로 AgentConfig 를 찾는다. 없으면 DiscussionError."""
    for agent in state.agents:
        if agent.agent_id == agent_id:
            return agent
    raise DiscussionError(f"에이전트 '{agent_id}' 를 찾을 수 없습니다.")


def _llm_agent(state: DiscussionState) -> AgentConfig:
    """요약/합성 등 시스템 LLM 호출에 쓸 에이전트 — 첫 번째 비-manual 에이전트."""
    for agent in state.agents:
        if agent.provider is not ModelProvider.MANUAL:
            return agent
    return state.agents[0]


def _failure_turn(
    agent: AgentConfig, phase: DiscussionPhase, exc: BaseException
) -> AgentTurn:
    """실패한 에이전트의 발언칸에 적재할 시스템 경고 턴 (우아한 부분 실패 수용)."""
    return AgentTurn(
        agent_id=agent.agent_id,
        phase=phase,
        content=f"[시스템 경고: 에이전트 {agent.agent_id}의 응답 생성 실패 - {exc}]",
        metadata={"failed": True, "error": repr(exc)},
    )


def _render_phase_summary(
    state: DiscussionState, phase: DiscussionPhase, summary: PhaseSummary
) -> str:
    """단계 요약 메트릭스(LTM)를 경량 텍스트로 렌더링한다."""
    name_of = {a.agent_id: a.name for a in state.agents}
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


def _render_history(
    state: DiscussionState,
    current_phase: DiscussionPhase,
    prior_turns: list[AgentTurn],
    force_full: bool = False,
) -> str:
    """이전 단계 발언 + 유저 개입 + (순차) 선행 의견을 텍스트로 렌더링한다.

    콘텍스트 압축(LTM): 3단계 이후이면 1·2단계 원본 로그 대신 ``phase_summaries``
    의 요약 메트릭스를 경량 주입한다. ``force_full=True`` 이면 압축을 끈다.
    """
    name_of = {a.agent_id: a.name for a in state.agents}
    summary_of = {s.phase: s for s in state.phase_summaries}
    lines: list[str] = []

    pre = [iv for iv in state.user_interventions if iv.after_phase is None]
    if pre:
        lines.append("== 진행자 사전 지시 ==")
        lines.extend(f"[참가자 H] {iv.message}" for iv in pre)

    current_idx = PHASE_SEQUENCE.index(current_phase)
    use_ltm = current_idx >= 2 and not force_full
    for past in PHASE_SEQUENCE[:current_idx]:
        past_idx = PHASE_SEQUENCE.index(past)
        if use_ltm and past_idx <= 1 and past in summary_of:
            lines.append(_render_phase_summary(state, past, summary_of[past]))
        else:
            turns = state.record_for_phase(past)
            if turns:
                lines.append(f"== {_PHASE_LABELS[past]} ==")
                for turn in turns:
                    speaker = name_of.get(turn.agent_id, turn.agent_id)
                    lines.append(f"[{speaker}] {turn.content}")
        after = [iv for iv in state.user_interventions if iv.after_phase is past]
        if after:
            lines.append(f"-- {_PHASE_LABELS[past]} 이후 진행자 개입 --")
            lines.extend(f"[참가자 H] {iv.message}" for iv in after)

    if prior_turns:
        lines.append("== 이번 단계 선행 의견 ==")
        for turn in prior_turns:
            speaker = name_of.get(turn.agent_id, turn.agent_id)
            lines.append(f"[{speaker}] {turn.content}")
    return "\n".join(lines)


def _render_delta(
    state: DiscussionState,
    current_phase: DiscussionPhase,
    prior_turns: list[AgentTurn],
) -> str:
    """직전 단계 발언 + 그 직후 진행자 개입 + (순차) 이번 단계 선행 의견만."""
    name_of = {a.agent_id: a.name for a in state.agents}
    lines: list[str] = []
    current_idx = PHASE_SEQUENCE.index(current_phase)
    if current_idx >= 1:
        prev = PHASE_SEQUENCE[current_idx - 1]
        turns = state.record_for_phase(prev)
        if turns:
            lines.append(f"== 직전 단계({_PHASE_LABELS[prev]}) 신규 발언 ==")
            for turn in turns:
                speaker = name_of.get(turn.agent_id, turn.agent_id)
                lines.append(f"[{speaker}] {turn.content}")
        after = [iv for iv in state.user_interventions if iv.after_phase is prev]
        if after:
            lines.append("-- 직후 진행자 개입 --")
            lines.extend(f"[참가자 H] {iv.message}" for iv in after)
    if prior_turns:
        lines.append("== 이번 단계 선행 의견 ==")
        for turn in prior_turns:
            speaker = name_of.get(turn.agent_id, turn.agent_id)
            lines.append(f"[{speaker}] {turn.content}")
    return "\n".join(lines)


def _build_prompt(
    state: DiscussionState,
    agent: AgentConfig,
    phase: DiscussionPhase,
    prior_turns: list[AgentTurn],
    force_full: bool = False,
) -> tuple[str, str]:
    """[공통규칙]+[페르소나] -> system, [맥락]+[단계지침] -> user 로 조립한다."""
    system = (
        _COMMON_RULES.format(topic=state.topic)
        + f"\n\n[너의 페르소나]\n{agent.persona_prompt}"
    )
    instruction = _PHASE_INSTRUCTIONS[phase]
    if phase in _SEQUENTIAL_PHASES and prior_turns:
        instruction = f"{instruction}\n\n{_SEQUENTIAL_REVISION_HINT}"
    history = _render_history(state, phase, prior_turns, force_full=force_full)
    user_sections = [f"[토론 주제]\n{state.topic}"]
    if history:
        user_sections.append(history)
    user_sections.append(f"[지금 너({agent.name})가 수행할 작업]\n{instruction}")
    return system, "\n\n".join(user_sections)


def generate_deep_copy(
    state: DiscussionState,
    agent_id: str,
    phase: DiscussionPhase,
    prior_turns: Optional[list[AgentTurn]] = None,
) -> str:
    """딥 카피: 시스템 프롬프트 + 전체 원본 이력 전문 — 새 LLM 세션 붙여넣기용."""
    agent = _agent_by_id(state, agent_id)
    system, user = _build_prompt(
        state, agent, phase, list(prior_turns or []), force_full=True
    )
    return (
        "[복사 유형] 딥 카피 — 새 대화 세션에 그대로 붙여넣으세요.\n"
        "================ SYSTEM ================\n"
        f"{system}\n"
        "================ USER ==================\n"
        f"{user}"
    )


def generate_general_copy(
    state: DiscussionState,
    agent_id: str,
    phase: DiscussionPhase,
    prior_turns: Optional[list[AgentTurn]] = None,
) -> str:
    """일반 복사: 압축 메모리·시스템 프롬프트 제외, 직전 단계 신규 델타 맥락만."""
    agent = _agent_by_id(state, agent_id)
    prior = list(prior_turns or [])
    delta = _render_delta(state, phase, prior)
    instruction = _PHASE_INSTRUCTIONS[phase]
    if phase in _SEQUENTIAL_PHASES and prior:
        instruction = f"{instruction}\n\n{_SEQUENTIAL_REVISION_HINT}"
    sections = ["[복사 유형] 일반 복사 — 진행 중인 대화 세션에 이어 붙이세요."]
    if delta:
        sections.append(delta)
    sections.append(f"[지금 너({agent.name})가 수행할 작업]\n{instruction}")
    return "\n\n".join(sections)


# ===========================================================================
# 무상태 오케스트레이터
# ===========================================================================
class Orchestrator:
    """이벤트 구동형 무상태 오케스트레이터.

    토론별 상태(`DiscussionState`)를 인메모리에 들고 있지 않는다 — 인프라 의존성
    (LLM 풀, 브로드캐스트 콜백)만 보유한다. 모든 전이는 ``process_event`` 한
    진입점을 통하며, 매번 DB 로드 -> 전이 -> 저장 -> 종료한다.
    """

    def __init__(self, pool: LLMClientPool, broadcast: BroadcastCallback) -> None:
        self._pool = pool
        self._broadcast = broadcast
        # 진행 중 전이 태스크 핸들 (GC 방지용 — 토론별 상태가 아니다).
        self._inflight: set[asyncio.Task] = set()
        # 무거운 LLM 호출 동시성 상한 — 토론 수가 폭증해도 외부 API·추론을 실제로
        # 수행하는 코루틴 수를 _MAX_CONCURRENT_LLM 개로 엄격히 제한한다 (백프레셔).
        # 초과분은 이 세마포어에서 대기하며, trigger 의 태스크 자체는 막지 않는다.
        self._llm_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_LLM)

    # ----- 공개: 엔드포인트/복구가 호출 -----
    def trigger(
        self, discussion_id: str, event: PipelineEvent, payload: Optional[dict] = None
    ) -> None:
        """``process_event`` 를 백그라운드 태스크로 발사하고 즉시 반환한다.

        HTTP 워커는 이 호출 후 곧바로 응답을 돌려주고, 전이는 비동기로 진행된다.
        """
        task = asyncio.create_task(self._safe_process(discussion_id, event, payload))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _safe_process(
        self, discussion_id: str, event: PipelineEvent, payload: Optional[dict]
    ) -> None:
        try:
            await self.process_event(discussion_id, event, payload)
        except InvalidStateTransition as exc:
            # 레이스로 인한 상태 불일치 — 비치명적, 오류 상태로 만들지 않는다.
            logger.warning("전이 무시(%s/%s): %s", discussion_id, event.value, exc)
        except Exception as exc:  # noqa: BLE001 - 그 외 오류는 ERROR 상태로 흡수
            logger.exception("process_event 실패: %s/%s", discussion_id, event.value)
            await self._mark_error(discussion_id, str(exc))

    async def process_event(
        self, discussion_id: str, event: PipelineEvent, payload: Optional[dict] = None
    ) -> None:
        """단일 상태 전이 진입점 — DB 로드 -> 전이 -> 저장 -> 종료."""
        state = await self._load(discussion_id)
        if state is None:
            raise DiscussionError(f"토론 {discussion_id} 를 찾을 수 없습니다.")

        if event is PipelineEvent.START:
            await self._run_phase(discussion_id, PHASE_SEQUENCE[0])
        elif event is PipelineEvent.ADVANCE:
            if state.status is not DiscussionStatus.WAITING_FOR_USER:
                raise InvalidStateTransition(
                    f"advance 는 WAITING_FOR_USER 에서만 가능 (현재 {state.status.value})"
                )
            nxt = _next_phase(state.current_phase)
            if nxt is not None:
                await self._run_phase(discussion_id, nxt)
        elif event is PipelineEvent.MANUAL_RESPONSE:
            await self._on_manual_response(state, payload or {})
        elif event is PipelineEvent.RECOVER:
            await self._on_recover(state)
        else:
            raise DiscussionError(f"알 수 없는 이벤트: {event}")

    async def recover(self) -> dict:
        """서버 기동 시 DB 를 스캔해 중단된 토론을 복구한다 (lifespan startup).

        * RUNNING — 자동 API 연동 중 서버가 죽은 세션. 현재 단계를 멱등 재기동.
        * PENDING_MANUAL_INPUT — 수동 입력 대기 세션. DB status 가 곧 대기 플래그
          이므로 유실되지 않는다 — 재기동 없이 후속 /manual-response 를 수용한다.
        """
        stuck = await database.list_states_by_status(
            ("running", "pending_manual_input")
        )
        running = [s for s in stuck if s.status is DiscussionStatus.RUNNING]
        pending = [s for s in stuck
                   if s.status is DiscussionStatus.PENDING_MANUAL_INPUT]
        for s in pending:
            logger.info(
                "크래시 복구: %s — PENDING_MANUAL_INPUT 유지, /manual-response 수용 준비 완료",
                s.discussion_id,
            )
        for s in running:
            logger.info(
                "크래시 복구: %s — RUNNING(%s) 단계 멱등 재기동",
                s.discussion_id, s.current_phase.value,
            )
            self.trigger(s.discussion_id, PipelineEvent.RECOVER)
        return {"running_recovered": len(running), "pending_preserved": len(pending)}

    async def add_intervention(
        self, discussion_id: str, intervention: UserIntervention
    ) -> None:
        """유저 개입을 상태에 기록한다 (단계 전이는 트리거하지 않음).

        다음 단계 프롬프트 맥락에 '참가자 H' 로 반영된다. 낙관적 락 재시도 시
        중복 적재를 막기 위해 (created_at, message) 로 멱등성을 보장한다.
        """
        def mutate(s: DiscussionState) -> None:
            already = any(
                iv.created_at == intervention.created_at
                and iv.message == intervention.message
                for iv in s.user_interventions
            )
            if not already:
                s.user_interventions.append(intervention)
            s.touch()

        await self._commit(discussion_id, mutate)
        await self._emit(
            discussion_id, WSMessageType.USER_INTERVENTION,
            {"intervention": intervention.model_dump(mode="json")},
        )

    async def emit_manual_input_required_for_socket(
        self, discussion_id: str, websocket
    ) -> None:
        """수동 대기 '2중 방어선' — 특정 WebSocket 1개에 manual_input_required 재전송.

        WS 접속(특히 새로고침·재연결) 직후 호출된다. 세션이 PENDING_MANUAL_INPUT
        이면 아직 발언하지 않은 수동 에이전트별로 딥/일반 복사 페이로드를 만들어
        해당 소켓에만 보낸다 — 복붙 터널 패널이 증발해 세션이 고착되는 것을 막는다.
        best-effort: 어떤 실패도 호출부(WS 핸들러)로 전파하지 않는다.
        """
        try:
            state = await self._load(discussion_id)
            if (state is None
                    or state.status is not DiscussionStatus.PENDING_MANUAL_INPUT):
                return
            phase = state.current_phase
            if phase not in PHASE_SEQUENCE:
                return
            record = state.record_for_phase(phase)
            posted = {t.agent_id for t in record}
            prior = list(record)
            for agent in state.agents:
                if (agent.provider is not ModelProvider.MANUAL
                        or agent.agent_id in posted):
                    continue
                message = WSMessage(
                    type=WSMessageType.MANUAL_INPUT_REQUIRED,
                    payload={
                        "agent_id": agent.agent_id,
                        "phase": phase.value,
                        "deep_copy": generate_deep_copy(
                            state, agent.agent_id, phase, prior),
                        "general_copy": generate_general_copy(
                            state, agent.agent_id, phase, prior),
                    },
                )
                await websocket.send_json(message.model_dump(mode="json"))
        except Exception as exc:  # noqa: BLE001 - best-effort 재전송, 실패해도 무시
            logger.warning(
                "manual_input_required 소켓 재전송 실패(%s): %r", discussion_id, exc
            )

    # ----- 전이 핸들러 -----
    async def _on_recover(self, state: DiscussionState) -> None:
        """크래시 복구 전이. RUNNING 단계만 멱등 재기동한다."""
        if state.status is not DiscussionStatus.RUNNING:
            return  # PENDING 등은 DB 상태 그대로가 정상 — 손대지 않는다.
        phase = state.current_phase
        if phase not in PHASE_SEQUENCE:
            return
        # 멱등 재기동: 현재 단계의 부분 발언 기록·요약을 비우고 처음부터 재실행.
        def reset(s: DiscussionState) -> None:
            s.record_for_phase(phase).clear()
            s.phase_summaries[:] = [p for p in s.phase_summaries if p.phase is not phase]
            s.touch()

        await self._commit(state.discussion_id, reset)
        await self._run_phase(state.discussion_id, phase)

    async def _on_manual_response(
        self, state: DiscussionState, payload: dict
    ) -> None:
        """수동 에이전트 응답 주입 — 턴 기록 후 단계 진행을 재개한다."""
        agent_id = str(payload.get("agent_id", ""))
        phase = DiscussionPhase(payload["phase"])
        content = str(payload.get("content", "")).strip()

        if state.status is not DiscussionStatus.PENDING_MANUAL_INPUT:
            raise InvalidStateTransition(
                f"수동 입력은 PENDING_MANUAL_INPUT 에서만 가능 (현재 {state.status.value})"
            )
        agent = _agent_by_id(state, agent_id)
        if agent.provider is not ModelProvider.MANUAL:
            raise InvalidStateTransition(f"에이전트 {agent_id} 는 수동 에이전트가 아닙니다.")
        if phase is not state.current_phase:
            raise InvalidStateTransition(
                f"수동 입력 단계({phase.value})가 현재 단계와 다릅니다."
            )
        if any(t.agent_id == agent_id for t in state.record_for_phase(phase)):
            raise InvalidStateTransition(f"에이전트 {agent_id} 는 이미 응답했습니다.")
        if not content:
            raise InvalidStateTransition("수동 응답 내용이 비어 있습니다.")

        turn = AgentTurn(
            agent_id=agent_id, phase=phase, content=content,
            metadata={"provider": "manual"},
        )
        await self._commit(
            state.discussion_id, lambda s: _record_turns(s, phase, [turn])
        )
        await self._emit_turn(state.discussion_id, turn)
        # 단계 진행 재개 — 다음 에이전트(들) 처리 또는 단계 종료.
        await self._advance_phase_progress(state.discussion_id, phase)

    # ----- 단계 실행 -----
    async def _run_phase(self, discussion_id: str, phase: DiscussionPhase) -> None:
        """단계 진입 — status RUNNING 마킹 후 진행 가능한 데까지 수행한다."""
        await self._commit(discussion_id, lambda s: _mark_phase_start(s, phase))
        await self._emit(discussion_id, WSMessageType.PHASE_STARTED,
                         {"phase": phase.value})
        await self._advance_phase_progress(discussion_id, phase)

    async def _advance_phase_progress(
        self, discussion_id: str, phase: DiscussionPhase
    ) -> None:
        """현재 단계를 '인간 입력 없이 가능한 데까지' 진행한다.

        * 모든 에이전트가 게시 완료 -> ``_finish_phase``
        * 수동 에이전트 차례 -> ``_enter_pending`` 후 즉시 종료(메모리 반환)
        * API 에이전트 -> 순차 단계는 1명씩, 동시 단계는 일괄 호출
        """
        state = await self._load(discussion_id)
        if state is None:
            return

        if phase is DiscussionPhase.PHASE_5_CONCLUSION and state.force_consensus:
            await self._run_consensus(discussion_id)
            return

        record = state.record_for_phase(phase)
        posted = {t.agent_id for t in record}
        pending = [a for a in state.agents if a.agent_id not in posted]
        if not pending:
            await self._finish_phase(discussion_id, phase)
            return

        if phase in _SEQUENTIAL_PHASES:
            nxt = pending[0]
            if nxt.provider is ModelProvider.MANUAL:
                await self._enter_pending(discussion_id, phase, [nxt])
                return
            turn = await self._do_api_turn(state, phase, nxt, list(record))
            await self._commit(
                discussion_id, lambda s: _record_turns(s, phase, [turn])
            )
            await self._emit_turn(discussion_id, turn)
            # 다음 에이전트로 진행 (재귀 — 깊이 = 에이전트 수).
            await self._advance_phase_progress(discussion_id, phase)
        else:
            api_pending = [a for a in pending
                           if a.provider is not ModelProvider.MANUAL]
            manual_pending = [a for a in pending
                              if a.provider is ModelProvider.MANUAL]
            if api_pending:
                turns = await self._gather_api_turns(state, phase, api_pending)
                await self._commit(
                    discussion_id, lambda s: _record_turns(s, phase, turns)
                )
                for turn in turns:
                    await self._emit_turn(discussion_id, turn)
            if manual_pending:
                await self._enter_pending(discussion_id, phase, manual_pending)
                return
            await self._finish_phase(discussion_id, phase)

    async def _enter_pending(
        self, discussion_id: str, phase: DiscussionPhase,
        manual_agents: list[AgentConfig],
    ) -> None:
        """수동 에이전트 차례 — 복사 페이로드를 UI 로 보내고 PENDING 으로 마킹.

        Future 를 들고 대기하지 않는다 — status 를 DB 에 PENDING_MANUAL_INPUT 으로
        남기고 메모리 자원을 즉시 반환한다. 이후 /manual-response 가 재개한다.
        """
        state = await self._load(discussion_id)
        if state is None:
            return
        prior = list(state.record_for_phase(phase))
        for agent in manual_agents:
            deep = generate_deep_copy(state, agent.agent_id, phase, prior)
            general = generate_general_copy(state, agent.agent_id, phase, prior)
            await self._emit(
                discussion_id, WSMessageType.MANUAL_INPUT_REQUIRED,
                {
                    "agent_id": agent.agent_id, "phase": phase.value,
                    "deep_copy": deep, "general_copy": general,
                },
            )
        await self._commit(
            discussion_id,
            lambda s: _set_status(s, DiscussionStatus.PENDING_MANUAL_INPUT),
        )
        logger.info(
            "토론 %s — 수동 입력 대기 진입 (%s, %d명) — 메모리 반환",
            discussion_id, phase.value, len(manual_agents),
        )

    async def _finish_phase(self, discussion_id: str, phase: DiscussionPhase) -> None:
        """단계 종료 — 요약 생성 -> 게이트(WAITING) 또는 토론 종료(COMPLETED)."""
        state = await self._load(discussion_id)
        if state is None:
            return
        summary = await self._summarize_phase(state, phase,
                                              list(state.record_for_phase(phase)))
        nxt = _next_phase(phase)

        def mutate(s: DiscussionState) -> None:
            if not any(ps.phase is phase for ps in s.phase_summaries):
                s.phase_summaries.append(summary)
            if nxt is None:
                s.current_phase = DiscussionPhase.COMPLETED
                s.status = DiscussionStatus.COMPLETED
            else:
                s.status = DiscussionStatus.WAITING_FOR_USER
            s.touch()

        state = await self._commit(discussion_id, mutate)
        payload: dict[str, object] = {
            "phase": phase.value, "summary": summary.model_dump(mode="json"),
        }
        if state.final_joint_agreement is not None:
            payload["final_joint_agreement"] = state.final_joint_agreement
        await self._emit(discussion_id, WSMessageType.PHASE_COMPLETED, payload)
        if nxt is None:
            await self._emit(
                discussion_id, WSMessageType.DISCUSSION_COMPLETED,
                {"discussion_id": discussion_id,
                 "final_joint_agreement": state.final_joint_agreement},
            )
        else:
            await self._emit(discussion_id, WSMessageType.AWAITING_USER,
                             {"completed_phase": phase.value})

    async def _run_consensus(self, discussion_id: str) -> None:
        """5단계 force_consensus=True — 단일 합의안 문서를 합성한다."""
        state = await self._load(discussion_id)
        if state is None:
            return
        try:
            agreement = await self._synthesize(state)
        except Exception as exc:  # noqa: BLE001 - 우아한 부분 실패 수용
            logger.warning("합의안 생성 실패: %r", exc)
            agreement = f"[시스템 경고: 합의안 생성 실패 - {exc}]"
        await self._commit(
            discussion_id, lambda s: _set_agreement(s, agreement)
        )
        await self._finish_phase(discussion_id, DiscussionPhase.PHASE_5_CONCLUSION)

    # ----- 에이전트 호출 -----
    async def _do_api_turn(
        self, state: DiscussionState, phase: DiscussionPhase,
        agent: AgentConfig, prior_turns: list[AgentTurn],
    ) -> AgentTurn:
        """한 API 에이전트의 턴 — 스트리밍 LLM 호출. 실패는 시스템 경고 턴으로."""
        try:
            system, user = _build_prompt(state, agent, phase, prior_turns)

            async def on_token(token: str) -> None:
                await self._emit(
                    state.discussion_id, WSMessageType.TOKEN_STREAM,
                    {"agent_id": agent.agent_id, "phase": phase.value,
                     "token": token},
                )

            content = await self._invoke_agent(agent, system, user, on_token)
            if not content.strip():
                raise DiscussionError("응답이 비어 있습니다.")
            return AgentTurn(
                agent_id=agent.agent_id, phase=phase, content=content.strip(),
                metadata={"provider": agent.get_provider().value,
                          "model": agent.model},
            )
        except Exception as exc:  # noqa: BLE001 - 우아한 부분 실패 수용
            logger.warning("에이전트 %s 응답 실패: %r", agent.agent_id, exc)
            return _failure_turn(agent, phase, exc)

    async def _gather_api_turns(
        self, state: DiscussionState, phase: DiscussionPhase,
        agents: list[AgentConfig],
    ) -> list[AgentTurn]:
        """동시 단계 — 여러 API 에이전트를 asyncio.gather 로 병렬 호출한다."""
        return list(await asyncio.gather(
            *(self._do_api_turn(state, phase, a, []) for a in agents)
        ))

    async def _invoke_agent(
        self, agent: AgentConfig, system: str, user: str,
        on_token: Optional[TokenCallback] = None,
    ) -> str:
        """공급자별 스트리밍 LLM 호출로 분기한다 (풀에서 클라이언트 재사용).

        실제 네트워크·추론 비용이 드는 ``_call_*`` 호출은 고정 크기 세마포어
        (``self._llm_semaphore``) 안에서만 수행한다 — 토론 수백 개가 한꺼번에
        이벤트를 발사해도 동시에 진행되는 무거운 LLM 호출 수를 엄격히 제한해,
        CPU 폭주와 낙관적 락(StaleStateError) 경합 폭증을 막는다 (구조 검토 ② 교정).
        모든 LLM 경로(에이전트 턴·합의안 합성·단계 요약)가 이 함수를 거치므로,
        여기 한 곳의 세마포어가 전체 추론 동시성의 단일 통제점이 된다.
        """
        provider = agent.get_provider()
        client = self._pool.get(provider)
        async with self._llm_semaphore:
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

    async def _synthesize(self, state: DiscussionState) -> str:
        """4단계까지를 종합해 단일 최종 합의안 문서를 생성한다(스트리밍)."""
        history = _render_history(state, DiscussionPhase.PHASE_5_CONCLUSION, [])
        system = (
            "너는 다자 토론의 중립적 합의 조정자다. 특정 참가자 편을 들지 않고, "
            "모든 참가자가 받아들일 수 있는 공통분모를 찾는다."
        )
        user = (
            f"[토론 주제]\n{state.topic}\n\n[토론 전체 기록]\n{history}\n\n"
            "[지시] 위 토론, 특히 각 참가자의 4단계 수정 입장을 종합하여 모든 "
            "참가자가 동의할 수 있는 '단 하나의 최종 합의안'을 도출하라. 합의안은 "
            "(1) 합의 요지, (2) 합의에 이른 근거, (3) 남은 단서·전제 조건 순서의 "
            "문서로 작성하라."
        )
        did = state.discussion_id

        async def on_token(token: str) -> None:
            await self._emit(did, WSMessageType.TOKEN_STREAM,
                             {"agent_id": CONSENSUS_SPEAKER_ID,
                              "phase": DiscussionPhase.PHASE_5_CONCLUSION.value,
                              "token": token})

        return await self._invoke_agent(_llm_agent(state), system, user, on_token)

    async def _summarize_phase(
        self, state: DiscussionState, phase: DiscussionPhase,
        turns: list[AgentTurn],
    ) -> PhaseSummary:
        """단계 요약 메트릭스를 생성한다. 실패해도 파이프라인을 멈추지 않는다."""
        if phase is DiscussionPhase.PHASE_5_CONCLUSION and state.force_consensus:
            return PhaseSummary(phase=phase, convergence_score=1.0)
        if not turns:
            return PhaseSummary(phase=phase)
        try:
            return await self._llm_summarize(state, phase, turns)
        except Exception as exc:  # noqa: BLE001 - 요약 실패는 비치명적
            logger.warning("단계 %s 요약 생성 실패: %r", phase.value, exc)
            return PhaseSummary(phase=phase, key_conflicts=[f"[요약 생성 실패: {exc}]"])

    async def _llm_summarize(
        self, state: DiscussionState, phase: DiscussionPhase,
        turns: list[AgentTurn],
    ) -> PhaseSummary:
        """LLM 에 단계 발언을 분석시켜 PhaseSummary(주장 메트릭스)를 구성한다."""
        name_of = {a.agent_id: a.name for a in state.agents}
        transcript = "\n".join(
            f"[{name_of.get(t.agent_id, t.agent_id)} ({t.agent_id})] {t.content}"
            for t in turns
        )
        agent_ids = [a.agent_id for a in state.agents]
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
            "- convergence_score 는 0.0~1.0 실수. 모든 텍스트는 한국어."
        )
        raw = await self._invoke_agent(_llm_agent(state), system, user)
        data = _extract_json(raw)
        summaries: list[AgentStanceSummary] = []
        for item in data.get("agent_summaries", []):
            if not isinstance(item, dict) or "agent_id" not in item:
                continue
            summaries.append(AgentStanceSummary(
                agent_id=str(item.get("agent_id", "")),
                initial_claim=str(item.get("initial_claim", "")),
                current_stance=str(item.get("current_stance", "")),
                stance_shift=str(item.get("stance_shift", "")),
            ))
        score = max(0.0, min(1.0, float(data.get("convergence_score", 0.0) or 0.0)))
        conflicts = [str(c) for c in data.get("key_conflicts", []) if c]
        return PhaseSummary(phase=phase, agent_summaries=summaries,
                            key_conflicts=conflicts, convergence_score=score)

    # ----- DB 커밋 (낙관적 락) / 브로드캐스트 -----
    async def _load(self, discussion_id: str) -> Optional[DiscussionState]:
        """DB 에서 상태를 비동기로 로드한다 (완전 await — 스레드 오프로딩 없음)."""
        return await database.load_state(discussion_id)

    async def _commit(
        self, discussion_id: str,
        mutate: "Callable[[DiscussionState], None]",
    ) -> DiscussionState:
        """낙관적 락으로 상태를 갱신한다 (완전 비동기 — await 구조).

        ``await load_state -> mutate -> await update_state`` 를 시도하고, 버전
        충돌(`StaleStateError`) 시 신선한 상태로 재로드해 재시도한다. ``mutate``
        는 멱등이어야 한다 (재시도 시 신선한 상태에 재적용된다).
        """
        last_exc: Optional[Exception] = None
        for _ in range(_COMMIT_RETRIES):
            state = await database.load_state(discussion_id)
            if state is None:
                raise DiscussionError(f"토론 {discussion_id} 를 찾을 수 없습니다.")
            mutate(state)
            try:
                await database.update_state(state)
                return state
            except database.StaleStateError as exc:
                last_exc = exc  # 다른 트랜잭션이 먼저 갱신 — 재로드 후 재시도
                continue
        raise DiscussionError(f"낙관적 락 재시도 소진: {discussion_id} ({last_exc})")

    async def _mark_error(self, discussion_id: str, message: str) -> None:
        """토론을 ERROR 상태로 전이한다 (비치명적 — 실패해도 무시)."""
        try:
            await self._commit(discussion_id, lambda s: _set_error(s, message))
            await self._emit(discussion_id, WSMessageType.ERROR,
                             {"message": message})
        except Exception:  # noqa: BLE001
            logger.exception("오류 상태 기록 실패: %s", discussion_id)

    async def _emit(
        self, discussion_id: str, msg_type: WSMessageType, payload: dict
    ) -> None:
        """WS 메시지를 브로드캐스트한다."""
        await self._broadcast(discussion_id, WSMessage(type=msg_type, payload=payload))

    async def _emit_turn(self, discussion_id: str, turn: AgentTurn) -> None:
        """에이전트 발언 1건 완료를 알린다(최종 텍스트 — 실패/수동 턴 포함)."""
        await self._emit(discussion_id, WSMessageType.AGENT_TURN,
                         {"turn": turn.model_dump(mode="json")})


# ===========================================================================
# 멱등 mutate 함수 — _commit 에 전달 (재시도 시 신선한 state 에 재적용됨)
# ===========================================================================
def _mark_phase_start(state: DiscussionState, phase: DiscussionPhase) -> None:
    """단계 진입 마킹. 멱등."""
    state.current_phase = phase
    state.status = DiscussionStatus.RUNNING
    state.touch()


def _record_turns(
    state: DiscussionState, phase: DiscussionPhase, turns: list[AgentTurn]
) -> None:
    """발언 턴을 단계 기록에 추가한다. 멱등 — 이미 있는 agent_id 는 건너뛴다."""
    record = state.record_for_phase(phase)
    existing = {t.agent_id for t in record}
    for turn in turns:
        if turn.agent_id not in existing:
            record.append(turn)
            existing.add(turn.agent_id)
    if state.status is DiscussionStatus.PENDING_MANUAL_INPUT:
        state.status = DiscussionStatus.RUNNING
    state.touch()


def _set_status(state: DiscussionState, status: DiscussionStatus) -> None:
    """상태 전이. 멱등."""
    state.status = status
    state.touch()


def _set_agreement(state: DiscussionState, agreement: str) -> None:
    """5단계 합의안 문서 설정. 멱등."""
    state.final_joint_agreement = agreement
    state.touch()


def _set_error(state: DiscussionState, message: str) -> None:
    """오류 상태 전이. 멱등."""
    state.status = DiscussionStatus.ERROR
    state.error = message
    state.touch()
