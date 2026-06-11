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
import base64
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Optional

from . import database
from .formats import (
    PHASE_COMPLETED,
    PHASE_IDLE,
    DiscussionFormat,
    PhaseSpec,
    get_format,
    phase_key,
    phase_of_key,
    plan_next,
    round_of_key,
)
from .schemas import (
    AgentConfig,
    AgentStanceSummary,
    AgentTurn,
    DiscussionState,
    DiscussionStatus,
    FacilitatorNote,
    ModelProvider,
    PersonaType,
    PhaseSummary,
    ReviewExchange,
    ReviewState,
    UserIntervention,
    WSMessage,
    WSMessageType,
)

logger = logging.getLogger(__name__)

#: 브로드캐스트 콜백: (discussion_id, WSMessage) 를 받아 (비동기로) 전송한다.
BroadcastCallback = Callable[[str, WSMessage], Awaitable[None]]
#: 토큰 콜백: 스트리밍 청크 1개를 받아 (비동기로) 처리하는 함수.
TokenCallback = Callable[[str], Awaitable[None]]

#: 합의안 합성 발언에 쓰는 가상 발화자 ID (마지막 단계 force_consensus=True).
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

#: 전원 manual 토론에서 시스템 LLM 호출(단계 요약·합의 근접도·합의안 합성)에
#: 쓸 폴백 모델. 호출 가능한 API 에이전트가 하나도 없을 때만 사용한다.
#: OPENAI_API_KEY 가 있어야 동작하며, 없으면 해당 호출은 우아하게 실패 처리된다.
_FALLBACK_LLM_MODEL = os.getenv("AGORA_FALLBACK_MODEL", "gpt-4o-mini")


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
    REVIEW_QUESTION = "review_question"   # 검토 게이트 — 진행자 질문
    REVIEW_APPROVE = "review_approve"     # 검토 게이트 — 진행자 승인
    END = "end"                          # 게이트에서 토론 조기 종료
    RECOVER = "recover"                   # 서버 재기동 후 크래시 복구


# ===========================================================================
# 프롬프트 상수 / 형식 헬퍼
# ===========================================================================
#: 순차 포스팅 단계에서 후순위 에이전트에게 동적 삽입하는 중복 회피 개정 지침.
_SEQUENTIAL_REVISION_HINT = (
    "[개정 지침] 위 '이번 단계 선행 의견'을 너의 초안과 비교하라. 중복되는 "
    "논거는 해당 동료의 것으로 인정하고, 너만의 차별화된 관점·근거를 부각하여 "
    "최종 포스팅하라."
)

#: persona_type 을 프롬프트에 반영하는 역할 지침. NEUTRAL·레거시 역할은 힌트 없음.
_PERSONA_ROLE_HINTS: dict[PersonaType, str] = {
    PersonaType.IDEATOR: "이 세션에서 너는 새로운 아이디어를 적극적으로 발산하는 역할이다.",
    PersonaType.BUILDER: "이 세션에서 너는 다른 사람의 아이디어에 살을 붙여 키우는 역할이다.",
    PersonaType.CRITIC: "이 세션에서 너는 허점과 약점을 건설적으로 짚는 역할이다.",
    PersonaType.SYNTHESIZER: "이 세션에서 너는 흩어진 의견을 통합하고 정리하는 역할이다.",
    PersonaType.PRAGMATIST: "이 세션에서 너는 실행 가능성과 현실 제약을 따지는 역할이다.",
}


def _format_of(state: DiscussionState) -> DiscussionFormat:
    """토론 상태에 지정된 형식을 반환한다 (미지의 id 면 기본 형식)."""
    return get_format(state.format_id)


def _instance_label(spec: Optional[PhaseSpec], round_no: int) -> str:
    """단계 인스턴스의 표시 라벨. 반복 단계면 '· N라운드' 를 덧붙인다."""
    if spec is None:
        return "?"
    return f"{spec.label} · {round_no}라운드" if spec.repeatable else spec.label


def _instances_in_order(
    fmt: DiscussionFormat, state: DiscussionState
) -> list[tuple[PhaseSpec, int, str]]:
    """phase_records 에 실제로 나타난 단계 인스턴스를 (스펙, 라운드, 키) 로 반환한다.

    형식의 단계 순서를 따르되, 반복 단계는 기록에 존재하는 라운드만 오름차순으로
    펼친다. 비반복 단계는 단일 인스턴스(라운드 1). 아직 시작 안 한 단계는 제외.
    """
    out: list[tuple[PhaseSpec, int, str]] = []
    for spec in fmt.phases:
        if spec.repeatable:
            rounds = sorted(
                round_of_key(k) for k in state.phase_records
                if phase_of_key(k) == spec.id
            )
            out.extend((spec, r, phase_key(spec, r)) for r in rounds)
        elif spec.id in state.phase_records:
            out.append((spec, 1, spec.id))
    return out


def _latest_convergence(state: DiscussionState, instance_key: str) -> float:
    """주어진 단계 인스턴스의 합의 근접도. 요약이 없으면 0.0."""
    for summary in state.phase_summaries:
        if summary.phase == instance_key:
            return summary.convergence_score
    return 0.0


def _render_convergence_trajectory(
    state: DiscussionState, max_items: int = 3
) -> str:
    """직전까지 단계들의 합의 근접도 추이를 텍스트로 렌더한다.

    요약 프롬프트에 주입해 '이번 단계는 이 추이의 연장선' 임을 모델이 보게 한다 —
    단계가 진행될수록 자연스럽게 수렴하도록(과소평가 방지). 아직 요약된 단계가
    없으면 빈 문자열. 최근 ``max_items`` 개만 넣어 프롬프트 길이를 묶는다.
    """
    fmt = _format_of(state)
    summary_of = {s.phase: s for s in state.phase_summaries}
    lines: list[str] = []
    for spec, rnd, key in _instances_in_order(fmt, state):
        s = summary_of.get(key)
        if s is None:
            continue
        label = _instance_label(spec, rnd)
        lines.append(f"- {label}: 합의 근접도 {s.convergence_score:.0%}")
    if not lines:
        return ""
    return "[직전까지의 합의 근접도 추이]\n" + "\n".join(lines[-max_items:])


def _stored_decision(state: DiscussionState, instance_key: str) -> Optional[str]:
    """주어진 단계 인스턴스에 대해 사회자가 내린 진행 결정 (있으면 그 값).

    ``_finish_phase`` 가 게이트에서 결정을 한 번 내려 facilitator_notes 에 남기고,
    ADVANCE 경로가 같은 결정을 재사용한다 — 두 경로의 plan_next 가 일치한다.
    """
    for note in state.facilitator_notes:
        if note.kind == "decision" and note.phase == instance_key:
            return note.decision
    return None


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
    if provider is ModelProvider.DEEPSEEK:
        # DeepSeek 은 OpenAI-호환 API — openai SDK 의 AsyncOpenAI 를 base_url 만
        # 바꿔 재사용한다.
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise DiscussionError("openai 패키지가 설치되지 않았습니다.") from exc
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise DiscussionError("환경 변수 DEEPSEEK_API_KEY 가 설정되지 않았습니다.")
        return AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    if provider is ModelProvider.GEMINI:
        # Gemini(Google)도 OpenAI-호환 엔드포인트를 제공 — 같은 AsyncOpenAI 재사용.
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise DiscussionError("openai 패키지가 설치되지 않았습니다.") from exc
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise DiscussionError(
                "환경 변수 GEMINI_API_KEY (또는 GOOGLE_API_KEY) 가 설정되지 않았습니다.")
        return AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
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
) -> tuple[str, dict]:
    """OpenAI Chat Completions 스트리밍 호출. (누적 텍스트, 토큰 사용량) 반환."""
    stream = await client.chat.completions.create(  # type: ignore[attr-defined]
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True},
    )
    parts: list[str] = []
    usage: dict = {}
    async for chunk in stream:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = {"prompt_tokens": chunk_usage.prompt_tokens,
                     "completion_tokens": chunk_usage.completion_tokens}
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
            if on_token is not None:
                await on_token(delta)
    return "".join(parts).strip(), usage


async def _call_anthropic(
    client: object, model: str, system: str, user: str,
    temperature: float, max_tokens: int, on_token: Optional[TokenCallback],
) -> tuple[str, dict]:
    """Anthropic Messages 스트리밍 호출. (누적 텍스트, 토큰 사용량) 반환."""
    parts: list[str] = []
    usage: dict = {}
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
        try:
            final = await stream.get_final_message()
            usage = {"prompt_tokens": final.usage.input_tokens,
                     "completion_tokens": final.usage.output_tokens}
        except Exception:  # noqa: BLE001 - 사용량 추출 실패는 비치명적
            usage = {}
    return "".join(parts).strip(), usage


async def _call_ollama(
    client: object, model: str, system: str, user: str,
    temperature: float, max_tokens: int, on_token: Optional[TokenCallback],
) -> tuple[str, dict]:
    """로컬 Ollama 스트리밍 호출 (API Key 불필요). (누적 텍스트, 사용량) 반환."""
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
    usage: dict = {}
    async for chunk in stream:
        getter = chunk.get if isinstance(chunk, dict) else (
            lambda k: getattr(chunk, k, None))
        message = getter("message")
        content = (message.get("content") if isinstance(message, dict)
                   else getattr(message, "content", None)) if message else None
        if content:
            parts.append(content)
            if on_token is not None:
                await on_token(content)
        prompt_n, eval_n = getter("prompt_eval_count"), getter("eval_count")
        if prompt_n is not None or eval_n is not None:
            usage = {"prompt_tokens": prompt_n or 0,
                     "completion_tokens": eval_n or 0}
    return "".join(parts).strip(), usage


#: 쟁점 항목 분류 → 합의 기여 가중치. 대립=0, 부분합의=0.5, 합의=1.
_ISSUE_STATUS_WEIGHT = {"agreed": 1.0, "partial": 0.5, "contested": 0.0}


def _convergence_from_summary(data: dict) -> float:
    """요약 JSON 에서 합의 근접도를 산출한다 (0.0~1.0).

    가능하면 ``issue_points`` 분류 분포에서 *계산*한다 — 모델이 근거 없이 뱉는
    holistic 숫자보다, 셀 수 있는 쟁점별 합의/대립 분류에서 점수를 끌어내는 편이
    과소평가에 덜 취약하다. issue_points 가 없으면 모델의 convergence_score 로,
    둘 다 있으면 두 값을 평균해 한쪽 극단을 누그러뜨린다.
    """
    points = data.get("issue_points")
    derived: Optional[float] = None
    if isinstance(points, list) and points:
        weights = [
            _ISSUE_STATUS_WEIGHT.get(str(p.get("status", "")).lower())
            for p in points if isinstance(p, dict)
        ]
        weights = [w for w in weights if w is not None]
        if weights:
            derived = sum(weights) / len(weights)

    raw_score: Optional[float]
    try:
        raw_score = float(data.get("convergence_score"))
    except (TypeError, ValueError):
        raw_score = None

    if derived is not None and raw_score is not None:
        score = (derived + raw_score) / 2.0   # 분해 점수와 직출 점수의 평균
    elif derived is not None:
        score = derived
    elif raw_score is not None:
        score = raw_score
    else:
        score = 0.0
    return max(0.0, min(1.0, score))


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
def _split_reasoning_draft(text: str) -> tuple[str, str]:
    """'[사고흐름] … [초안] …' 형식 응답을 (사고흐름, 초안)으로 분리한다.

    [초안] 마커가 없으면 전체를 초안으로 보고 사고흐름은 빈 문자열로 둔다.
    """
    marker = "[초안]"
    idx = text.find(marker)
    if idx < 0:
        return "", text.strip()
    reasoning = text[:idx].replace("[사고흐름]", "").strip()
    draft = text[idx + len(marker):].strip()
    return reasoning, draft


def _agent_by_id(state: DiscussionState, agent_id: str) -> AgentConfig:
    """agent_id 로 AgentConfig 를 찾는다. 없으면 DiscussionError."""
    for agent in state.agents:
        if agent.agent_id == agent_id:
            return agent
    raise DiscussionError(f"에이전트 '{agent_id}' 를 찾을 수 없습니다.")


def _llm_agent(state: DiscussionState) -> AgentConfig:
    """요약/합성 등 시스템 LLM 호출에 쓸 에이전트를 고른다.

    첫 번째 비-manual 에이전트를 쓴다. 모든 참가자가 manual 이라 호출 가능한
    API 가 하나도 없으면 — manual 공급자로는 단계 요약·합의 근접도·합의안 합성
    같은 시스템 호출을 수행할 수 없으므로 — 폴백 모델(``_FALLBACK_LLM_MODEL``,
    기본 ``gpt-4o-mini``)로 시스템 호출 전용 임시 에이전트를 만들어 반환한다.
    provider 는 model 명에서 추론되며(gpt-* → OpenAI), OPENAI_API_KEY 가 없으면
    호출이 실패해도 호출부(_summarize_phase 등)가 우아하게 흡수한다.
    """
    for agent in state.agents:
        if agent.provider is not ModelProvider.MANUAL:
            return agent
    return AgentConfig(
        agent_id="_system_fallback", name="시스템",
        model=_FALLBACK_LLM_MODEL, persona_prompt="",
    )


def _failure_turn(
    agent: AgentConfig, phase: str, exc: BaseException
) -> AgentTurn:
    """실패한 에이전트의 발언칸에 적재할 시스템 경고 턴 (우아한 부분 실패 수용)."""
    return AgentTurn(
        agent_id=agent.agent_id,
        phase=phase,
        content=f"[시스템 경고: 에이전트 {agent.agent_id}의 응답 생성 실패 - {exc}]",
        metadata={"failed": True, "error": repr(exc)},
    )


def _render_phase_summary(
    state: DiscussionState, phase: str, summary: PhaseSummary
) -> str:
    """단계 요약 메트릭스(LTM)를 경량 텍스트로 렌더링한다."""
    name_of = {a.agent_id: a.name for a in state.agents}
    spec = _format_of(state).phase(phase)
    label = _instance_label(spec, round_of_key(phase))
    lines = [f"== {label} [요약 메트릭스 · LTM] =="]
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
    current_key: str,
    prior_turns: list[AgentTurn],
    force_full: bool = False,
) -> str:
    """이전 단계 인스턴스 발언 + 유저 개입 + (순차) 선행 의견을 렌더링한다.

    가변 길이 형식에서는 단계 인스턴스(반복 단계의 라운드 포함)를 실행 순서대로
    펼친다. 콘텍스트 압축(LTM): 최근 2개 인스턴스를 제외한 더 오래된 것은 원본
    로그 대신 ``phase_summaries`` 요약을 경량 주입한다. ``force_full=True`` 이면
    압축을 끈다.
    """
    name_of = {a.agent_id: a.name for a in state.agents}
    summary_of = {s.phase: s for s in state.phase_summaries}
    lines: list[str] = []

    pre = [iv for iv in state.user_interventions if iv.after_phase is None]
    if pre:
        lines.append("== 진행자 사전 지시 ==")
        lines.extend(f"[참가자 H] {iv.message}" for iv in pre)

    # 현재 인스턴스 직전까지가 '과거' — 현재 라운드 동석 발언은 prior_turns 로 받는다.
    past: list[tuple[PhaseSpec, int, str]] = []
    for spec, rnd, key in _instances_in_order(_format_of(state), state):
        if key == current_key:
            break
        past.append((spec, rnd, key))

    for idx, (spec, rnd, key) in enumerate(past):
        is_recent = idx >= len(past) - 2   # 최근 2개 인스턴스는 원본 유지
        label = _instance_label(spec, rnd)
        if not force_full and not is_recent and key in summary_of:
            lines.append(_render_phase_summary(state, key, summary_of[key]))
        else:
            turns = state.record_for_phase(key)
            if turns:
                lines.append(f"== {label} ==")
                for turn in turns:
                    speaker = name_of.get(turn.agent_id, turn.agent_id)
                    lines.append(f"[{speaker}] {turn.content}")
        after = [iv for iv in state.user_interventions if iv.after_phase == key]
        if after:
            lines.append(f"-- {label} 이후 진행자 개입 --")
            lines.extend(f"[참가자 H] {iv.message}" for iv in after)

    if prior_turns:
        lines.append("== 이번 단계 선행 의견 ==")
        for turn in prior_turns:
            speaker = name_of.get(turn.agent_id, turn.agent_id)
            lines.append(f"[{speaker}] {turn.content}")
    return "\n".join(lines)


def _render_delta(
    state: DiscussionState,
    current_key: str,
    prior_turns: list[AgentTurn],
) -> str:
    """직전 단계 인스턴스 발언 + 그 직후 진행자 개입 + (순차) 이번 선행 의견만."""
    name_of = {a.agent_id: a.name for a in state.agents}
    instances = _instances_in_order(_format_of(state), state)
    keys = [k for _, _, k in instances]
    prev: Optional[tuple[PhaseSpec, int, str]] = None
    if current_key in keys:
        idx = keys.index(current_key)
        if idx >= 1:
            prev = instances[idx - 1]
    elif instances:
        prev = instances[-1]   # 현재 인스턴스가 아직 기록 전 — 마지막 과거가 직전

    lines: list[str] = []
    if prev is not None:
        spec, rnd, key = prev
        turns = state.record_for_phase(key)
        if turns:
            lines.append(
                f"== 직전 단계({_instance_label(spec, rnd)}) 신규 발언 ==")
            for turn in turns:
                speaker = name_of.get(turn.agent_id, turn.agent_id)
                lines.append(f"[{speaker}] {turn.content}")
        after = [iv for iv in state.user_interventions if iv.after_phase == key]
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
    phase: str,
    prior_turns: list[AgentTurn],
    force_full: bool = False,
) -> tuple[str, str]:
    """[공통규칙]+[페르소나] -> system, [맥락]+[단계지침] -> user 로 조립한다."""
    fmt = _format_of(state)
    spec = fmt.phase(phase)
    system = (
        fmt.common_rules.format(topic=state.topic)
        + f"\n\n[너의 페르소나]\n{agent.persona_prompt}"
    )
    role_hint = _PERSONA_ROLE_HINTS.get(agent.persona_type)
    if role_hint:
        system += f"\n\n[너의 역할]\n{role_hint}"
    instruction = spec.instruction if spec else f"[{phase}] 단계 작업을 수행하라."
    if spec and spec.repeatable:
        instruction += (
            f"\n(지금은 {round_of_key(phase)}라운드다 — 앞 라운드의 문답을 "
            "딛고 논점을 더 좁혀라.)"
        )
    if spec and spec.sequential and prior_turns:
        instruction = f"{instruction}\n\n{_SEQUENTIAL_REVISION_HINT}"
    history = _render_history(state, phase, prior_turns, force_full=force_full)
    user_sections = [f"[토론 주제]\n{state.topic}"]
    if history:
        user_sections.append(history)
    user_sections.append(f"[지금 너({agent.name})가 수행할 작업]\n{instruction}")
    return system, "\n\n".join(user_sections)


#: 사회자 훅 종류별 작업 지침. ``decision`` 은 가변 길이 형식 구동에 쓰인다.
_FACILITATOR_TASKS: dict[str, str] = {
    "open": (
        "토론을 연다. 주제의 핵심 쟁점을 짚고, 첫 단계에서 참가자들이 무엇에 "
        "집중하면 좋을지 2~3문장으로 안내하라."
    ),
    "between": (
        "방금 끝난 단계를 짚어라. 아직 풀리지 않은 가장 첨예한 쟁점 하나를 "
        "골라, 다음 단계의 초점을 2~3문장으로 제시하라."
    ),
    "close": (
        "토론 전체를 마무리한다. 합의된 지점과 끝내 갈린 지점을 구분해 "
        "3~4문장으로 정리하라."
    ),
    "decision": (
        "이 반복 단계(문답 라운드)를 한 번 더 진행할지 판단하라. 첫 줄에 정확히 "
        "'[결정] continue'(라운드 계속)·'[결정] next'(다음 단계로)·"
        "'[결정] conclude'(토론 종료) 중 하나만 쓰고, 다음 줄부터 그 판단의 "
        "근거를 2~3문장으로 적어라."
    ),
}


def _build_facilitator_prompt(
    state: DiscussionState, kind: str, phase: str
) -> tuple[str, str]:
    """사회자 훅용 (system, user) 프롬프트를 조립한다.

    사회자는 토론 전체 기록을 압축 없이(force_full) 받아, 종류별 작업을 수행한다.
    """
    fac = state.facilitator
    persona = fac.persona_prompt if fac is not None else ""
    system = (
        f"너는 '{state.topic}' 토론의 사회자(facilitator)다. 토론자가 아니다 — "
        "어느 한쪽 입장을 편들지 말고, 토론이 생산적으로 굴러가도록 조율하라. "
        "한국어로 간결하게 작성한다.\n\n"
        f"[너의 진행 스타일]\n{persona}"
    )
    history = _render_history(state, PHASE_COMPLETED, [], force_full=True)
    task = _FACILITATOR_TASKS.get(kind, "지금까지의 진행 상황을 정리하라.")
    user = (
        f"[토론 주제]\n{state.topic}\n\n"
        f"[지금까지의 토론]\n{history or '(아직 발언 없음)'}\n\n"
        f"[사회자 작업]\n{task}"
    )
    return system, user


def _parse_facilitator_decision(text: str) -> Optional[str]:
    """사회자 응답에서 '[결정] continue|next|conclude' 진행 결정을 추출한다."""
    match = re.search(r"\[결정\]\s*(continue|next|conclude)", text)
    return match.group(1) if match else None


def generate_deep_copy(
    state: DiscussionState,
    agent_id: str,
    phase: str,
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
    phase: str,
    prior_turns: Optional[list[AgentTurn]] = None,
) -> str:
    """일반 복사: 압축 메모리·시스템 프롬프트 제외, 직전 단계 신규 델타 맥락만."""
    agent = _agent_by_id(state, agent_id)
    prior = list(prior_turns or [])
    spec = _format_of(state).phase(phase)
    delta = _render_delta(state, phase, prior)
    instruction = spec.instruction if spec else f"[{phase}] 단계 작업을 수행하라."
    if spec and spec.sequential and prior:
        instruction = f"{instruction}\n\n{_SEQUENTIAL_REVISION_HINT}"
    sections = ["[복사 유형] 일반 복사 — 진행 중인 대화 세션에 이어 붙이세요."]
    if delta:
        sections.append(delta)
    sections.append(f"[지금 너({agent.name})가 수행할 작업]\n{instruction}")
    return "\n\n".join(sections)


def render_transcript(state: DiscussionState) -> str:
    """토론 전체 기록을 마크다운 텍스트로 렌더링한다 (텍스트 내보내기용).

    프롬프트용 ``_render_history`` 와 달리 LTM 압축을 적용하지 않고, 모든 단계의
    발언 원문·요약·진행자 개입·최종 합의안을 사람이 읽기 좋은 문서로 펼친다.
    """
    fmt = _format_of(state)
    name_of = {a.agent_id: a.name for a in state.agents}
    summary_of = {s.phase: s for s in state.phase_summaries}
    lines: list[str] = [
        f"# Agent Agora — {state.topic}",
        "",
        f"- 형식: {fmt.name} (`{state.format_id}`)",
        f"- 상태: {state.status.value}",
        f"- 생성: {state.created_at:%Y-%m-%d %H:%M} · "
        f"갱신: {state.updated_at:%Y-%m-%d %H:%M}",
    ]
    total_p = total_c = 0
    for turns in state.phase_records.values():
        for turn in turns:
            used = (turn.metadata or {}).get("usage") or {}
            total_p += used.get("prompt_tokens", 0)
            total_c += used.get("completion_tokens", 0)
    if total_p or total_c:
        lines.append(
            f"- 토큰(에이전트 발언): 입력 {total_p:,} · 출력 {total_c:,} "
            f"· 합계 {total_p + total_c:,}")
    lines += ["", "## 참여 에이전트", ""]
    lines += [f"- **{a.name}** (`{a.agent_id}`) — {a.model}" for a in state.agents]
    if state.facilitator is not None:
        lines.append(
            f"- 사회자: **{state.facilitator.name}** — {state.facilitator.model}")

    pre = [iv for iv in state.user_interventions if iv.after_phase is None]
    if pre:
        lines += ["", "## 진행자 사전 지시", ""]
        lines += [f"> {iv.message}" for iv in pre]

    opening = next(
        (n for n in state.facilitator_notes if n.kind == "open"), None)
    if opening is not None:
        lines += ["", "## 사회자 — 개회", "", opening.content]

    for spec, rnd, key in _instances_in_order(fmt, state):
        turns = state.phase_records.get(key, [])
        if not turns:
            continue
        lines += ["", f"## {_instance_label(spec, rnd)}", ""]
        for turn in turns:
            speaker = name_of.get(turn.agent_id, turn.agent_id)
            lines += [f"### {speaker}", "", turn.content, ""]
        summary = summary_of.get(key)
        if summary:
            lines.append(
                f"_단계 요약 · 합의 근접도 {summary.convergence_score:.0%}_")
            if summary.key_conflicts:
                lines.append(
                    f"_주요 쟁점: {' · '.join(summary.key_conflicts)}_")
        for iv in state.user_interventions:
            if iv.after_phase == key:
                lines.append(f"> 진행자 개입: {iv.message}")
        for note in state.facilitator_notes:
            if note.phase == key and note.kind in ("between", "decision"):
                tag = "진행 결정" if note.kind == "decision" else "중간 조율"
                lines += ["", f"**사회자 · {tag}**", "", note.content]

    closing = next(
        (n for n in state.facilitator_notes if n.kind == "close"), None)
    if closing is not None:
        lines += ["", "## 사회자 — 폐회", "", closing.content]

    if state.final_joint_agreement:
        lines += ["", "## 최종 합의안", "", state.final_joint_agreement]
    return "\n".join(lines) + "\n"


#: .md 끝에 심는 복원용 상태 블록의 마커. HTML 주석이라 마크다운 렌더에 안 보인다.
_STATE_BLOCK_RE = re.compile(
    r"<!--\s*AGORA-STATE-V1\n([A-Za-z0-9+/=\s]+?)\n-->", re.DOTALL)


def render_transcript_with_state(state: DiscussionState) -> str:
    """기록 문서 + 복원용 상태 블록.

    사람이 읽는 마크다운 본문은 ``render_transcript`` 그대로 두고, 끝에 전체
    ``DiscussionState`` 를 base64(JSON) HTML 주석으로 심는다 — .md 파일 하나가
    사람용 기록이자 기계용 완전 복원 소스가 된다(별도 sidecar 파일 불필요).
    base64 인코딩은 본문에 '-->' 같은 문자가 들어가 주석이 깨지는 것을 막는다.
    """
    payload = base64.b64encode(
        state.model_dump_json().encode("utf-8")).decode("ascii")
    return (render_transcript(state)
            + f"\n<!-- AGORA-STATE-V1\n{payload}\n-->\n")


def extract_embedded_state(markdown: str) -> Optional[DiscussionState]:
    """.md 의 AGORA-STATE-V1 블록에서 DiscussionState 를 복원한다. 없거나
    손상됐으면 None (호출부는 패턴 파싱으로 폴백)."""
    match = _STATE_BLOCK_RE.search(markdown)
    if not match:
        return None
    try:
        raw = base64.b64decode("".join(match.group(1).split()))
        return DiscussionState.model_validate_json(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001 - 손상 블록은 조용히 폴백
        return None


def archive_transcript(state: DiscussionState) -> str:
    """토론 기록을 마크다운 파일로 로컬 폴더에 저장하고 그 경로를 반환한다.

    한 토론당 파일 하나(``agora-{id}.md``)이며, 다시 저장하면 최신 상태로
    덮어쓴다. 저장 폴더는 ``AGORA_ARCHIVE_DIR`` 환경변수로 바꿀 수 있다
    (기본 ``discussions``, 서버 작업 디렉터리 기준 상대 경로). 파일 끝에
    복원용 상태 블록이 들어가 '불러오기' 가 완벽 복원할 수 있다.
    """
    directory = os.getenv("AGORA_ARCHIVE_DIR", "discussions")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"agora-{state.discussion_id[:8]}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_transcript_with_state(state))
    return path


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
            first = plan_next(_format_of(state), PHASE_IDLE)
            if first is not None:
                await self._run_facilitator(discussion_id, "open", first)
                await self._run_phase(discussion_id, first)
        elif event is PipelineEvent.ADVANCE:
            if state.status is not DiscussionStatus.WAITING_FOR_USER:
                raise InvalidStateTransition(
                    f"advance 는 WAITING_FOR_USER 에서만 가능 (현재 {state.status.value})"
                )
            nxt = plan_next(
                _format_of(state), state.current_phase,
                _latest_convergence(state, state.current_phase),
                _stored_decision(state, state.current_phase),
            )
            if nxt is not None:
                await self._run_phase(discussion_id, nxt)
        elif event is PipelineEvent.MANUAL_RESPONSE:
            await self._on_manual_response(state, payload or {})
        elif event is PipelineEvent.REVIEW_QUESTION:
            await self._on_review_question(state, payload or {})
        elif event is PipelineEvent.REVIEW_APPROVE:
            await self._on_review_approve(state)
        elif event is PipelineEvent.END:
            if state.status is not DiscussionStatus.WAITING_FOR_USER:
                raise InvalidStateTransition(
                    f"end 는 WAITING_FOR_USER 에서만 가능 (현재 {state.status.value})"
                )
            await self._end_discussion(discussion_id)
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
            ("running", "pending_manual_input", "pending_review")
        )
        running = [s for s in stuck if s.status is DiscussionStatus.RUNNING]
        pending = [s for s in stuck
                   if s.status in (DiscussionStatus.PENDING_MANUAL_INPUT,
                                   DiscussionStatus.PENDING_REVIEW)]
        for s in pending:
            logger.info(
                "크래시 복구: %s — %s 유지 (후속 입력 수용 준비 완료)",
                s.discussion_id, s.status.value,
            )
        for s in running:
            logger.info(
                "크래시 복구: %s — RUNNING(%s) 단계 멱등 재기동",
                s.discussion_id, s.current_phase,
            )
            self.trigger(s.discussion_id, PipelineEvent.RECOVER)
        return {"running_recovered": len(running), "pending_preserved": len(pending)}

    async def add_intervention(
        self, discussion_id: str, intervention: UserIntervention
    ) -> None:
        """유저 개입을 상태에 기록한다 (단계 전이는 트리거하지 않음).

        다음 단계 프롬프트 맥락에 '참가자 H' 로 반영된다. 낙관적 락 재시도 시
        중복 적재를 막기 위해 (created_at, message) 로 멱등성을 보장한다.

        사회자가 있고 현재가 비반복 단계 게이트면(=between 노트가 떠 있는 상태),
        개입 직후 between 을 다시 쓰도록 백그라운드 태스크를 발사한다 — 라우터
        응답을 LLM 호출 시간만큼 늦추지 않으면서 사회자가 H 의 발언을 반영한
        새 중간 조율을 만들도록 한다.
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

        # 사회자 있고 비반복 단계 게이트면 between 재작성 — 백그라운드 발사.
        state = await self._load(discussion_id)
        if state is None or state.facilitator is None:
            return
        if state.status is not DiscussionStatus.WAITING_FOR_USER:
            return
        spec = _format_of(state).phase(state.current_phase)
        if spec is None or spec.repeatable:
            return
        task = asyncio.create_task(
            self._regenerate_facilitator_between(
                discussion_id, state.current_phase))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _regenerate_facilitator_between(
        self, discussion_id: str, phase: str
    ) -> None:
        """H 개입 직후 호출 — 기존 between 노트를 지우고 사회자가 다시 쓰게 한다.

        브로드캐스트되는 새 facilitator_note 를 받은 UI 가 기존 같은 (kind, phase)
        노트를 교체한다(렌더링 측 dedup).
        """
        def remove(s: DiscussionState) -> None:
            s.facilitator_notes[:] = [
                n for n in s.facilitator_notes
                if not (n.kind == "between" and n.phase == phase)
            ]
            s.touch()

        await self._commit(discussion_id, remove)
        await self._run_facilitator(discussion_id, "between", phase)

    async def set_intercepts(
        self, discussion_id: str, agent_ids: list[str]
    ) -> None:
        """검토 게이트로 가로챌 에이전트 목록을 설정한다 (빈 목록=해제).

        가로채기는 *다음* 턴부터 적용된다 — 지정된 API 에이전트는 자동 포스팅
        대신 초안·사고흐름을 만들고 PENDING_REVIEW 로 대기한다.
        """
        await self._commit(
            discussion_id, lambda s: _set_intercepts(s, agent_ids))

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
            spec = _format_of(state).phase(phase)
            if spec is None:
                return
            record = state.record_for_phase(phase)
            posted = {t.agent_id for t in record}
            # posted 는 이미 발언한 에이전트 판별용(항상 전체), prior 는 복사본에
            # 넣을 맥락 — 동시 단계에서는 서로의 발제가 새지 않도록 비운다.
            prior = list(record) if spec.sequential else []
            for agent in state.agents:
                if (agent.provider is not ModelProvider.MANUAL
                        or agent.agent_id in posted):
                    continue
                message = WSMessage(
                    type=WSMessageType.MANUAL_INPUT_REQUIRED,
                    payload={
                        "agent_id": agent.agent_id,
                        "phase": phase,
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
        if _format_of(state).phase(phase) is None:
            return
        # 멱등 재기동: 현재 단계의 부분 발언 기록·요약을 비우고 처음부터 재실행.
        def reset(s: DiscussionState) -> None:
            s.record_for_phase(phase).clear()
            s.phase_summaries[:] = [p for p in s.phase_summaries
                                    if p.phase != phase]
            s.touch()

        await self._commit(state.discussion_id, reset)
        await self._run_phase(state.discussion_id, phase)

    async def _on_manual_response(
        self, state: DiscussionState, payload: dict
    ) -> None:
        """수동 에이전트 응답 주입 — 턴 기록 후 단계 진행을 재개한다."""
        agent_id = str(payload.get("agent_id", ""))
        phase = str(payload["phase"])
        content = str(payload.get("content", "")).strip()

        if state.status is not DiscussionStatus.PENDING_MANUAL_INPUT:
            raise InvalidStateTransition(
                f"수동 입력은 PENDING_MANUAL_INPUT 에서만 가능 (현재 {state.status.value})"
            )
        agent = _agent_by_id(state, agent_id)
        if agent.provider is not ModelProvider.MANUAL:
            raise InvalidStateTransition(f"에이전트 {agent_id} 는 수동 에이전트가 아닙니다.")
        if phase != state.current_phase:
            raise InvalidStateTransition(
                f"수동 입력 단계({phase})가 현재 단계와 다릅니다."
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

    async def _on_review_question(
        self, state: DiscussionState, payload: dict
    ) -> None:
        """검토 중인 에이전트에게 진행자 질문을 던지고 답변을 받는다."""
        if (state.status is not DiscussionStatus.PENDING_REVIEW
                or state.review is None):
            raise InvalidStateTransition(
                f"검토 문답은 PENDING_REVIEW 에서만 가능 (현재 {state.status.value})"
            )
        question = str(payload.get("question", "")).strip()
        if not question:
            raise InvalidStateTransition("질문이 비어 있습니다.")
        review = state.review
        agent = _agent_by_id(state, review.agent_id)
        try:
            answer = await self._answer_review_question(
                state, agent, review, question)
        except Exception as exc:  # noqa: BLE001 - 우아한 부분 실패 수용
            logger.warning("검토 답변 생성 실패: %r", exc)
            answer = f"[시스템 경고: 답변 생성 실패 - {exc}]"
        exchange = ReviewExchange(question=question, answer=answer)
        await self._commit(
            state.discussion_id, lambda s: _append_review_qa(s, exchange))
        await self._emit(
            state.discussion_id, WSMessageType.REVIEW_ANSWER,
            {"exchange": exchange.model_dump(mode="json")},
        )

    async def _on_review_approve(self, state: DiscussionState) -> None:
        """검토 승인 — 초안·문답을 종합한 최종 발언을 확정하고 단계 진행을 재개한다."""
        if (state.status is not DiscussionStatus.PENDING_REVIEW
                or state.review is None):
            raise InvalidStateTransition(
                f"검토 승인은 PENDING_REVIEW 에서만 가능 (현재 {state.status.value})"
            )
        review = state.review
        agent = _agent_by_id(state, review.agent_id)
        try:
            final = await self._finalize_reviewed_turn(state, agent, review)
        except Exception as exc:  # noqa: BLE001 - 우아한 부분 실패 수용
            logger.warning("검토 최종 발언 생성 실패: %r", exc)
            final = review.draft or f"[시스템 경고: 최종 발언 생성 실패 - {exc}]"
        phase = review.phase
        turn = AgentTurn(
            agent_id=review.agent_id, phase=phase, content=final,
            metadata={"provider": agent.get_provider().value,
                      "model": agent.model, "reviewed": True},
        )

        def commit_final(s: DiscussionState) -> None:
            _record_turns(s, phase, [turn])
            _clear_review(s)

        await self._commit(state.discussion_id, commit_final)
        await self._emit_turn(state.discussion_id, turn)
        await self._advance_phase_progress(state.discussion_id, phase)

    async def _answer_review_question(
        self, state: DiscussionState, agent: AgentConfig,
        review: ReviewState, question: str,
    ) -> str:
        """검토 중 진행자 질문에 에이전트로서 답한다."""
        qa_text = "\n".join(
            f"진행자: {x.question}\n너: {x.answer}" for x in review.qa
        )
        system = (
            f"너는 '{state.topic}' 토론의 참가자다. 진행자가 너의 발언 초안에 "
            "대해 질문한다. 초안의 의도를 솔직하게 설명하고, 타당한 지적이면 "
            "수용 의사를 밝혀라. 간결하게 답한다."
        )
        user = (
            f"[너의 사고흐름]\n{review.reasoning}\n\n"
            f"[너의 발언 초안]\n{review.draft}\n\n"
            + (f"[지금까지의 문답]\n{qa_text}\n\n" if qa_text else "")
            + f"[진행자 질문]\n{question}\n\n[지시] 위 질문에 답하라."
        )
        return (await self._invoke_agent(agent, system, user)).strip()

    async def _finalize_reviewed_turn(
        self, state: DiscussionState, agent: AgentConfig, review: ReviewState,
    ) -> str:
        """검토 초안 + 문답을 종합한 최종 발언을 만든다.

        문답이 없으면 초안을 그대로 최종 발언으로 쓴다 (불필요한 LLM 호출 회피).
        """
        if not review.qa:
            return review.draft
        qa_text = "\n".join(
            f"진행자: {x.question}\n너: {x.answer}" for x in review.qa
        )
        fmt = _format_of(state)
        spec = fmt.phase(review.phase)
        instruction = spec.instruction if spec else ""
        system = (
            fmt.common_rules.format(topic=state.topic)
            + f"\n\n[너의 페르소나]\n{agent.persona_prompt}"
        )
        user = (
            f"[너의 발언 초안]\n{review.draft}\n\n"
            f"[진행자와의 문답]\n{qa_text}\n\n"
            f"[이번 단계 작업]\n{instruction}\n\n"
            "[지시] 위 문답을 반영해 최종 발언을 완성하라. 발언 본문만 출력하라."
        )

        async def on_token(token: str) -> None:
            await self._emit(
                state.discussion_id, WSMessageType.TOKEN_STREAM,
                {"agent_id": review.agent_id, "phase": review.phase,
                 "token": token},
            )

        return (await self._invoke_agent(
            agent, system, user, on_token)).strip()

    # ----- 단계 실행 -----
    async def _run_phase(self, discussion_id: str, phase: str) -> None:
        """단계 인스턴스 진입 — status RUNNING 마킹 후 가능한 데까지 진행한다.

        ``phase`` 는 단계 인스턴스 키 — 반복 단계면 ``'probe#3'`` 처럼 라운드를
        담는다(비반복 단계는 순수 id).
        """
        state = await self._commit(
            discussion_id, lambda s: _mark_phase_start(s, phase))
        spec = _format_of(state).phase(phase)
        await self._emit(
            discussion_id, WSMessageType.PHASE_STARTED,
            {"phase": phase, "round": round_of_key(phase),
             "label": _instance_label(spec, round_of_key(phase))},
        )
        await self._advance_phase_progress(discussion_id, phase)

    async def _advance_phase_progress(
        self, discussion_id: str, phase: str
    ) -> None:
        """현재 단계를 '인간 입력 없이 가능한 데까지' 진행한다.

        * 모든 에이전트가 게시 완료 -> ``_finish_phase``
        * 수동 에이전트 차례 -> ``_enter_pending`` 후 즉시 종료(메모리 반환)
        * API 에이전트 -> 순차 단계는 1명씩, 동시 단계는 일괄 호출
        """
        state = await self._load(discussion_id)
        if state is None:
            return

        fmt = _format_of(state)
        spec = fmt.phase(phase)
        # 합의안 합성은 '비반복' 마지막 단계에서만 — 반복 단계는 매 라운드가
        # 마지막 스펙이 아니므로 is_last_phase 만으로는 부족하다.
        if (spec is not None and not spec.repeatable and fmt.is_last_phase(phase)
                and fmt.supports_consensus and state.force_consensus):
            await self._run_consensus(discussion_id, phase)
            return

        record = state.record_for_phase(phase)
        posted = {t.agent_id for t in record}
        pending = [a for a in state.agents if a.agent_id not in posted]
        if not pending:
            await self._finish_phase(discussion_id, phase)
            return

        if spec is not None and spec.sequential:
            nxt = pending[0]
            if nxt.provider is ModelProvider.MANUAL:
                await self._enter_pending(discussion_id, phase, [nxt])
                return
            if nxt.agent_id in state.intercept_agents:
                await self._enter_review(discussion_id, phase, nxt, list(record))
                return
            turn = await self._do_api_turn(state, phase, nxt, list(record))
            await self._commit(
                discussion_id, lambda s: _record_turns(s, phase, [turn])
            )
            await self._emit_turn(discussion_id, turn)
            # 다음 에이전트로 진행 (재귀 — 깊이 = 에이전트 수).
            await self._advance_phase_progress(discussion_id, phase)
        else:
            auto_api = [a for a in pending
                        if a.provider is not ModelProvider.MANUAL
                        and a.agent_id not in state.intercept_agents]
            review_api = [a for a in pending
                          if a.provider is not ModelProvider.MANUAL
                          and a.agent_id in state.intercept_agents]
            manual_pending = [a for a in pending
                              if a.provider is ModelProvider.MANUAL]
            if auto_api:
                turns = await self._gather_api_turns(state, phase, auto_api)
                await self._commit(
                    discussion_id, lambda s: _record_turns(s, phase, turns)
                )
                for turn in turns:
                    await self._emit_turn(discussion_id, turn)
            # 가로채기된 에이전트·수동 에이전트는 한 번에 하나씩 대기로 보낸다 —
            # 해당 대기가 풀리면 _advance_phase_progress 가 재호출돼 나머지를 잇는다.
            if review_api:
                await self._enter_review(
                    discussion_id, phase, review_api[0], [])
                return
            if manual_pending:
                await self._enter_pending(discussion_id, phase, manual_pending)
                return
            await self._finish_phase(discussion_id, phase)

    async def _enter_pending(
        self, discussion_id: str, phase: str,
        manual_agents: list[AgentConfig],
    ) -> None:
        """수동 에이전트 차례 — 복사 페이로드를 UI 로 보내고 PENDING 으로 마킹.

        Future 를 들고 대기하지 않는다 — status 를 DB 에 PENDING_MANUAL_INPUT 으로
        남기고 메모리 자원을 즉시 반환한다. 이후 /manual-response 가 재개한다.
        """
        state = await self._load(discussion_id)
        if state is None:
            return
        # 같은 단계의 선행 의견은 '순차' 단계에서만 맥락으로 넣는다. 동시 단계는
        # 서로의 발제를 보면 안 되므로 빈 맥락으로 복사본을 만든다.
        spec = _format_of(state).phase(phase)
        prior = (
            list(state.record_for_phase(phase))
            if spec is not None and spec.sequential else []
        )
        for agent in manual_agents:
            deep = generate_deep_copy(state, agent.agent_id, phase, prior)
            general = generate_general_copy(state, agent.agent_id, phase, prior)
            await self._emit(
                discussion_id, WSMessageType.MANUAL_INPUT_REQUIRED,
                {
                    "agent_id": agent.agent_id, "phase": phase,
                    "deep_copy": deep, "general_copy": general,
                },
            )
        await self._commit(
            discussion_id,
            lambda s: _set_status(s, DiscussionStatus.PENDING_MANUAL_INPUT),
        )
        logger.info(
            "토론 %s — 수동 입력 대기 진입 (%s, %d명) — 메모리 반환",
            discussion_id, phase, len(manual_agents),
        )

    async def _enter_review(
        self, discussion_id: str, phase: str, agent: AgentConfig,
        prior_turns: list[AgentTurn],
    ) -> None:
        """가로채기된 API 에이전트 차례 — 초안·사고흐름을 만들고 PENDING_REVIEW 로.

        에이전트가 곧바로 포스팅하지 않고 사고흐름과 발언 초안을 먼저 생성한다.
        토론은 PENDING_REVIEW 로 대기하며 진행자의 문답·승인을 기다린다 —
        manual 복붙 터널과 평행한 새 대기 상태다(메모리 자원 즉시 반환).
        """
        state = await self._load(discussion_id)
        if state is None:
            return
        try:
            reasoning, draft = await self._draft_with_reasoning(
                state, phase, agent, prior_turns)
        except Exception as exc:  # noqa: BLE001 - 우아한 부분 실패 수용
            logger.warning("검토 초안 생성 실패(%s): %r", agent.agent_id, exc)
            reasoning, draft = "", f"[시스템 경고: 초안 생성 실패 - {exc}]"
        review = ReviewState(agent_id=agent.agent_id, phase=phase,
                             reasoning=reasoning, draft=draft)
        await self._commit(discussion_id, lambda s: _set_review(s, review))
        await self._emit(
            discussion_id, WSMessageType.REVIEW_REQUIRED,
            {"review": review.model_dump(mode="json")},
        )
        logger.info(
            "토론 %s — 검토 대기 진입 (%s, 에이전트 %s) — 메모리 반환",
            discussion_id, phase, agent.agent_id,
        )

    async def _draft_with_reasoning(
        self, state: DiscussionState, phase: str, agent: AgentConfig,
        prior_turns: list[AgentTurn],
    ) -> tuple[str, str]:
        """발언 전 사고흐름 + 초안을 한 번의 LLM 호출로 생성한다."""
        system, user = _build_prompt(state, agent, phase, prior_turns)
        user += (
            "\n\n[검토 모드] 최종 발언을 곧바로 쓰지 말고, 먼저 너의 사고 과정을 "
            "적은 뒤 발언 초안을 작성하라. 정확히 아래 형식으로만 출력하라:\n"
            "[사고흐름]\n(왜 그렇게 판단했는지 간단히)\n[초안]\n(발언 초안)"
        )
        content = await self._invoke_agent(agent, system, user)
        return _split_reasoning_draft(content)

    async def _finish_phase(self, discussion_id: str, phase: str) -> None:
        """단계 인스턴스 종료 — 요약 생성 -> 게이트(WAITING) 또는 종료(COMPLETED).

        다음 인스턴스는 ``plan_next`` 가 결정한다 — 정적 형식이면 다음 단계, 반복
        단계면 합의 근접도에 따라 같은 단계의 다음 라운드 또는 후속 단계.
        """
        state = await self._load(discussion_id)
        if state is None:
            return
        fmt = _format_of(state)
        summary = await self._summarize_phase(state, phase,
                                              list(state.record_for_phase(phase)))
        # 반복 단계 게이트 + 사회자 → 사회자가 라운드 지속 여부를 판단한다.
        # 결정은 facilitator_notes 에 남아 ADVANCE 경로의 plan_next 도 재사용한다.
        spec = fmt.phase(phase)
        facilitated_gate = (
            spec is not None and spec.repeatable
            and state.facilitator is not None)
        decision = None
        if facilitated_gate:
            decision = await self._run_facilitator(
                discussion_id, "decision", phase)
        nxt = plan_next(fmt, phase, summary.convergence_score, decision)

        def mutate(s: DiscussionState) -> None:
            if not any(ps.phase == phase for ps in s.phase_summaries):
                s.phase_summaries.append(summary)
            if nxt is None:
                s.current_phase = PHASE_COMPLETED
                s.status = DiscussionStatus.COMPLETED
            else:
                s.status = DiscussionStatus.WAITING_FOR_USER
            s.touch()

        state = await self._commit(discussion_id, mutate)
        payload: dict[str, object] = {
            "phase": phase, "summary": summary.model_dump(mode="json"),
        }
        if state.final_joint_agreement is not None:
            payload["final_joint_agreement"] = state.final_joint_agreement
        await self._emit(discussion_id, WSMessageType.PHASE_COMPLETED, payload)
        if nxt is None:
            await self._run_facilitator(discussion_id, "close", PHASE_COMPLETED)
            await self._emit(
                discussion_id, WSMessageType.DISCUSSION_COMPLETED,
                {"discussion_id": discussion_id,
                 "final_joint_agreement": state.final_joint_agreement},
            )
        else:
            # 반복 게이트는 'decision' 노트가 진행 코멘트를 겸한다 — between 생략.
            if not facilitated_gate:
                await self._run_facilitator(discussion_id, "between", phase)
            nxt_spec = fmt.phase(nxt)
            await self._emit(
                discussion_id, WSMessageType.AWAITING_USER,
                {"completed_phase": phase, "next_phase": nxt,
                 "next_round": round_of_key(nxt),
                 "next_label": _instance_label(nxt_spec, round_of_key(nxt)),
                 "facilitator_decision": decision},
            )

    async def _end_discussion(self, discussion_id: str) -> None:
        """게이트 구간에서 토론을 조기 종료한다 — 남은 단계는 진행하지 않는다.

        합의 근접도가 충분히 높을 때 유저가 누르는 '여기서 종료' 의 진입점이다.
        """
        state = await self._commit(discussion_id, _mark_completed)
        await self._run_facilitator(discussion_id, "close", PHASE_COMPLETED)
        await self._emit(
            discussion_id, WSMessageType.DISCUSSION_COMPLETED,
            {"discussion_id": discussion_id,
             "final_joint_agreement": state.final_joint_agreement},
        )

    async def _run_facilitator(
        self, discussion_id: str, kind: str, phase: str
    ) -> Optional[str]:
        """사회자 훅 — 단계 경계에서 진행 노트를 생성·기록·브로드캐스트한다.

        사회자는 토론 진행을 막지 않는 부가 레이어다. LLM 호출이 실패하거나 빈
        응답이면 — 개회·중간·폐회는 *보이는 경고 노트*로 남겨(침묵 금지) 사용자가
        원인을 알 수 있게 하고, 진행 결정(``kind="decision"``)은 노트 없이 None 을
        반환해 합의 근접도 숫자 폴백에 맡긴다.
        """
        state = await self._load(discussion_id)
        if state is None or state.facilitator is None:
            return None
        # 멱등 — 같은 (kind, phase) 노트가 이미 있으면 LLM 재호출 없이 그 결정을 쓴다.
        for note in state.facilitator_notes:
            if note.kind == kind and note.phase == phase:
                return note.decision
        system, user = _build_facilitator_prompt(state, kind, phase)
        decision: Optional[str] = None
        failed = False
        try:
            text, usage = await self._invoke_agent_usage(
                state.facilitator, system, user)
            content = text.strip()
            if not content:
                raise DiscussionError("사회자 응답이 비어 있습니다.")
        except Exception as exc:  # noqa: BLE001 - 사회자 실패는 비치명적
            logger.warning("사회자 호출 실패(%s/%s): %r", kind, phase, exc)
            if kind == "decision":
                return None   # 진행 결정 실패 → plan_next 가 숫자 근접도로 폴백
            content = f"[시스템 경고: 사회자 응답 생성 실패 - {exc}]"
            usage, failed = {}, True
        else:
            decision = (
                _parse_facilitator_decision(content) if kind == "decision"
                else None)
        meta: dict = {"usage": usage}
        if failed:
            meta["failed"] = True
        note = FacilitatorNote(
            phase=phase, kind=kind, content=content, decision=decision,
            metadata=meta)
        await self._commit(
            discussion_id, lambda s: _append_facilitator_note(s, note))
        await self._emit(
            discussion_id, WSMessageType.FACILITATOR_NOTE,
            {"note": note.model_dump(mode="json")})
        return decision

    async def _run_consensus(self, discussion_id: str, phase: str) -> None:
        """마지막 단계 force_consensus=True — 단일 합의안 문서를 합성한다."""
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
        await self._finish_phase(discussion_id, phase)

    # ----- 에이전트 호출 -----
    async def _do_api_turn(
        self, state: DiscussionState, phase: str,
        agent: AgentConfig, prior_turns: list[AgentTurn],
    ) -> AgentTurn:
        """한 API 에이전트의 턴 — 스트리밍 LLM 호출. 실패는 시스템 경고 턴으로."""
        try:
            system, user = _build_prompt(state, agent, phase, prior_turns)

            async def on_token(token: str) -> None:
                await self._emit(
                    state.discussion_id, WSMessageType.TOKEN_STREAM,
                    {"agent_id": agent.agent_id, "phase": phase,
                     "token": token},
                )

            content, usage = await self._invoke_agent_usage(
                agent, system, user, on_token)
            if not content.strip():
                raise DiscussionError("응답이 비어 있습니다.")
            return AgentTurn(
                agent_id=agent.agent_id, phase=phase, content=content.strip(),
                metadata={"provider": agent.get_provider().value,
                          "model": agent.model, "usage": usage},
            )
        except Exception as exc:  # noqa: BLE001 - 우아한 부분 실패 수용
            logger.warning("에이전트 %s 응답 실패: %r", agent.agent_id, exc)
            return _failure_turn(agent, phase, exc)

    async def _gather_api_turns(
        self, state: DiscussionState, phase: str,
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
        """LLM 을 호출해 응답 텍스트만 반환한다.

        토큰 사용량이 필요하면 ``_invoke_agent_usage`` 를 직접 쓴다.
        """
        text, _usage = await self._invoke_agent_usage(
            agent, system, user, on_token)
        return text

    async def _invoke_agent_usage(
        self, agent: AgentConfig, system: str, user: str,
        on_token: Optional[TokenCallback] = None,
    ) -> tuple[str, dict]:
        """공급자별 스트리밍 LLM 호출 — (응답 텍스트, 토큰 사용량)을 반환한다.

        실제 네트워크·추론 비용이 드는 ``_call_*`` 호출은 고정 크기 세마포어
        (``self._llm_semaphore``) 안에서만 수행한다 — 동시 LLM 호출 수를 엄격히
        제한해 CPU 폭주와 낙관적 락 경합 폭증을 막는 단일 통제점이다.
        ``_call_*`` 가 (text, usage) 튜플을 주면 그대로, 문자열만 주면(테스트
        가짜 등) usage 를 빈 dict 로 본다.
        """
        provider = agent.get_provider()
        client = self._pool.get(provider)
        async with self._llm_semaphore:
            if provider in (ModelProvider.OPENAI, ModelProvider.DEEPSEEK,
                            ModelProvider.GEMINI):
                # DeepSeek·Gemini 는 OpenAI 호환 — 같은 스트리밍 경로 재사용.
                result = await _call_openai(
                    client, agent.model, system, user,
                    agent.temperature, agent.max_tokens, on_token)
            elif provider is ModelProvider.ANTHROPIC:
                result = await _call_anthropic(
                    client, agent.model, system, user,
                    agent.temperature, agent.max_tokens, on_token)
            elif provider is ModelProvider.OLLAMA:
                result = await _call_ollama(
                    client, agent.model, system, user,
                    agent.temperature, agent.max_tokens, on_token)
            else:
                raise DiscussionError(f"미지원 LLM 공급자: {provider}")
        if isinstance(result, tuple):
            return result[0], (result[1] or {})
        return result, {}

    async def refine_persona(
        self, *, topic: str, draft: str, provider: ModelProvider,
        model: str, name: str = "", persona_role: str = "",
    ) -> str:
        """페르소나 초안을 토론 주제 맥락에 맞춰 윤문한다.

        지정된 ``provider``/``model`` (보통 해당 에이전트 슬롯의 설정)로 LLM 을
        호출한다. 모든 LLM 경로와 동일하게 ``_invoke_agent`` 를 거치므로 동시
        호출 세마포어의 백프레셔도 함께 적용된다.
        """
        role_hint = (
            f" 이 에이전트의 역할 분류는 '{persona_role}' 다." if persona_role else ""
        )
        system = (
            "너는 다자 브레인스토밍·토론 세션에 투입할 에이전트의 '페르소나 "
            "프롬프트'를 다듬는 편집자다. 사용자가 대강 적어 둔 초안을, 주어진 "
            "주제에 어울리는 명확하고 구체적인 1인칭 지시문으로 윤문한다. 2~4문장 "
            "으로 간결하게, 초안의 의도는 보존하되 표현을 또렷하게 한다. 결과는 "
            "페르소나 본문만 출력하고 따옴표·머리말·부연 설명은 붙이지 않는다."
        )
        user = (
            f"[토론 주제]\n{topic}\n\n"
            f"[에이전트 이름] {name or '(미정)'}.{role_hint}\n\n"
            f"[페르소나 초안]\n{draft}\n\n"
            "[지시] 위 초안을 주제에 맞춰 윤문한 페르소나 프롬프트만 출력하라."
        )
        refiner = AgentConfig(
            agent_id="_refiner", name="refiner", model=model,
            persona_prompt="", provider=provider, temperature=0.4,
        )
        refined = await self._invoke_agent(refiner, system, user)
        return refined.strip()

    async def _synthesize(self, state: DiscussionState) -> str:
        """직전 단계까지를 종합해 단일 최종 합의안 문서를 생성한다(스트리밍)."""
        last_phase = _format_of(state).phases[-1].id
        history = _render_history(state, last_phase, [])
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
                              "phase": last_phase,
                              "token": token})

        return await self._invoke_agent(_llm_agent(state), system, user, on_token)

    async def _summarize_phase(
        self, state: DiscussionState, phase: str,
        turns: list[AgentTurn],
    ) -> PhaseSummary:
        """단계 요약 메트릭스를 생성한다. 실패해도 파이프라인을 멈추지 않는다."""
        fmt = _format_of(state)
        if (fmt.is_last_phase(phase) and fmt.supports_consensus
                and state.force_consensus):
            return PhaseSummary(phase=phase, convergence_score=1.0)
        if not turns:
            return PhaseSummary(phase=phase)
        try:
            return await self._llm_summarize(state, phase, turns)
        except Exception as exc:  # noqa: BLE001 - 요약 실패는 비치명적
            logger.warning("단계 %s 요약 생성 실패: %r", phase, exc)
            return PhaseSummary(phase=phase, key_conflicts=[f"[요약 생성 실패: {exc}]"])

    async def _llm_summarize(
        self, state: DiscussionState, phase: str,
        turns: list[AgentTurn],
    ) -> PhaseSummary:
        """LLM 에 단계 발언을 분석시켜 PhaseSummary(주장 메트릭스)를 구성한다."""
        name_of = {a.agent_id: a.name for a in state.agents}
        transcript = "\n".join(
            f"[{name_of.get(t.agent_id, t.agent_id)} ({t.agent_id})] {t.content}"
            for t in turns
        )
        agent_ids = [a.agent_id for a in state.agents]
        spec = _format_of(state).phase(phase)
        label = spec.label if spec else phase
        system = (
            "너는 중립적 토론 분석가다. 발언에서 합의점과 이견을 한쪽으로 치우치지 "
            "않고 균형 있게 식별한다. 요청한 JSON 객체만 출력하고 다른 설명은 하지 "
            "않는다."
        )
        # 합의 근접도를 '느낌 숫자' 로 바로 받지 않는다 — 먼저 쟁점을 항목별로 분해해
        # 합의/부분합의/대립으로 분류시키고(issue_points), 그 분포에서 점수를 계산한다.
        # 채점 앵커를 명시해 '입장/프레임 일치' 가 0 점으로 과소평가되는 것을 막는다.
        # 또한 직전 단계들의 추이를 함께 줘, 이번 단계가 그 연장선임을 보게 한다.
        trajectory = _render_convergence_trajectory(state)
        traj_block = f"{trajectory}\n\n" if trajectory else ""
        user = (
            f"다음은 '{label}' 단계의 발언이다.\n"
            f"{transcript}\n\n"
            f"{traj_block}"
            "[작업]\n"
            "1) 이 단계에서 다뤄진 '쟁점 항목'을 모두 나열하고, 각각을 다음 중 "
            "하나로 분류하라:\n"
            "   - agreed: 참가자들이 사실상 같은 결론·입장에 도달\n"
            "   - partial: 큰 틀(프레임/방향)은 같으나 세부에서 갈림\n"
            "   - contested: 입장이 명확히 갈림\n"
            "2) 각 참가자의 입장을 요약하라.\n"
            "3) convergence_score(0.0~1.0)를 매겨라. 앵커:\n"
            "   - 0.0 모든 핵심 쟁점이 대립 / 0.25 공통 전제는 있으나 결론 대부분 대립\n"
            "   - 0.5 핵심 쟁점 절반쯤 합의 / 0.75 대부분 합의, 세부만 이견\n"
            "   - 1.0 모든 핵심 쟁점 합의(표현 차이만 남음)\n"
            "   ※ 참가자들이 같은 프레임·입장을 채택했다면 세부 구현 이견이 남아도 "
            "그 자체로 0.6 이상이다. 토론은 단계가 진행될수록 수렴하는 경향이 "
            "있으니, 위 추이가 있다면 새 합의가 생겼을 때 직전보다 낮게 매기지 "
            "말라(새 이견이 불거진 경우는 예외).\n\n"
            "아래 JSON 스키마로만 응답하라:\n"
            '{"agent_summaries": [{"agent_id": "...", "initial_claim": "...", '
            '"current_stance": "...", "stance_shift": "..."}], '
            '"issue_points": [{"issue": "...", "status": "agreed|partial|contested"}], '
            '"key_conflicts": ["..."], "convergence_score": 0.0}\n'
            f"- agent_id 는 반드시 다음 중 하나: {agent_ids}\n"
            "- 모든 텍스트는 한국어."
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
        score = _convergence_from_summary(data)
        # key_conflicts 가 비어 있으면 대립(contested) 항목으로 채운다.
        conflicts = [str(c) for c in data.get("key_conflicts", []) if c]
        if not conflicts:
            conflicts = [
                str(p.get("issue", "")) for p in data.get("issue_points", [])
                if isinstance(p, dict)
                and str(p.get("status", "")).lower() == "contested"
                and p.get("issue")
            ]
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
def _mark_phase_start(state: DiscussionState, phase: str) -> None:
    """단계 진입 마킹. 멱등."""
    state.current_phase = phase
    state.status = DiscussionStatus.RUNNING
    state.touch()


def _record_turns(
    state: DiscussionState, phase: str, turns: list[AgentTurn]
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


def _set_review(state: DiscussionState, review: ReviewState) -> None:
    """검토 세션을 설정하고 PENDING_REVIEW 로 전이한다. 멱등."""
    state.review = review
    state.status = DiscussionStatus.PENDING_REVIEW
    state.touch()


def _clear_review(state: DiscussionState) -> None:
    """검토 세션을 종료하고 RUNNING 으로 되돌린다 (승인 후). 멱등."""
    state.review = None
    if state.status is DiscussionStatus.PENDING_REVIEW:
        state.status = DiscussionStatus.RUNNING
    state.touch()


def _append_review_qa(state: DiscussionState, exchange: ReviewExchange) -> None:
    """검토 문답 1쌍을 추가한다. 멱등 — 같은 (created_at, question) 은 건너뛴다."""
    if state.review is None:
        return
    already = any(
        x.created_at == exchange.created_at and x.question == exchange.question
        for x in state.review.qa
    )
    if not already:
        state.review.qa.append(exchange)
    state.touch()


def _append_facilitator_note(
    state: DiscussionState, note: FacilitatorNote
) -> None:
    """사회자 노트를 추가한다. 멱등 — 같은 (kind, phase) 노트는 한 번만 적재한다."""
    already = any(
        n.kind == note.kind and n.phase == note.phase
        for n in state.facilitator_notes
    )
    if not already:
        state.facilitator_notes.append(note)
    state.touch()


def _set_intercepts(state: DiscussionState, agent_ids: list[str]) -> None:
    """가로채기 대상 에이전트 목록을 설정한다. 멱등."""
    state.intercept_agents = list(agent_ids)
    state.touch()


def _mark_completed(state: DiscussionState) -> None:
    """토론을 COMPLETED 로 종료 전이한다. 멱등."""
    state.current_phase = PHASE_COMPLETED
    state.status = DiscussionStatus.COMPLETED
    state.touch()
