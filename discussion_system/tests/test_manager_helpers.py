"""manager.py 순수 헬퍼 — 환경변수 파싱, 시스템 LLM 에이전트 선택, 단계 전이."""
import pytest

from app import manager
from app.manager import _llm_agent, _next_phase, _positive_int_env
from app.schemas import AgentConfig, DiscussionPhase, DiscussionState, ModelProvider


def _agent(agent_id, provider):
    return AgentConfig(agent_id=agent_id, name=agent_id, model="m",
                       persona_prompt="p", provider=provider)


def _state(*providers):
    agents = [_agent(f"a{i}", p) for i, p in enumerate(providers)]
    return DiscussionState(discussion_id="d", topic="t", agents=agents)


@pytest.mark.parametrize("raw,expected", [
    (None, 8), ("", 8), ("notanumber", 8), ("0", 8), ("-3", 8),
    ("1", 1), ("12", 12),
])
def test_positive_int_env(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("AGORA_TEST_VAR", raising=False)
    else:
        monkeypatch.setenv("AGORA_TEST_VAR", raw)
    assert _positive_int_env("AGORA_TEST_VAR", 8) == expected


def test_only_phase_2_is_sequential():
    # 회귀 방지: 1단계는 동시 발제 단계 — 순차 집합엔 2단계만 있어야 한다.
    assert manager._SEQUENTIAL_PHASES == frozenset({DiscussionPhase.PHASE_2_CRITIQUE})


def test_llm_agent_picks_first_non_manual():
    state = _state(ModelProvider.MANUAL, ModelProvider.OPENAI,
                   ModelProvider.ANTHROPIC)
    chosen = _llm_agent(state)
    assert chosen.agent_id == "a1"
    assert chosen.get_provider() is ModelProvider.OPENAI


def test_llm_agent_falls_back_when_all_manual():
    # 전원 manual — manual 로는 시스템 호출 불가하므로 폴백 모델을 쓴다.
    chosen = _llm_agent(_state(ModelProvider.MANUAL, ModelProvider.MANUAL))
    assert chosen.get_provider() is not ModelProvider.MANUAL
    assert chosen.model == manager._FALLBACK_LLM_MODEL


def test_fallback_model_default_is_gpt_4o_mini():
    assert manager._FALLBACK_LLM_MODEL == "gpt-4o-mini"


def test_next_phase_sequence():
    assert _next_phase(DiscussionPhase.PHASE_1_OPINION) is \
        DiscussionPhase.PHASE_2_CRITIQUE
    assert _next_phase(DiscussionPhase.PHASE_4_REVISION) is \
        DiscussionPhase.PHASE_5_CONCLUSION
    assert _next_phase(DiscussionPhase.PHASE_5_CONCLUSION) is None
