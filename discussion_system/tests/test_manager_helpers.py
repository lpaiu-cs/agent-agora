"""manager.py 순수 헬퍼 — 환경변수 파싱, 시스템 LLM 에이전트 선택."""
import pytest

from app import manager
from app.manager import _llm_agent, _positive_int_env, _split_reasoning_draft
from app.schemas import AgentConfig, DiscussionState, ModelProvider


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


def test_split_reasoning_draft():
    reasoning, draft = _split_reasoning_draft(
        "[사고흐름]\n이렇게 생각한다\n[초안]\n발언 초안 본문")
    assert reasoning == "이렇게 생각한다"
    assert draft == "발언 초안 본문"
    # [초안] 마커가 없으면 전체를 초안으로 본다
    reasoning, draft = _split_reasoning_draft("마커 없는 텍스트")
    assert reasoning == ""
    assert draft == "마커 없는 텍스트"
