"""schemas.py — 공급자 추론, 모델 검증, 상태 헬퍼."""
import pytest
from pydantic import ValidationError

from app.schemas import (
    AgentConfig,
    DiscussionPhase,
    DiscussionState,
    ModelProvider,
)


def _agent(model="gpt-4o-mini", provider=None, agent_id="a", name="A"):
    return AgentConfig(agent_id=agent_id, name=name, model=model,
                       persona_prompt="p", provider=provider)


@pytest.mark.parametrize("model,expected", [
    ("gpt-4o-mini", ModelProvider.OPENAI),
    ("o3-mini", ModelProvider.OPENAI),
    ("chatgpt-4o-latest", ModelProvider.OPENAI),
    ("claude-opus-4-7", ModelProvider.ANTHROPIC),
    ("llama3.1", ModelProvider.OLLAMA),
    ("mistral-large", ModelProvider.OLLAMA),
    ("qwen2.5", ModelProvider.OLLAMA),
])
def test_get_provider_inferred_from_model(model, expected):
    assert _agent(model).get_provider() is expected


def test_get_provider_explicit_overrides_inference():
    # model 명이 gpt-* 라도 명시한 provider 가 우선한다
    a = _agent("gpt-4o-mini", provider=ModelProvider.OLLAMA)
    assert a.get_provider() is ModelProvider.OLLAMA


def test_get_provider_unrecognized_raises():
    with pytest.raises(ValueError):
        _agent("totally-unknown-model").get_provider()


def test_manual_provider_must_be_explicit():
    a = _agent("manual-x", provider=ModelProvider.MANUAL)
    assert a.get_provider() is ModelProvider.MANUAL


def test_agent_requires_persona_prompt():
    with pytest.raises(ValidationError):
        AgentConfig(agent_id="a", name="A", model="gpt-4o-mini")


def test_agent_temperature_out_of_bounds_rejected():
    with pytest.raises(ValidationError):
        AgentConfig(agent_id="a", name="A", model="gpt-4o-mini",
                    persona_prompt="p", temperature=3.0)


def test_discussion_requires_two_agents():
    with pytest.raises(ValidationError):
        DiscussionState(discussion_id="d", topic="t",
                        agents=[_agent(agent_id="a1")])


def test_discussion_defaults():
    s = DiscussionState(discussion_id="d", topic="t",
                        agents=[_agent(agent_id="a1"),
                                _agent("claude-x", agent_id="a2")])
    assert s.status.value == "created"
    assert s.current_phase is DiscussionPhase.IDLE
    assert s.version == 0
    assert s.force_consensus is False
    assert s.final_joint_agreement is None


def test_record_for_phase_returns_the_phase_list():
    s = DiscussionState(discussion_id="d", topic="t",
                        agents=[_agent(agent_id="a1"),
                                _agent("claude-x", agent_id="a2")])
    assert s.record_for_phase(DiscussionPhase.PHASE_1_OPINION) is s.phase_1_opinions
    assert s.record_for_phase(DiscussionPhase.PHASE_5_CONCLUSION) is s.phase_5_conclusions


def test_record_for_phase_rejects_non_record_phase():
    s = DiscussionState(discussion_id="d", topic="t",
                        agents=[_agent(agent_id="a1"),
                                _agent("claude-x", agent_id="a2")])
    with pytest.raises(ValueError):
        s.record_for_phase(DiscussionPhase.IDLE)
