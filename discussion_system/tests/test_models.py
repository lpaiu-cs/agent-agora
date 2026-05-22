"""models.py — DiscussionState ↔ ORM 행 변환 라운드트립."""
from app.models import row_to_state, state_to_columns, state_to_row
from app.schemas import (
    AgentConfig,
    AgentTurn,
    DiscussionPhase,
    DiscussionState,
    ModelProvider,
    PhaseSummary,
    UserIntervention,
)


def _rich_state():
    agents = [
        AgentConfig(agent_id="a1", name="알파", model="gpt-4o-mini",
                    persona_prompt="알파 페르소나"),
        AgentConfig(agent_id="a2", name="베타", model="claude-x",
                    persona_prompt="베타 페르소나", provider=ModelProvider.MANUAL),
    ]
    s = DiscussionState(discussion_id="d-rich", topic="라운드트립 주제",
                        agents=agents, force_consensus=True)
    s.phase_1_opinions.append(
        AgentTurn(agent_id="a1", phase=DiscussionPhase.PHASE_1_OPINION,
                  content="알파의 1단계 발제"))
    s.phase_summaries.append(
        PhaseSummary(phase=DiscussionPhase.PHASE_1_OPINION, convergence_score=0.5))
    s.user_interventions.append(UserIntervention(message="진행자 개입"))
    s.version = 3
    return s


def test_state_to_columns_has_scalar_and_json_fields():
    cols = state_to_columns(_rich_state())
    for field in ("discussion_id", "topic", "status", "version", "agents",
                  "phase_1_opinions", "phase_summaries", "user_interventions"):
        assert field in cols


def test_round_trip_preserves_all_fields():
    original = _rich_state()
    restored = row_to_state(state_to_row(original))

    assert restored.discussion_id == "d-rich"
    assert restored.topic == "라운드트립 주제"
    assert restored.force_consensus is True
    assert restored.version == 3
    assert len(restored.agents) == 2
    assert restored.agents[1].get_provider() is ModelProvider.MANUAL
    assert len(restored.phase_1_opinions) == 1
    assert restored.phase_1_opinions[0].content == "알파의 1단계 발제"
    assert restored.phase_1_opinions[0].phase is DiscussionPhase.PHASE_1_OPINION
    assert restored.phase_summaries[0].convergence_score == 0.5
    assert restored.user_interventions[0].message == "진행자 개입"


def test_round_trip_empty_collections():
    s = DiscussionState(
        discussion_id="d-empty", topic="t",
        agents=[AgentConfig(agent_id="a1", name="A", model="gpt-4o-mini",
                            persona_prompt="p"),
                AgentConfig(agent_id="a2", name="B", model="claude-x",
                            persona_prompt="p")])
    restored = row_to_state(state_to_row(s))
    assert restored.phase_1_opinions == []
    assert restored.phase_summaries == []
    assert restored.user_interventions == []
