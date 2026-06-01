"""manager.py 순수 헬퍼 — 환경변수 파싱, 시스템 LLM 에이전트 선택, 합의 근접도."""
import pytest

from app import manager
from app.manager import (
    _convergence_from_summary,
    _llm_agent,
    _positive_int_env,
    _render_convergence_trajectory,
    _split_reasoning_draft,
)
from app.schemas import (
    AgentConfig,
    DiscussionState,
    ModelProvider,
    PhaseSummary,
)


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


def test_convergence_from_issue_points_distribution():
    # 모든 쟁점 합의 → 1.0, 모든 쟁점 대립 → 0.0.
    assert _convergence_from_summary(
        {"issue_points": [{"status": "agreed"}, {"status": "agreed"}]}) == 1.0
    assert _convergence_from_summary(
        {"issue_points": [{"status": "contested"}, {"status": "contested"}]}) == 0.0
    # 합의2·부분합의1·대립1 → (1+1+0.5+0)/4 = 0.625, 직출 점수 없으면 그대로.
    score = _convergence_from_summary({"issue_points": [
        {"status": "agreed"}, {"status": "agreed"},
        {"status": "partial"}, {"status": "contested"}]})
    assert abs(score - 0.625) < 1e-9


def test_convergence_averages_derived_and_holistic():
    # issue_points(파생=1.0) 와 직출 convergence_score(0.4) 가 둘 다 있으면 평균.
    score = _convergence_from_summary({
        "issue_points": [{"status": "agreed"}],
        "convergence_score": 0.4})
    assert abs(score - 0.7) < 1e-9


def test_convergence_falls_back_to_holistic_when_no_points():
    # issue_points 가 없으면 모델의 convergence_score 를 그대로 (구버전 호환).
    assert _convergence_from_summary({"convergence_score": 0.55}) == 0.55
    # 둘 다 없으면 0.0, 범위 밖 값은 클램프.
    assert _convergence_from_summary({}) == 0.0
    assert _convergence_from_summary({"convergence_score": 1.7}) == 1.0


def test_render_convergence_trajectory():
    # 요약된 단계가 없으면 빈 문자열.
    state = _state(ModelProvider.OPENAI, ModelProvider.OPENAI)  # debate 기본
    assert _render_convergence_trajectory(state) == ""
    # 직전 두 단계 요약이 있으면 추이가 단계 순서대로(라벨·%) 렌더된다.
    state.phase_records = {"opinion": [], "critique": []}
    state.phase_summaries = [
        PhaseSummary(phase="opinion", convergence_score=0.45),
        PhaseSummary(phase="critique", convergence_score=0.6),
    ]
    out = _render_convergence_trajectory(state)
    assert "합의 근접도 추이" in out
    assert "45%" in out and "60%" in out
    # opinion(45%)이 critique(60%)보다 먼저 나온다 — 실행 순서.
    assert out.index("45%") < out.index("60%")


def test_split_reasoning_draft():
    reasoning, draft = _split_reasoning_draft(
        "[사고흐름]\n이렇게 생각한다\n[초안]\n발언 초안 본문")
    assert reasoning == "이렇게 생각한다"
    assert draft == "발언 초안 본문"
    # [초안] 마커가 없으면 전체를 초안으로 본다
    reasoning, draft = _split_reasoning_draft("마커 없는 텍스트")
    assert reasoning == ""
    assert draft == "마커 없는 텍스트"
