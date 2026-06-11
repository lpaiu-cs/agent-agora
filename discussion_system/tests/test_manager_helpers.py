"""manager.py 순수 헬퍼 — 환경변수 파싱, 시스템 LLM 에이전트 선택, 합의 근접도."""
import pytest

from app import manager
from app.manager import (
    _convergence_from_summary,
    _llm_agent,
    _positive_int_env,
    _render_convergence_trajectory,
    _split_reasoning_draft,
    extract_embedded_state,
    render_transcript,
    render_transcript_with_state,
)
from app.schemas import (
    AgentConfig,
    AgentTurn,
    DiscussionState,
    FacilitatorNote,
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


# ---------------------------------------------------------------------------
# .md 내장 상태 블록 — 저장·복원 라운드트립
# ---------------------------------------------------------------------------
def _rich_state():
    """본문에 '-->'·마크다운 헤더 같은 함정 문자를 넣은 상태 (블록 견고성 검증)."""
    state = _state(ModelProvider.OPENAI, ModelProvider.MANUAL)
    state.topic = "## 마크다운 주제\n화살표 A --> B 도 있다"
    state.phase_records = {"opinion": [
        AgentTurn(agent_id="a0", phase="opinion", content="## 헤더 발언 --> 끝",
                  metadata={"usage": {"prompt_tokens": 3, "completion_tokens": 7}}),
    ]}
    state.phase_summaries = [PhaseSummary(phase="opinion", convergence_score=0.7)]
    state.facilitator_notes = [
        FacilitatorNote(phase="opinion", kind="between", content="사회자 노트")]
    return state


def test_transcript_state_block_round_trip():
    doc = render_transcript_with_state(_rich_state())
    assert "AGORA-STATE-V1" in doc
    # 사람용 본문은 그대로 시작하고, 상태 블록은 끝의 HTML 주석이다.
    assert doc.startswith(render_transcript(_rich_state())[:50])
    restored = extract_embedded_state(doc)
    assert restored is not None
    assert restored.topic == "## 마크다운 주제\n화살표 A --> B 도 있다"
    assert restored.phase_records["opinion"][0].content == "## 헤더 발언 --> 끝"
    assert restored.phase_records["opinion"][0].metadata["usage"][
        "completion_tokens"] == 7
    assert restored.phase_summaries[0].convergence_score == 0.7
    assert restored.facilitator_notes[0].kind == "between"
    assert [a.agent_id for a in restored.agents] == ["a0", "a1"]


def test_extract_embedded_state_absent_or_corrupt_returns_none():
    # 구버전 파일(블록 없음) → None (호출부가 패턴 파싱으로 폴백).
    assert extract_embedded_state(render_transcript(_rich_state())) is None
    # 손상된 base64 → None (예외 없이 조용히 폴백).
    assert extract_embedded_state(
        "# 문서\n<!-- AGORA-STATE-V1\n!!!깨진페이로드!!!\n-->") is None


def test_split_reasoning_draft():
    reasoning, draft = _split_reasoning_draft(
        "[사고흐름]\n이렇게 생각한다\n[초안]\n발언 초안 본문")
    assert reasoning == "이렇게 생각한다"
    assert draft == "발언 초안 본문"
    # [초안] 마커가 없으면 전체를 초안으로 본다
    reasoning, draft = _split_reasoning_draft("마커 없는 텍스트")
    assert reasoning == ""
    assert draft == "마커 없는 텍스트"
