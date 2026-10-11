"""manager.py 순수 헬퍼 — 환경변수 파싱, 시스템 LLM 에이전트 선택, 합의 근접도."""
import base64
import re

import pytest

from app import manager
from app.manager import (
    _convergence_from_summary,
    _intervention_lines,
    _llm_agent,
    _parse_facilitator_targets,
    _positive_int_env,
    _render_convergence_trajectory,
    _split_reasoning_draft,
    decode_meta_hint,
    encode_meta_hint,
    extract_embedded_state,
    render_transcript,
    render_transcript_with_meta,
    total_token_usage,
)
from app.schemas import (
    AgentConfig,
    AgentTurn,
    DiscussionState,
    FacilitatorNote,
    ModelProvider,
    PhaseSummary,
    UserIntervention,
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
# 지향 개입 / 사회자 지목 질문 — 순수 헬퍼
# ---------------------------------------------------------------------------
def test_intervention_lines_respect_target():
    state = _state(ModelProvider.OPENAI, ModelProvider.OPENAI)   # a0, a1
    ivs = [
        UserIntervention(message="전체 공지"),
        UserIntervention(message="a0 비밀 지시", target_agent_id="a0"),
    ]
    # a0 시점 — 전체 공지 + 자기 지향('너에게') 둘 다.
    a0 = _intervention_lines(state, ivs, "a0")
    assert a0 == ["[참가자 H] 전체 공지", "[참가자 H → 너에게] a0 비밀 지시"]
    # a1 시점 — 남에게 지향된 개입은 숨겨진다.
    assert _intervention_lines(state, ivs, "a1") == ["[참가자 H] 전체 공지"]
    # 전체 시점(None) — 대상 이름 표기와 함께 모두 보인다.
    all_view = _intervention_lines(state, ivs, None)
    assert "[참가자 H → a0] a0 비밀 지시" in all_view


def test_parse_facilitator_targets():
    text = ("다음 단계의 초점은 비용 검증이다.\n"
            "[지목] 알파: 누수 임계값을 숫자로 제시하라\n"
            "[지목] 베타: decay 상수의 근거는?\n"
            "맺음말.")
    targets = _parse_facilitator_targets(text)
    assert targets == {"알파": "누수 임계값을 숫자로 제시하라",
                       "베타": "decay 상수의 근거는?"}
    assert _parse_facilitator_targets("지목 없는 코멘트") == {}


def test_total_token_usage_counts_turns_and_facilitator_notes():
    state = _state(ModelProvider.OPENAI, ModelProvider.OPENAI)
    state.phase_records = {"opinion": [
        AgentTurn(agent_id="a0", phase="opinion", content="x",
                  metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 50}}),
        AgentTurn(agent_id="a1", phase="opinion", content="y",
                  metadata={}),   # usage 없는 턴(수동 등)은 0 으로
    ]}
    state.facilitator_notes = [FacilitatorNote(
        phase="opinion", kind="between", content="노트",
        metadata={"usage": {"prompt_tokens": 30, "completion_tokens": 20}})]
    usage = total_token_usage(state)
    assert usage == {"prompt": 130, "completion": 70, "total": 200}
    # 빈 토론은 전부 0.
    assert total_token_usage(
        _state(ModelProvider.OPENAI, ModelProvider.OPENAI))["total"] == 0


# ---------------------------------------------------------------------------
# .md 복원 힌트(A2) — 초소형 인코딩 + 정확한 길이 앵커 / 옛 V1 블록 하위호환
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


def test_meta_hint_round_trip():
    state = _rich_state()
    doc = render_transcript_with_meta(state)
    # 사람용 본문은 그대로 시작하고, 힌트는 끝의 한 줄 HTML 주석(A2)이다.
    assert doc.startswith(render_transcript(state)[:50])
    block = doc[doc.index("<!-- A2 "):]
    # 발언 원문을 중복 저장하지 않으므로 블록은 본문 대비 아주 짧다.
    assert "\n" not in block.strip() and len(block) < 200
    payload = re.search(r"<!-- A2 (\S+) -->", doc).group(1)
    meta = decode_meta_hint(payload)
    assert "fmt" not in meta                       # format_id 는 본문에서 읽으므로 미포함
    assert meta["types"] == [a.persona_type for a in state.agents]
    # 길이 앵커 = 본문에 펼친 모든 자유 텍스트(주제·발언·노트…)의 정확한 길이, 본문 순서.
    blocks = manager._render_transcript_parts(state)[1]
    assert blocks == [state.topic, "## 헤더 발언 --> 끝", "사회자 노트"]
    assert meta["version"] == 5
    assert meta["lens"] == [len(b) for b in blocks]


def test_meta_hint_keeps_exact_long_turn_length():
    """8K를 넘는 발언도 A2 앵커가 포화·절단되지 않는다."""
    state = _rich_state()
    content = "앞부분\n## 본문 안의 헤더\n" + "x" * 9_000
    state.phase_records["opinion"][0].content = content
    meta = decode_meta_hint(encode_meta_hint(state))
    assert meta["lens"][1] == len(content)   # [주제, 발언, 노트] 순서


def test_decode_meta_hint_rejects_unknown_version():
    # v3~v5 외 버전(예: 손상/구버전 바이트)은 None 으로 조용히 폴백.
    assert decode_meta_hint(base64.b64encode(bytes([2, 0, 0])).decode()) is None


def test_legacy_v1_block_still_decodes():
    """예전 V1 전체상태 블록이 든 .md 는 그대로 완벽 복원된다(하위 호환)."""
    state = _rich_state()
    payload = base64.b64encode(state.model_dump_json().encode()).decode("ascii")
    doc = render_transcript(state) + f"\n<!-- AGORA-STATE-V1\n{payload}\n-->\n"
    restored = extract_embedded_state(doc)
    assert restored is not None
    assert restored.phase_records["opinion"][0].content == "## 헤더 발언 --> 끝"
    assert restored.phase_summaries[0].convergence_score == 0.7
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
