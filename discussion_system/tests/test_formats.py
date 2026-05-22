"""formats.py — 토론 형식 정의·조회."""
from app.formats import BRAINSTORM, DEBATE, DEFAULT_FORMAT_ID, FORMATS, get_format


def test_registry_has_debate_and_brainstorm():
    assert set(FORMATS) >= {"debate", "brainstorm"}
    assert DEBATE.id == "debate"
    assert BRAINSTORM.id == "brainstorm"


def test_debate_five_phases_only_critique_sequential():
    assert [p.id for p in DEBATE.phases] == [
        "opinion", "critique", "rebuttal", "revision", "conclusion"]
    assert [p.id for p in DEBATE.phases if p.sequential] == ["critique"]
    assert DEBATE.supports_consensus is True


def test_brainstorm_four_phases_no_consensus():
    assert [p.id for p in BRAINSTORM.phases] == [
        "diverge", "expand", "converge", "action"]
    assert BRAINSTORM.supports_consensus is False


def test_next_phase_and_last_phase():
    assert DEBATE.next_phase("opinion").id == "critique"
    assert DEBATE.next_phase("conclusion") is None
    assert DEBATE.is_last_phase("conclusion") is True
    assert DEBATE.is_last_phase("opinion") is False
    assert DEBATE.first_phase.id == "opinion"


def test_phase_lookup():
    assert DEBATE.phase("critique").label == "2단계 · 상호비판"
    assert DEBATE.phase("nonexistent") is None
    assert DEBATE.phase_index("rebuttal") == 2
    assert DEBATE.phase_index("nonexistent") == -1


def test_get_format_unknown_falls_back_to_default():
    assert get_format("zzz").id == DEFAULT_FORMAT_ID
    assert get_format("brainstorm").id == "brainstorm"


def test_common_rules_have_topic_placeholder():
    # _build_prompt 이 .format(topic=...) 으로 채우므로 플레이스홀더가 있어야 한다.
    assert "{topic}" in DEBATE.common_rules
    assert "{topic}" in BRAINSTORM.common_rules
