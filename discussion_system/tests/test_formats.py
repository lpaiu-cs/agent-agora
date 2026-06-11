"""formats.py — 토론 형식 정의·조회·진행 결정·JSON 로더."""
import json

import pytest

from app.formats import (
    BRAINSTORM,
    DEBATE,
    DEFAULT_FORMAT_ID,
    FORMATS,
    PHASE_IDLE,
    SOCRATIC,
    format_from_dict,
    get_format,
    load_custom_formats,
    phase_key,
    phase_of_key,
    plan_next,
    round_of_key,
)


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
    assert "{topic}" in SOCRATIC.common_rules


# ---------------------------------------------------------------------------
# 가변 길이 형식 — 단계 인스턴스 키 / 진행 결정자(plan_next)
# ---------------------------------------------------------------------------
def test_socratic_registered_and_variable_length():
    assert FORMATS["socratic"].id == "socratic"
    assert [p.id for p in SOCRATIC.phases] == ["position", "probe", "synthesis"]
    probe = SOCRATIC.phase("probe")
    assert probe.repeatable is True
    assert (probe.min_rounds, probe.max_rounds) == (2, 6)
    # 나머지 단계는 비반복 — 기존 형식과 동일한 정적 단계.
    assert SOCRATIC.phase("position").repeatable is False
    assert all(not p.repeatable for p in DEBATE.phases)


def test_phase_key_helpers_round_trip():
    probe = SOCRATIC.phase("probe")          # 반복 단계
    position = SOCRATIC.phase("position")    # 비반복 단계
    assert phase_key(probe, 3) == "probe#3"
    assert phase_key(position, 1) == "position"   # 비반복은 순수 id (호환)
    assert phase_of_key("probe#3") == "probe"
    assert phase_of_key("position") == "position"
    assert round_of_key("probe#3") == 3
    assert round_of_key("position") == 1


def test_phase_lookup_accepts_round_key():
    # 'probe#4' 같은 라운드 인스턴스 키도 단계 id 로 정규화돼 조회된다.
    assert SOCRATIC.phase("probe#4").id == "probe"
    assert SOCRATIC.phase_index("probe#4") == 1
    assert SOCRATIC.is_last_phase("synthesis") is True
    assert SOCRATIC.is_last_phase("probe#4") is False


def test_plan_next_static_format_is_index_progression():
    # 정적 형식(전 단계 비반복)에서 plan_next 는 next_phase 인덱스 진행과 동치.
    assert plan_next(DEBATE, PHASE_IDLE) == "opinion"
    assert plan_next(DEBATE, "opinion") == "critique"
    assert plan_next(DEBATE, "conclusion") is None


def test_plan_next_repeats_until_min_then_threshold():
    # min_rounds=2 — 1라운드는 합의 근접도와 무관하게 반복.
    assert plan_next(SOCRATIC, "probe#1", latest_convergence=1.0) == "probe#2"
    # 2라운드 이후 근접도 ≥ 0.8(converge_threshold) 이면 다음 단계로.
    assert plan_next(SOCRATIC, "probe#2", latest_convergence=0.85) == "synthesis"
    # 근접도 < 0.8 이면 라운드 계속.
    assert plan_next(SOCRATIC, "probe#2", latest_convergence=0.5) == "probe#3"


def test_plan_next_caps_at_max_rounds():
    # max_rounds=6 — 근접도가 낮아도 6라운드에서 멈추고 다음 단계로.
    assert plan_next(SOCRATIC, "probe#6", latest_convergence=0.0) == "synthesis"


# ---------------------------------------------------------------------------
# JSON 형식 로더 — 내장·커스텀 형식이 같은 선언·검증 경로를 지난다
# ---------------------------------------------------------------------------
def _valid_def(**over):
    base = {
        "id": "roundtable", "name": "원탁 토론", "description": "자유 토론.",
        "common_rules": "너는 '{topic}' 원탁 토론의 참여자다.",
        "supports_consensus": True,
        "phases": [
            {"id": "talk", "label": "자유 발언", "instruction": "[자유 발언] 의견을 내라.",
             "sequential": True, "repeatable": True,
             "min_rounds": 1, "max_rounds": 4, "converge_threshold": 0.7},
            {"id": "wrap", "label": "정리", "instruction": "[정리] 결론을 내라."},
        ],
    }
    base.update(over)
    return base


def test_format_from_dict_builds_full_format():
    fmt = format_from_dict(_valid_def())
    assert fmt.id == "roundtable"
    assert fmt.supports_consensus is True
    talk = fmt.phase("talk")
    assert (talk.sequential, talk.repeatable) == (True, True)
    assert (talk.min_rounds, talk.max_rounds, talk.converge_threshold) == (1, 4, 0.7)
    assert fmt.phase("wrap").repeatable is False   # 기본값
    # plan_next 등 런타임 결정자와 곧장 호환된다.
    assert plan_next(fmt, PHASE_IDLE) == "talk#1"


@pytest.mark.parametrize("broken,msg", [
    (_valid_def(id=""), "필수 필드"),
    ({k: v for k, v in _valid_def().items() if k != "phases"}, "필수 필드"),
    (_valid_def(phases=[]), "필수 필드"),                     # 빈 phases 도 거부
    (_valid_def(phases=[{"id": "a", "label": "L"}]), "필수 필드"),
    (_valid_def(phases=[
        {"id": "a#1", "label": "L", "instruction": "I"}]), "예약"),
    (_valid_def(phases=[
        {"id": "idle", "label": "L", "instruction": "I"}]), "예약"),
    (_valid_def(phases=[
        {"id": "a", "label": "L", "instruction": "I"},
        {"id": "a", "label": "L2", "instruction": "I2"}]), "중복"),
    (_valid_def(phases=[
        {"id": "a", "label": "L", "instruction": "I",
         "repeatable": True, "min_rounds": 5, "max_rounds": 2}]), "min_rounds"),
    (_valid_def(common_rules="중괄호 {깨진"), "중괄호"),
])
def test_format_from_dict_rejects_invalid(broken, msg):
    with pytest.raises(ValueError, match=msg):
        format_from_dict(broken)


def test_load_custom_formats_dir(tmp_path):
    (tmp_path / "roundtable.json").write_text(
        json.dumps(_valid_def(), ensure_ascii=False), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{잘못된 json", encoding="utf-8")
    (tmp_path / "clash.json").write_text(
        json.dumps(_valid_def(id="debate"), ensure_ascii=False), encoding="utf-8")
    try:
        loaded = load_custom_formats(str(tmp_path))
        # 정상 1건만 등록 — 불량 파일 스킵, 내장 id(debate) 충돌 거부.
        assert loaded == ["roundtable"]
        assert get_format("roundtable").name == "원탁 토론"
        assert get_format("debate") is DEBATE      # 내장이 바꿔치기되지 않음
    finally:
        FORMATS.pop("roundtable", None)            # 전역 레지스트리 원복


def test_load_custom_formats_missing_dir_is_noop(tmp_path):
    assert load_custom_formats(str(tmp_path / "없는폴더")) == []
