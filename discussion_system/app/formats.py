"""토론 형식(DiscussionFormat) 정의 — 단계 구성·프롬프트를 형식별로 캡슐화한다.

형식은 코드가 아니라 **JSON 선언**으로 정의된다 — 내장 형식(debate ·
brainstorm · socratic)은 ``app/format_defs/*.json`` 에, 커스텀 형식은 서버
작업 디렉터리의 ``formats/*.json``(``AGORA_FORMATS_DIR``)에 둔다. 양쪽 모두
같은 로더(``format_from_dict``)를 지나므로 스키마·검증이 동일하다.

  * 내장 형식은 임포트 시 레지스트리(``FORMATS``)에 적재되고, 커스텀 형식은
    서버 기동 시 ``load_custom_formats()`` 가 추가한다.
  * ``DiscussionState`` 는 ``format_id`` 만 저장하고, 런타임에 ``get_format`` 으로
    형식을 조회한다.
  * 단계(phase)는 형식 안에서 문자열 id 로 식별된다 (``"opinion"``, ``"diverge"``…).
    ``"idle"`` / ``"completed"`` 는 단계가 아닌 수명주기 표식으로 예약돼 있다.

이 모듈은 leaf 모듈이다 — app 내부 어떤 모듈도 임포트하지 않아 순환 의존이 없다.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: 토론 시작 전 / 종료 후를 나타내는 예약 단계 식별자 (실제 단계 아님).
PHASE_IDLE = "idle"
PHASE_COMPLETED = "completed"

#: 반복(가변 길이) 단계의 라운드 구분자. 단계 인스턴스 키는 'probe#3' 처럼
#: '단계id#라운드' 로 합성된다 — 단계 id 에는 이 문자를 쓰지 않는다.
ROUND_SEP = "#"

#: 형식을 지정하지 않은 토론의 기본 형식.
DEFAULT_FORMAT_ID = "debate"


@dataclass(frozen=True)
class PhaseSpec:
    """형식 안의 한 단계.

    Attributes:
        id: 형식 내 단계 식별자 (``"opinion"`` 등). ``#`` 는 라운드 구분자로
            예약돼 있으므로 단계 id 에 쓰지 않는다.
        label: 사람이 읽는 표시 라벨 (``"1단계 · 초기주장"``).
        instruction: 이 단계에서 에이전트가 수행할 작업 지침(프롬프트).
        sequential: 순차 포스팅 여부. True 면 후순위 에이전트가 같은 단계의
            선행 의견을 맥락으로 받는다. False(동시)면 서로의 발제를 보지 않는다.
        repeatable: 가변 길이 단계 여부. True 면 이 단계는 '라운드'로 여러 번
            반복되며, 각 라운드는 ``'id#N'`` 인스턴스 키로 구분된다. 반복 종료는
            ``min_rounds``·``max_rounds``·``converge_threshold`` 가 정한다.
        min_rounds: 반복 단계의 최소 라운드 수 — 이만큼은 무조건 반복한다.
        max_rounds: 반복 단계의 최대 라운드 수 — 안전 상한.
        converge_threshold: 합의 근접도가 이 값 이상이면 반복을 조기 종료한다.
    """

    id: str
    label: str
    instruction: str
    sequential: bool = False
    repeatable: bool = False
    min_rounds: int = 1
    max_rounds: int = 1
    converge_threshold: float = 1.0


@dataclass(frozen=True)
class DiscussionFormat:
    """하나의 토론 형식 — 공통 규칙 + 순서가 있는 단계 목록.

    Attributes:
        id: 형식 식별자 (``"debate"``, ``"brainstorm"``).
        name: 표시 이름.
        description: 형식 설명 (UI 안내용).
        common_rules: 모든 단계 공통 시스템 규칙. ``{topic}`` 플레이스홀더를 가질
            수 있다 (``str.format(topic=...)`` 로 채운다).
        phases: 순서가 있는 단계 목록 (1개 이상).
        supports_consensus: 마지막 단계에서 force_consensus 합의안 합성을
            지원하는지. 토론형은 True, 발산 위주 형식은 False.
    """

    id: str
    name: str
    description: str
    common_rules: str
    phases: tuple[PhaseSpec, ...]
    supports_consensus: bool = False

    def phase(self, phase_id: str) -> Optional[PhaseSpec]:
        """phase_id 에 해당하는 PhaseSpec 을 반환한다. 없으면 None.

        ``'probe#3'`` 같은 라운드 인스턴스 키를 줘도 단계 id 로 정규화해 조회한다.
        """
        base = phase_of_key(phase_id)
        for spec in self.phases:
            if spec.id == base:
                return spec
        return None

    def phase_index(self, phase_id: str) -> int:
        """phase_id 의 0-기반 순서를 반환한다. 없으면 -1 (라운드 키 허용)."""
        base = phase_of_key(phase_id)
        for idx, spec in enumerate(self.phases):
            if spec.id == base:
                return idx
        return -1

    def next_phase(self, phase_id: str) -> Optional[PhaseSpec]:
        """다음 단계를 반환한다. 마지막 단계이거나 미지의 id 면 None (라운드 키 허용)."""
        idx = self.phase_index(phase_id)
        if idx < 0 or idx + 1 >= len(self.phases):
            return None
        return self.phases[idx + 1]

    def is_last_phase(self, phase_id: str) -> bool:
        """주어진 단계가 형식의 마지막 단계인지 (라운드 키 허용)."""
        return bool(self.phases) and self.phases[-1].id == phase_of_key(phase_id)

    @property
    def first_phase(self) -> PhaseSpec:
        """형식의 첫 단계 (토론 START 시 진입)."""
        return self.phases[0]


# ===========================================================================
# 단계 인스턴스 키 — 가변 길이(반복) 단계의 라운드 식별
# ===========================================================================
# 단계 기록(phase_records)·요약(phase_summaries)·current_phase 는 '단계 인스턴스
# 키'로 식별된다. 비반복 단계는 순수 id('opinion'), 반복 단계는 'probe#3' 처럼
# 라운드 번호를 붙인다 — 한 단계가 런타임에 여러 라운드로 펼쳐지기 때문이다.
def phase_of_key(key: str) -> str:
    """단계 인스턴스 키에서 단계 id 를 추출한다 ('probe#3' -> 'probe')."""
    return key.split(ROUND_SEP, 1)[0]


def round_of_key(key: str) -> int:
    """단계 인스턴스 키의 라운드 번호 ('probe#3' -> 3, 'opinion' -> 1)."""
    _, _, suffix = key.partition(ROUND_SEP)
    return int(suffix) if suffix.isdigit() else 1


def phase_key(spec: PhaseSpec, round_no: int) -> str:
    """단계 인스턴스 키를 만든다. 반복 단계만 ``'#N'`` 접미사를 붙인다.

    비반복 단계는 항상 순수 id ('opinion') 라 기존 형식·저장 데이터와 호환된다.
    """
    return f"{spec.id}{ROUND_SEP}{round_no}" if spec.repeatable else spec.id


# ===========================================================================
# 형식 로더 — 내장·커스텀 형식을 동일한 JSON 선언으로 정의한다
# ===========================================================================
#: 내장 형식 JSON 정의 디렉터리 (debate.json · brainstorm.json · socratic.json).
_BUILTIN_DEFS_DIR = Path(__file__).resolve().parent / "format_defs"

#: 커스텀 형식 디렉터리 — 서버 작업 디렉터리 기준 상대 경로 (환경변수로 변경).
CUSTOM_FORMATS_DIR_ENV = "AGORA_FORMATS_DIR"
DEFAULT_CUSTOM_FORMATS_DIR = "formats"

logger = logging.getLogger(__name__)


def format_from_dict(data: dict) -> DiscussionFormat:
    """JSON 선언(dict)을 검증해 DiscussionFormat 으로 만든다.

    내장 형식과 커스텀 형식이 같은 경로를 지난다 — 검증 실패는 한국어
    ``ValueError`` 로 즉시 알린다 (호출부가 파일 단위로 흡수).
    """
    for key in ("id", "name", "description", "common_rules", "phases"):
        if not data.get(key):
            raise ValueError(f"필수 필드 누락 또는 빈 값: {key}")
    common_rules = str(data["common_rules"])
    try:
        common_rules.format(topic="검증용 주제")
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            "common_rules 는 str.format 으로 {topic} 을 채울 수 있어야 한다 — "
            f"중괄호 문법 오류: {exc!r}"
        ) from None
    phases = data["phases"]
    if not isinstance(phases, list) or not phases:
        raise ValueError("phases 는 1개 이상의 단계 목록이어야 한다")
    specs: list[PhaseSpec] = []
    seen: set[str] = set()
    for i, p in enumerate(phases):
        if not isinstance(p, dict):
            raise ValueError(f"phases[{i}] 는 객체여야 한다")
        for key in ("id", "label", "instruction"):
            if not p.get(key):
                raise ValueError(f"phases[{i}] 필수 필드 누락 또는 빈 값: {key}")
        pid = str(p["id"])
        if ROUND_SEP in pid:
            raise ValueError(
                f"단계 id '{pid}' — '{ROUND_SEP}' 는 라운드 구분자로 예약됨")
        if pid in (PHASE_IDLE, PHASE_COMPLETED):
            raise ValueError(f"단계 id '{pid}' 는 수명주기 표식으로 예약됨")
        if pid in seen:
            raise ValueError(f"중복 단계 id: {pid}")
        seen.add(pid)
        min_rounds = int(p.get("min_rounds", 1))
        max_rounds = int(p.get("max_rounds", 1))
        repeatable = bool(p.get("repeatable", False))
        if repeatable and not (1 <= min_rounds <= max_rounds):
            raise ValueError(
                f"단계 '{pid}': 1 <= min_rounds <= max_rounds 여야 한다 "
                f"(min={min_rounds}, max={max_rounds})")
        specs.append(PhaseSpec(
            id=pid,
            label=str(p["label"]),
            instruction=str(p["instruction"]),
            sequential=bool(p.get("sequential", False)),
            repeatable=repeatable,
            min_rounds=min_rounds,
            max_rounds=max_rounds,
            converge_threshold=float(p.get("converge_threshold", 1.0)),
        ))
    return DiscussionFormat(
        id=str(data["id"]),
        name=str(data["name"]),
        description=str(data["description"]),
        common_rules=common_rules,
        phases=tuple(specs),
        supports_consensus=bool(data.get("supports_consensus", False)),
    )


def _load_format_file(path: Path) -> DiscussionFormat:
    """JSON 파일 1개를 읽어 형식으로 만든다 (형식 오류는 ValueError)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("형식 정의는 최상위 JSON 객체여야 한다")
    return format_from_dict(data)


def _load_builtin_formats() -> dict[str, DiscussionFormat]:
    """내장 형식(JSON 정의)을 적재한다 — 실패는 앱 결함이므로 즉시 전파."""
    registry: dict[str, DiscussionFormat] = {}
    for path in sorted(_BUILTIN_DEFS_DIR.glob("*.json")):
        fmt = _load_format_file(path)
        registry[fmt.id] = fmt
    if DEFAULT_FORMAT_ID not in registry:
        raise RuntimeError(
            f"내장 형식 '{DEFAULT_FORMAT_ID}' 가 없다 — format_defs/ 손상")
    return registry


def load_custom_formats(directory: Optional[str] = None) -> list[str]:
    """커스텀 형식 디렉터리의 *.json 을 레지스트리에 추가한다.

    서버 기동 시(lifespan) 호출된다. 디렉터리가 없으면 조용히 건너뛰고,
    불량 파일은 경고 로그 후 스킵하며, 내장 형식 id 와 충돌하는 정의는
    거부한다 — 커스텀이 내장 동작을 바꿔치기하지 못하게. 등록된 형식 id
    목록을 반환한다.
    """
    directory = directory or os.getenv(
        CUSTOM_FORMATS_DIR_ENV, DEFAULT_CUSTOM_FORMATS_DIR)
    base = Path(directory)
    if not base.is_dir():
        return []
    builtin_ids = set(_BUILTIN_IDS)
    loaded: list[str] = []
    for path in sorted(base.glob("*.json")):
        try:
            fmt = _load_format_file(path)
        except Exception as exc:  # noqa: BLE001 - 불량 파일은 형식 단위로 스킵
            logger.warning("커스텀 형식 적재 실패(%s): %s", path.name, exc)
            continue
        if fmt.id in builtin_ids:
            logger.warning(
                "커스텀 형식 '%s'(%s) — 내장 형식 id 와 충돌, 거부", fmt.id, path.name)
            continue
        FORMATS[fmt.id] = fmt
        loaded.append(fmt.id)
        logger.info("커스텀 형식 등록: %s (%s)", fmt.id, path.name)
    return loaded


#: 형식 레지스트리 — id -> DiscussionFormat. 내장은 임포트 시, 커스텀은
#: 서버 기동 시(load_custom_formats) 추가된다.
FORMATS: dict[str, DiscussionFormat] = _load_builtin_formats()
_BUILTIN_IDS: tuple[str, ...] = tuple(FORMATS)

#: 내장 형식 별칭 — 기존 코드·테스트 호환.
DEBATE = FORMATS["debate"]
BRAINSTORM = FORMATS["brainstorm"]
SOCRATIC = FORMATS["socratic"]


def get_format(format_id: str) -> DiscussionFormat:
    """format_id 에 해당하는 형식을 반환한다. 미지의 id 면 기본 형식(debate)."""
    return FORMATS.get(format_id, FORMATS[DEFAULT_FORMAT_ID])


# ===========================================================================
# 단계 진행 결정자 — 정적·가변 길이 형식을 공통으로 다룬다
# ===========================================================================
def plan_next(
    fmt: DiscussionFormat,
    current_key: str,
    latest_convergence: float = 0.0,
    decision: Optional[str] = None,
) -> Optional[str]:
    """현재 단계 인스턴스에서 다음에 실행할 인스턴스 키를 결정한다.

    정적(전 단계 비반복) 형식에서는 ``next_phase`` 인덱스 진행과 동치다. 반복
    단계에서는 min/max 라운드와 합의 근접도(또는 사회자 ``decision``)로 루프를
    제어한다.

    Args:
        fmt: 토론 형식.
        current_key: 현재 단계 인스턴스 키. 시작 전이면 ``PHASE_IDLE``.
        latest_convergence: 방금 끝난 단계의 합의 근접도 (반복 종료 판정용).
        decision: 사회자의 진행 결정("continue"/"next"/"conclude"). 없으면 None.

    Returns:
        다음에 실행할 단계 인스턴스 키. 토론을 끝내야 하면 None.
    """
    if current_key == PHASE_IDLE:
        return phase_key(fmt.first_phase, 1)
    spec = fmt.phase(current_key)
    if spec is None:
        return None
    if spec.repeatable:
        rnd = round_of_key(current_key)
        if _should_repeat(spec, rnd, latest_convergence, decision):
            return phase_key(spec, rnd + 1)
        if decision == "conclude":
            return None
    nxt = fmt.next_phase(current_key)
    return phase_key(nxt, 1) if nxt is not None else None


def _should_repeat(
    spec: PhaseSpec,
    current_round: int,
    latest_convergence: float,
    decision: Optional[str],
) -> bool:
    """반복 단계를 한 라운드 더 진행할지 결정한다.

    우선순위: 최소 라운드 바닥 → 최대 라운드 천장 → 사회자 결정 → 합의 근접도.
    """
    if current_round < spec.min_rounds:
        return True
    if current_round >= spec.max_rounds:
        return False
    if decision is not None:
        return decision == "continue"
    return latest_convergence < spec.converge_threshold
