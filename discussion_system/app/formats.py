"""토론 형식(DiscussionFormat) 정의 — 단계 구성·프롬프트를 형식별로 캡슐화한다.

기존에는 5단계 토론 프로토콜이 코드 전반에 하드코딩돼 있었다. 이 모듈은 그
구조를 데이터로 끌어내, 형식마다 단계 개수·순서·지침·순차성을 자유롭게 정의할
수 있게 한다.

  * 형식은 코드로 정의된 레지스트리(``FORMATS``)에 등록된다.
  * ``DiscussionState`` 는 ``format_id`` 만 저장하고, 런타임에 ``get_format`` 으로
    형식을 조회한다.
  * 단계(phase)는 형식 안에서 문자열 id 로 식별된다 (``"opinion"``, ``"diverge"``…).
    ``"idle"`` / ``"completed"`` 는 단계가 아닌 수명주기 표식으로 예약돼 있다.

이 모듈은 leaf 모듈이다 — app 내부 어떤 모듈도 임포트하지 않아 순환 의존이 없다.
"""
from __future__ import annotations

from dataclasses import dataclass
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
# 공통 규칙
# ===========================================================================
_DEBATE_RULES = (
    "너는 '{topic}' 주제의 다자(多者) 구조화 토론에 참여하는 토론자다.\n"
    "토론은 5단계(① 초기주장 → ② 상호비판 → ③ 반론·방어 → ④ 입장수정 "
    "→ ⑤ 최종결론)로 진행된다.\n"
    "[공통 규칙]\n"
    "- 한국어로, 핵심 위주로 간결하게(6~8문장 이내) 작성한다.\n"
    "- 다른 참가자를 이름으로 직접 지칭하며 구체적으로 논평한다.\n"
    "- '참가자 H' 로 표기된 발언은 토론을 지켜보는 인간 진행자의 개입이다. "
    "그 지시는 최우선으로 반영한다.\n"
    "- '[시스템 경고: ...]' 로 표기된 발언은 해당 에이전트의 응답 생성 실패를 "
    "뜻한다. 그 내용에 의존하거나 인용하지 말고 토론을 정상 진행한다."
)

_BRAINSTORM_RULES = (
    "너는 '{topic}' 주제의 다자(多者) 브레인스토밍 세션에 참여하는 참여자다.\n"
    "세션은 4단계(① 아이디어 발산 → ② 상호 확장 → ③ 수렴·선별 → ④ 실행안)로 "
    "진행된다.\n"
    "[공통 규칙]\n"
    "- 한국어로, 핵심 위주로 간결하게(6~8문장 이내) 작성한다.\n"
    "- 비판보다 발전에 무게를 둔다 — 남의 아이디어를 깎아내리기보다 키운다.\n"
    "- 다른 참가자를 이름으로 직접 지칭하며 구체적으로 반응한다.\n"
    "- '참가자 H' 로 표기된 발언은 세션을 지켜보는 인간 진행자의 개입이다. "
    "그 지시는 최우선으로 반영한다.\n"
    "- '[시스템 경고: ...]' 로 표기된 발언은 해당 에이전트의 응답 생성 실패를 "
    "뜻한다. 그 내용에 의존하거나 인용하지 말고 세션을 정상 진행한다."
)

_SOCRATIC_RULES = (
    "너는 '{topic}' 주제를 놓고 벌이는 소크라테스식 문답 토론의 참여자다.\n"
    "토론은 입장 제시 → 문답 라운드(반복) → 종합 순으로 진행되며, 문답 라운드는 "
    "논의가 충분히 수렴할 때까지 여러 번 반복된다.\n"
    "[공통 규칙]\n"
    "- 한국어로, 핵심 위주로 간결하게(6~8문장 이내) 작성한다.\n"
    "- 단정하기보다 질문한다 — 상대 주장의 숨은 전제를 캐묻고, 받은 질문에는 "
    "회피 없이 성실히 답한다.\n"
    "- 다른 참가자를 이름으로 직접 지칭하며 구체적으로 묻고 답한다.\n"
    "- '참가자 H' 로 표기된 발언은 토론을 지켜보는 인간 진행자의 개입이다. "
    "그 지시는 최우선으로 반영한다.\n"
    "- '[시스템 경고: ...]' 로 표기된 발언은 해당 에이전트의 응답 생성 실패를 "
    "뜻한다. 그 내용에 의존하거나 인용하지 말고 토론을 정상 진행한다."
)


# ===========================================================================
# 형식 정의
# ===========================================================================
#: 구조화 토론 — 기존 5단계 프로토콜. 단계 지침은 종전과 동일(동작 불변).
DEBATE = DiscussionFormat(
    id="debate",
    name="구조화 토론",
    description="찬반·쟁점 중심의 5단계 토론 — 초기주장부터 최종결론까지.",
    common_rules=_DEBATE_RULES,
    supports_consensus=True,
    phases=(
        PhaseSpec(
            id="opinion",
            label="1단계 · 초기주장",
            instruction=(
                "[1단계 · 초기주장] 주제에 대한 너의 입장과 이를 뒷받침하는 핵심 "
                "논거 2~3가지를 명확히 제시하라."
            ),
            sequential=False,
        ),
        PhaseSpec(
            id="critique",
            label="2단계 · 상호비판",
            instruction=(
                "[2단계 · 상호비판] 다른 참가자들의 주장에서 가장 약한 지점을 찾아 "
                "근거를 들어 비판적으로 검토하라."
            ),
            sequential=True,
        ),
        PhaseSpec(
            id="rebuttal",
            label="3단계 · 반론·방어",
            instruction=(
                "[3단계 · 반론·방어] 너의 주장에 제기된 비판을 직접 거론하며 "
                "반론하고, 필요하면 논거를 보강해 입장을 방어하라."
            ),
            sequential=False,
        ),
        PhaseSpec(
            id="revision",
            label="4단계 · 입장수정",
            instruction=(
                "[4단계 · 입장수정] 지금까지의 토론을 반영하여 너의 입장을 "
                "갱신하라. 바뀐 부분과 그대로 유지하는 부분을 구분해 밝혀라."
            ),
            sequential=False,
        ),
        PhaseSpec(
            id="conclusion",
            label="5단계 · 최종결론",
            instruction=(
                "[5단계 · 최종결론] 다른 참가자들의 4단계 입장을 검토하고, 너의 "
                "최종 입장과 끝내 좁혀지지 않은 핵심 차이점을 '이견 일람표'(쟁점 | "
                "나의 입장 | 상대 입장 형식의 표)로 정리하라."
            ),
            sequential=False,
        ),
    ),
)

#: 브레인스토밍 — 발산에서 실행안까지 4단계. 비판보다 발전 지향.
BRAINSTORM = DiscussionFormat(
    id="brainstorm",
    name="브레인스토밍",
    description="아이디어 발산부터 실행안까지 — 발산·확장·수렴·실행 4단계.",
    common_rules=_BRAINSTORM_RULES,
    supports_consensus=False,
    phases=(
        PhaseSpec(
            id="diverge",
            label="1단계 · 아이디어 발산",
            instruction=(
                "[1단계 · 아이디어 발산] 주제에 대해 제약을 두지 말고 새로운 "
                "아이디어를 2~3개 제시하라. 실현 가능성보다 발상의 폭과 참신함을 "
                "우선한다."
            ),
            sequential=False,
        ),
        PhaseSpec(
            id="expand",
            label="2단계 · 상호 확장",
            instruction=(
                "[2단계 · 상호 확장] 다른 참가자들의 아이디어 중 가장 유망한 것을 "
                "골라, '예, 그리고'의 태도로 살을 붙여 더 구체적이고 강한 형태로 "
                "발전시켜라."
            ),
            sequential=True,
        ),
        PhaseSpec(
            id="converge",
            label="3단계 · 수렴·선별",
            instruction=(
                "[3단계 · 수렴·선별] 지금까지 나온 아이디어들을 평가해, 가장 가치 "
                "있다고 판단하는 1~2개를 선별하고 그 선정 근거를 밝혀라."
            ),
            sequential=False,
        ),
        PhaseSpec(
            id="action",
            label="4단계 · 실행안",
            instruction=(
                "[4단계 · 실행안] 선별된 아이디어를 실제로 추진하기 위한 구체적인 "
                "다음 단계와 예상되는 제약·위험을 제시하라."
            ),
            sequential=False,
        ),
    ),
)


#: 소크라테스식 문답 — 가변 길이 형식. 문답 라운드(probe)가 합의 근접도에 따라
#: 2~6회 반복된다 — 총 단계 수가 런타임에 결정되는 첫 형식이다.
SOCRATIC = DiscussionFormat(
    id="socratic",
    name="소크라테스식 문답",
    description=(
        "입장 제시 후 문답 라운드를 합의에 가까워질 때까지 반복하는 가변 길이 토론."
    ),
    common_rules=_SOCRATIC_RULES,
    supports_consensus=True,
    phases=(
        PhaseSpec(
            id="position",
            label="입장 제시",
            instruction=(
                "[입장 제시] 주제에 대한 너의 입장과 그 입장이 기대고 있는 핵심 "
                "전제 1~2가지를 명확히 밝혀라."
            ),
            sequential=False,
        ),
        PhaseSpec(
            id="probe",
            label="문답 라운드",
            instruction=(
                "[문답 라운드] 다른 참가자의 입장에서 가장 검증이 필요한 전제를 "
                "하나 골라 날카롭게 질문하라. 동시에 네가 받은 질문에는 회피 없이 "
                "답하고, 설득력 있는 지적은 너의 입장에 반영하라."
            ),
            sequential=True,
            repeatable=True,
            min_rounds=2,
            max_rounds=6,
            converge_threshold=0.8,
        ),
        PhaseSpec(
            id="synthesis",
            label="종합",
            instruction=(
                "[종합] 문답을 거치며 도달한 너의 최종 입장을 정리하고, 합의된 "
                "지점과 끝내 남은 이견을 구분해 밝혀라."
            ),
            sequential=False,
        ),
    ),
)


#: 형식 레지스트리 — id -> DiscussionFormat.
FORMATS: dict[str, DiscussionFormat] = {
    DEBATE.id: DEBATE,
    BRAINSTORM.id: BRAINSTORM,
    SOCRATIC.id: SOCRATIC,
}


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
