"""SQLAlchemy ORM 모델 + DiscussionState <-> ORM 행 변환.

`DiscussionState` 의 스칼라 필드는 전용 컬럼으로, 복잡한 컬렉션(에이전트 목록,
단계별 발언 기록, 요약, 개입)은 JSON 컬럼으로 직렬화 매핑한다 — 한 토론 =
`discussions` 테이블 1행. `version` 컬럼은 낙관적 락(optimistic lock)에 쓰인다.
"""

from __future__ import annotations

from sqlalchemy import JSON, Boolean, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base

from .schemas import DiscussionState

Base = declarative_base()

#: DiscussionState 의 컬렉션 필드 — JSON 컬럼으로 직렬화 저장한다.
#: phase_records 는 dict(단계 id -> 발언 목록), 나머지는 list.
_JSON_FIELDS = (
    "agents",
    "phase_records",
    "intercept_agents",
    "review",
    "facilitator",
    "facilitator_notes",
    "phase_summaries",
    "user_interventions",
)

#: None 이 유효값인 JSON 필드 — 비었을 때 [] / {} 로 치환하지 않는다.
_NULLABLE_JSON_FIELDS = ("review", "facilitator")

#: DiscussionState 의 스칼라 필드 — 전용 컬럼으로 매핑한다.
_SCALAR_FIELDS = (
    "discussion_id",
    "topic",
    "format_id",
    "status",
    "current_phase",
    "force_consensus",
    "reference_materials",
    "token_budget",
    "error",
    "final_joint_agreement",
    "created_at",
    "updated_at",
    "version",
)


class DiscussionRow(Base):
    """`discussions` 테이블 — DiscussionState 1건을 1행으로 매핑한다."""

    __tablename__ = "discussions"

    # --- 스칼라 컬럼 (조회 / 필터용) ---
    discussion_id = Column(String, primary_key=True)
    topic = Column(String, nullable=False)
    format_id = Column(String, nullable=False, default="debate")
    status = Column(String, nullable=False, index=True)
    current_phase = Column(String, nullable=False)
    force_consensus = Column(Boolean, nullable=False, default=False)
    reference_materials = Column(Text, nullable=True)   # 선택 참고 자료
    token_budget = Column(Integer, nullable=True)       # 선택 토큰 예산
    error = Column(Text, nullable=True)
    final_joint_agreement = Column(Text, nullable=True)
    created_at = Column(String, nullable=False)          # ISO-8601 문자열
    updated_at = Column(String, nullable=False, index=True)
    version = Column(Integer, nullable=False, default=0)  # 낙관적 락 버전

    # --- JSON 컬럼 (복잡한 딕셔너리/리스트 필드 직렬화) ---
    agents = Column(JSON, nullable=False, default=list)
    phase_records = Column(JSON, nullable=False, default=dict)
    intercept_agents = Column(JSON, nullable=False, default=list)
    review = Column(JSON, nullable=True)                  # 검토 세션 (없으면 NULL)
    facilitator = Column(JSON, nullable=True)             # 사회자 (없으면 NULL)
    facilitator_notes = Column(JSON, nullable=False, default=list)
    phase_summaries = Column(JSON, nullable=False, default=list)
    user_interventions = Column(JSON, nullable=False, default=list)


def state_to_columns(state: DiscussionState) -> dict:
    """DiscussionState -> 컬럼명->값 dict (INSERT/UPDATE 양쪽에서 재사용)."""
    dumped = state.model_dump(mode="json")
    return {field: dumped[field] for field in (*_SCALAR_FIELDS, *_JSON_FIELDS)}


def state_to_row(state: DiscussionState) -> DiscussionRow:
    """DiscussionState -> DiscussionRow ORM 객체 (INSERT 용)."""
    return DiscussionRow(**state_to_columns(state))


def row_to_state(row: DiscussionRow) -> DiscussionState:
    """DiscussionRow -> DiscussionState (Pydantic 재검증)."""
    data: dict = {field: getattr(row, field) for field in _SCALAR_FIELDS}
    for field in _JSON_FIELDS:
        value = getattr(row, field)
        if value is None and field not in _NULLABLE_JSON_FIELDS:
            value = {} if field == "phase_records" else []
        data[field] = value
    return DiscussionState.model_validate(data)
