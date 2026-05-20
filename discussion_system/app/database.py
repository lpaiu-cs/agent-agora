"""SQLite 영속성 레이어 — 토론 상태 저장 / 조회.

SQLAlchemy(동기) + SQLite 파일 기반. repository 함수(`save_state`/`load_state`/
`list_states`)는 동기이며, 비동기 코드에서는 `asyncio.to_thread` 로 감싸 호출해
이벤트 루프를 막지 않는다.

DB 경로는 환경 변수 `AGORA_DB_URL` 로 바꿀 수 있다 (기본: `sqlite:///./agora.db`).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from .models import Base, DiscussionRow, row_to_state, state_to_row
from .schemas import DiscussionState

logger = logging.getLogger(__name__)

#: DB 접속 URL. 기본은 작업 디렉터리의 `agora.db` SQLite 파일.
DB_URL = os.getenv("AGORA_DB_URL", "sqlite:///./agora.db")

# check_same_thread=False — asyncio.to_thread 의 워커 스레드에서도 접속 허용.
_engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    """`discussions` 테이블을 생성한다. 멱등(이미 있으면 무시)."""
    Base.metadata.create_all(_engine)
    logger.info("SQLite 영속성 레이어 초기화 완료: %s", DB_URL)


def save_state(state: DiscussionState) -> None:
    """DiscussionState 1건을 DB 에 upsert 한다 (discussion_id 기준 merge)."""
    with _Session() as session:
        session.merge(state_to_row(state))
        session.commit()


def load_state(discussion_id: str) -> Optional[DiscussionState]:
    """discussion_id 로 DiscussionState 를 조회한다. 없으면 None."""
    with _Session() as session:
        row = session.get(DiscussionRow, discussion_id)
        return row_to_state(row) if row is not None else None


def list_states() -> list[DiscussionState]:
    """모든 토론 상태를 최근 갱신순(updated_at desc)으로 반환한다."""
    with _Session() as session:
        rows = session.scalars(
            select(DiscussionRow).order_by(DiscussionRow.updated_at.desc())
        ).all()
        return [row_to_state(row) for row in rows]
