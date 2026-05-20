"""비동기 영속성 레이어 — 토론 상태 저장 / 조회 + 낙관적 락.

phase-7: 동기 `Session` + `asyncio.to_thread` 구조를 SQLAlchemy 의 `AsyncSession`
+ `create_async_engine` 로 전면 전환했다. 모든 DB 진입점은 `async def` 이며 내부
SQL 은 `await session.execute(...)` 로 실행된다 — 스레드 풀 오프로딩 병목 제거.

DB 드라이버는 환경 변수 `DATABASE_URL` 로 주입한다 (없으면 하위 호환용
`AGORA_DB_URL`, 그것도 없으면 기본 `sqlite+aiosqlite:///./agora.db`). 같은 코드가
``sqlite+aiosqlite`` 와 ``postgresql+asyncpg`` 를 그대로 수용한다.

동시성 제어: `update_state` 는 `version` 컬럼 낙관적 락을 적용한다 —
``UPDATE ... WHERE discussion_id=? AND version=?`` 가 0행을 갱신하면
`StaleStateError`. SQLite·PostgreSQL 양쪽에서 이 검사-후-갱신은 트랜잭션
안에서 원자적이다.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .models import Base, DiscussionRow, row_to_state, state_to_columns, state_to_row
from .schemas import DiscussionState

logger = logging.getLogger(__name__)


def _resolve_db_url() -> str:
    """DB URL 을 환경 변수에서 결정한다 (DATABASE_URL > AGORA_DB_URL > 기본 SQLite)."""
    url = os.getenv("DATABASE_URL") or os.getenv("AGORA_DB_URL")
    return url or "sqlite+aiosqlite:///./agora.db"


#: 비동기 DB URL — 드라이버까지 포함 (sqlite+aiosqlite / postgresql+asyncpg).
DATABASE_URL = _resolve_db_url()

#: 비동기 엔진 + 세션 팩토리. 환경 변수만 바꾸면 SQLite ↔ PostgreSQL 이 전환된다.
_engine: AsyncEngine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
_Session = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)


class StaleStateError(RuntimeError):
    """낙관적 락 충돌 — 로드 이후 다른 트랜잭션이 먼저 상태를 갱신했다."""


async def init_db() -> None:
    """`discussions` 테이블을 생성한다. 멱등(이미 있으면 무시)."""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("비동기 영속성 레이어 초기화 완료: %s", DATABASE_URL)


async def dispose_engine() -> None:
    """엔진 커넥션 풀을 정리한다 (서버 종료 시 lifespan 이 호출)."""
    await _engine.dispose()


async def insert_state(state: DiscussionState) -> None:
    """새 토론 상태를 INSERT 한다 (version 은 state 의 값, 보통 0)."""
    async with _Session() as session:
        session.add(state_to_row(state))
        await session.commit()


async def update_state(state: DiscussionState) -> None:
    """낙관적 락으로 토론 상태를 UPDATE 한다.

    ``WHERE version = <로드 시점 버전>`` 이 0행을 갱신하면 `StaleStateError`.
    성공 시 ``state.version`` 을 +1 로 올린다 (DB 값과 동기화).
    """
    expected = state.version
    columns = state_to_columns(state)
    columns["version"] = expected + 1
    async with _Session() as session:
        result = await session.execute(
            update(DiscussionRow)
            .where(DiscussionRow.discussion_id == state.discussion_id)
            .where(DiscussionRow.version == expected)
            .values(**columns)
        )
        await session.commit()
    if result.rowcount == 0:
        raise StaleStateError(
            f"낙관적 락 충돌: {state.discussion_id} "
            f"(로드 버전 {expected} — 다른 트랜잭션이 먼저 갱신함)"
        )
    state.version = expected + 1


async def load_state(discussion_id: str) -> Optional[DiscussionState]:
    """discussion_id 로 DiscussionState 를 조회한다. 없으면 None."""
    async with _Session() as session:
        row = await session.get(DiscussionRow, discussion_id)
        return row_to_state(row) if row is not None else None


async def list_states() -> list[DiscussionState]:
    """모든 토론 상태를 최근 갱신순(updated_at desc)으로 반환한다."""
    async with _Session() as session:
        result = await session.scalars(
            select(DiscussionRow).order_by(DiscussionRow.updated_at.desc())
        )
        return [row_to_state(row) for row in result.all()]


async def list_states_by_status(statuses: tuple[str, ...]) -> list[DiscussionState]:
    """주어진 status 목록에 해당하는 토론 상태를 반환한다 (크래시 복구용)."""
    async with _Session() as session:
        result = await session.scalars(
            select(DiscussionRow).where(DiscussionRow.status.in_(statuses))
        )
        return [row_to_state(row) for row in result.all()]
