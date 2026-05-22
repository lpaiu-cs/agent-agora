"""테스트 공용 픽스처 — 격리된 임시 DB, 에이전트/상태 팩토리.

app 모듈을 임포트하기 '전에' 환경 변수를 안전한 기본값으로 고정한다:
database 모듈이 임포트 시점에 ./agora.db 엔진을 만들지 않게 하고, LLM 풀이
클라이언트 생성 단계의 키 검사를 통과하게 한다(실제 호출은 테스트가 가로챈다).
"""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("OPENAI_API_KEY", "test-dummy-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-dummy-key")

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app import database  # noqa: E402
from app.schemas import AgentConfig, DiscussionState  # noqa: E402


@pytest_asyncio.fixture
async def fresh_db(tmp_path, monkeypatch):
    """테스트마다 격리된 임시 SQLite 파일로 database 모듈의 엔진을 교체한다.

    database 의 함수들은 모듈 전역 ``_engine`` / ``_Session`` 을 호출 시점에
    참조하므로, monkeypatch 로 교체하면 모든 SQL 이 임시 DB 로 향한다.
    """
    url = f"sqlite+aiosqlite:///{tmp_path}/test.db"
    engine = create_async_engine(url)
    session_factory = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )
    monkeypatch.setattr(database, "_engine", engine)
    monkeypatch.setattr(database, "_Session", session_factory)
    monkeypatch.setattr(database, "DATABASE_URL", url)
    await database.init_db()
    yield
    await engine.dispose()


@pytest.fixture
def make_agent():
    """AgentConfig 팩토리 — 키워드로 필요한 필드만 덮어쓴다."""
    def _make(agent_id="a1", name="알파", model="gpt-4o-mini",
              provider=None, persona_prompt=None, **kw):
        return AgentConfig(
            agent_id=agent_id, name=name, model=model, provider=provider,
            persona_prompt=persona_prompt or f"{name} 페르소나", **kw,
        )
    return _make


@pytest.fixture
def make_state(make_agent):
    """DiscussionState 팩토리 — 인자 없으면 기본 2인 토론."""
    def _make(discussion_id="d1", topic="테스트 주제", agents=None, **kw):
        if agents is None:
            agents = [make_agent("a1", "알파"), make_agent("a2", "베타")]
        return DiscussionState(
            discussion_id=discussion_id, topic=topic, agents=agents, **kw,
        )
    return _make
