"""FastAPI 애플리케이션 진입점.

`uvicorn app.main:app --reload` 로 기동한다.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from . import __version__, database
from .routers import discussion

#: index.html 단일 파일 UI 템플릿 디렉터리.
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """앱 수명주기 관리.

    글로벌 LLM 클라이언트 풀(`DiscussionRegistry.pool`)은 모든 토론 세션이
    공유하며, 서버 종료 시 이 lifespan 의 종료 구간에서 일괄 폐쇄한다.
    기동 시 SQLite 영속성 레이어(테이블)를 초기화한다.
    """
    database.init_db()
    logger.info(
        "Agent Agora %s 기동 — SQLite 영속성 · 글로벌 LLM 클라이언트 풀 준비됨",
        __version__,
    )
    yield
    await discussion.registry.pool.aclose()
    logger.info("글로벌 LLM 클라이언트 풀 폐쇄 완료 — 서버 종료")

app = FastAPI(
    title="Agent Agora — 다중 에이전트 토론 시스템",
    description=(
        "5단계 턴 파이프라인 기반 LLM 에이전트 토론 오케스트레이터. "
        "멀티 공급자 연동 · 토큰 스트리밍 · 콘텍스트 압축(LTM) · 웹 UI."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.include_router(discussion.router)


@app.get("/", response_class=HTMLResponse, tags=["ui"], include_in_schema=False)
async def index() -> HTMLResponse:
    """단일 파일 웹 UI(index.html)를 반환한다."""
    return HTMLResponse((_TEMPLATE_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """헬스 체크."""
    return {"status": "ok", "version": __version__}
