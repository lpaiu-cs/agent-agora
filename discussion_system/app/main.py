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

    기동 시: SQLite 영속성 레이어(테이블)를 초기화하고, 크래시 복구
    coordinator(`Orchestrator.recover`)를 가동해 중단된 토론을 재기동/유지한다.
    종료 시: 글로벌 LLM 클라이언트 풀을 일괄 폐쇄한다.
    """
    await database.init_db()
    recovered = await discussion.orchestrator.recover()
    logger.info(
        "Agent Agora %s 기동 — DB 초기화 + 크래시 복구 "
        "(RUNNING 재기동 %d / PENDING 유지 %d)",
        __version__,
        recovered["running_recovered"],
        recovered["pending_preserved"],
    )
    yield
    await discussion.pool.aclose()
    await database.dispose_engine()
    logger.info("LLM 클라이언트 풀 · DB 엔진 폐쇄 완료 — 서버 종료")

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
