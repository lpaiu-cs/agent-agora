# Agent Agora — 다중 에이전트 토론 시스템

여러 LLM 에이전트가 하나의 주제를 두고 **5단계 프로토콜**로 토론하도록
오케스트레이션하는 FastAPI 기반 백엔드.

## 5단계 토론 프로토콜

1·2단계는 **순차 포스팅**(후순위 에이전트가 선행 의견을 맥락으로 받음),
3·4단계는 **동시 호출**(`asyncio.gather`)로 진행되며, 단계가 끝나면 파이프라인이
**락(`asyncio.Event`)** 되어 유저 개입을 기다린다.

| 단계 | 이름 | 설명 |
|------|------|------|
| 1 | 초기 주장 (opinion)    | 각 에이전트가 주제에 대한 입장을 제시 |
| 2 | 상호 비판 (critique)   | 다른 에이전트의 주장을 비판 |
| 3 | 반론 및 방어 (rebuttal)| 받은 비판에 반론 / 자기 입장 방어 |
| 4 | 입장 수정 (revision)   | 토론을 반영해 입장을 갱신 |
| 5 | 최종 입장 / 합의 (conclusion) | 최종 결론 (`force_consensus` 시 합의 강제) |

## 디렉터리 구조

```
discussion_system/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI 진입점 (lifespan: DB·풀 초기화/정리)
│   ├── schemas.py         # Pydantic 모델 (State, AgentConfig 등)
│   ├── manager.py         # 이벤트 구동형 무상태 오케스트레이터 + LLM 연동
│   ├── database.py        # SQLite 영속성 레이어 (저장/조회 + 낙관적 락)
│   ├── models.py          # SQLAlchemy ORM 모델 + 상태 <-> 행 변환
│   ├── routers/
│   │   └── discussion.py  # WebSocket 및 API 엔드포인트
│   └── templates/
│       └── index.html     # 단일 파일 웹 UI (HTML + Vanilla JS)
├── requirements.txt
└── README.md
```

## 요구 사항

- Python 3.10 이상 (`asyncio.Event`/`Lock` 의 루프 비종속 생성)

## 프로덕션 실행 가이드

### 1) 설치

```bash
cd discussion_system
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) 환경 변수 (LLM API Key) 설정

사용할 공급자의 API Key 를 환경 변수로 등록한다. 셋 다 필요하지는 않고,
실제 등록한 에이전트가 쓰는 공급자의 키만 있으면 된다.

```bash
export OPENAI_API_KEY="sk-..."                 # OpenAI(gpt-*) 에이전트 사용 시
export ANTHROPIC_API_KEY="sk-ant-..."          # Anthropic(claude-*) 에이전트 사용 시
export OLLAMA_HOST="http://localhost:11434"    # (선택) Ollama 호스트, 기본값 동일
```

| 변수 | 용도 |
|------|------|
| `OPENAI_API_KEY` | OpenAI 공급자 사용 시 필수 |
| `ANTHROPIC_API_KEY` | Anthropic 공급자 사용 시 필수 |
| `OLLAMA_HOST` | (선택) Ollama 호스트 주소. 기본 `http://localhost:11434` |
| `DATABASE_URL` | (선택) 비동기 DB URL. 기본 `sqlite+aiosqlite:///./agora.db`. `postgresql+asyncpg://…` 도 그대로 수용 (`AGORA_DB_URL` 은 하위 호환 폴백) |

> Key 가 없거나 호출이 실패해도 서버는 중단되지 않는다. 해당 에이전트의 발언만
> 시스템 경고로 대체되고 토론은 계속 진행된다(우아한 부분 실패 수용).

### 3) 서버 기동

```bash
# discussion_system/ 디렉터리에서
uvicorn app.main:app --reload                      # 개발용 (코드 변경 시 자동 리로드)
uvicorn app.main:app --host 0.0.0.0 --port 8000    # 배포용
```

| 진입점 | 주소 |
|--------|------|
| 웹 UI | <http://127.0.0.1:8000/> |
| Swagger UI | <http://127.0.0.1:8000/docs> |
| 헬스 체크 | <http://127.0.0.1:8000/health> |

LLM 클라이언트 연결 풀은 서버 기동 시 1회 준비되어 모든 토론 세션이 공유하며,
서버 종료(SIGTERM) 시 FastAPI `lifespan` 이 일괄 정리한다.

### 4) Docker 로 실행 (web + PostgreSQL)

저장소 루트의 `Dockerfile`(멀티 스테이지) · `docker-compose.yml` 로 FastAPI
서버와 PostgreSQL 을 함께 띄운다.

```bash
# 저장소 루트에서
export OPENAI_API_KEY="sk-..."      # (선택) 호스트 키를 web 컨테이너로 전달
docker compose up --build
```

`db`(PostgreSQL) 헬스체크가 통과한 뒤 `web` 이 기동되며, `web` 은
`DATABASE_URL=postgresql+asyncpg://…@db:5432/agora` 로 내부 네트워크의 DB 에
접속한다. 토론 데이터는 `agora_pgdata` 영속 볼륨에 보존된다.

## API 요약

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET  | `/` | 단일 파일 웹 UI (index.html) |
| POST | `/discussions` | 토론 생성 + 파이프라인 기동 |
| GET  | `/discussions/{id}` | 토론 전체 상태 스냅샷 조회 |
| POST | `/discussions/{id}/advance` | 게이트 락 해제 (다음 단계 진입 승인) |
| POST | `/discussions/{id}/interventions` | 유저 개입 주입 |
| POST | `/discussions/{id}/manual-response` | 수동 에이전트 응답 주입 (복붙 터널) |
| WS   | `/discussions/{id}/ws` | 진행 상황 실시간 스트림 + 개입 채널 |

## 공급자 선택

`AgentConfig.provider` 로 공급자를 명시하거나, 생략 시 `model` 명 접두사
(`gpt*`→openai, `claude*`→anthropic, `llama*` 등→ollama)에서 추론한다.

`provider="manual"` 인 에이전트는 API 를 호출하지 않는다. 그 턴이 오면 세션이
`PENDING_MANUAL_INPUT` 으로 대기하고, 웹 UI 에 딥/일반 복사본과 붙여넣기 창이
나타난다. 유저가 외부 LLM 응답을 붙여넣어 `POST /discussions/{id}/manual-response`
로 제출하면 파이프라인이 재구동된다.

## 현재 상태 — v0.6.0 (이벤트 구동형 무상태 오케스트레이션 + 비동기 DB)

**구현 완료 (전 단계 누적)**

- 전체 데이터 스키마 (`schemas.py`, Pydantic v2) + REST/WebSocket 라우터
- LLM 오케스트레이션:
  - 멀티 공급자 비동기 LLM 호출 + 토큰 단위 스트리밍
  - 동적 프롬프트 조립 / 1·2단계 순차 포스팅 / 우아한 부분 실패 수용
  - 5단계 `force_consensus` 분기 / 콘텍스트 압축(LTM)
- 영속성 (`database.py`/`models.py`) + `manual` 복붙 터널 + 단일 파일 웹 UI
- 앱 레벨 글로벌 `LLMClientPool`
- **phase-6 — 이벤트 구동형 무상태 오케스트레이션:**
  - 거대한 `_run_pipeline` 루프를 제거하고 단일 진입점
    `Orchestrator.process_event(id, event, payload)` 로 재편 — 매 이벤트가
    DB 로드 → 전이 → 저장 → 종료. 단계 사이·수동 입력 대기 중 메모리 점유 0.
  - 크래시 복구 — `lifespan` startup 의 `recover()` 가 DB 를 스캔해 `RUNNING`
    세션은 멱등 재기동, `PENDING_MANUAL_INPUT` 세션은 유실 없이 보존.
  - 낙관적 락 — `version` 컬럼 기반 `update_state` 가 동시 갱신 충돌을
    `StaleStateError` 로 검출하고 재시도한다.
- **phase-7 — 비동기 DB + 컨테이너화:**
  - 완전 비동기 영속성 — `AsyncSession`/`create_async_engine`, 모든 DB
    진입점이 `await`. `asyncio.to_thread` 스레드 오프로딩 제거.
  - `DATABASE_URL` 환경 변수로 `sqlite+aiosqlite` ↔ `postgresql+asyncpg` 전환.
  - `Dockerfile`(멀티 스테이지) + `docker-compose.yml`(web + PostgreSQL).
