# Agent Agora — 다중 에이전트 LLM 토론 시스템

여러 LLM 에이전트가 하나의 주제를 두고 **다단계 구조화 프로토콜**(구조화 토론·
브레인스토밍 등 선택 가능한 형식)로 협의하도록 오케스트레이션하는 FastAPI 기반
백엔드 + 단일 파일 웹 UI.

> **릴리즈: v0.6.0-final** — 이벤트 구동형 무상태 오케스트레이션 · 완전 비동기 DB ·
> 크래시 복구 · LLM 호출 백프레셔 · 컨테이너화 완료.

---

## 핵심 기능

- **다중 토론 형식** — 구조화 토론(5단계)·브레인스토밍(4단계)·소크라테스식 문답(가변 길이). 형식마다 단계 구성·프롬프트가 다르며 코드로 추가 가능
- **가변 길이 형식** — 반복(`repeatable`) 단계가 합의 근접도에 따라 라운드를 거듭한다 (소크라테스식 문답의 문답 라운드)
- **사회자(facilitator) 에이전트** — 선택적 사회자가 단계 경계에서 개회·중간 조율·폐회로 진행을 조율하고, 가변 길이 형식에서는 반복 라운드 지속 여부까지 판단 (토론자가 아닌 부가 레이어)
- **멀티 공급자 LLM** — OpenAI · Anthropic · Gemini · DeepSeek · Ollama, 그리고 복붙 기반 `manual` 공급자
- **토큰 단위 실시간 스트리밍** — WebSocket 으로 발언이 생성되는 과정을 그대로 전송
- **이벤트 구동형 무상태 오케스트레이션** — 매 이벤트가 `DB 로드 → 전이 → 저장 → 종료`,
  단계 사이·입력 대기 중 메모리 점유 0
- **크래시 복구** — 서버 재기동 시 DB 를 스캔해 진행 중이던 토론을 멱등 재기동
- **낙관적 락(optimistic lock)** — `version` 컬럼으로 동시 갱신 충돌을 검출·재시도
- **LLM 호출 백프레셔** — 동시 추론 호출 수를 세마포어로 엄격히 제한 (CPU 폭주 방지)
- **우아한 부분 실패 수용** — 일부 에이전트 호출이 실패해도 토론은 중단 없이 진행
- **유저 개입** — 단계 사이 게이트 락 구간에서 진행자가 지시를 주입
- **단일 파일 웹 UI** — HTML + Vanilla JS, WS 자동 재연결 및 작성 중 텍스트 보존

---

## 토론 형식

토론 진행 구조는 **형식(`DiscussionFormat`)**으로 정의된다 — 형식마다 단계 개수·
순서·지침·순차성이 다르다. 형식은 `app/formats.py` 레지스트리에 코드로 등록되며,
토론 생성 시 `format_id` 로 선택한다 (`GET /formats` 로 목록 조회).

**구조화 토론 (`debate`) — 5단계**

| 단계 | 진행 방식 | 설명 |
|------|:---:|------|
| 초기 주장 (opinion)    | 동시 | 각자 서로의 발제를 보지 않고 독립적으로 입장·논거 제시 |
| 상호 비판 (critique)   | 순차 | 후순위 에이전트가 같은 단계 선행 비판을 맥락으로 받아 중복 회피 |
| 반론·방어 (rebuttal)   | 동시 | 받은 비판에 반론하고 자기 입장을 방어 |
| 입장 수정 (revision)   | 동시 | 토론을 반영해 입장을 갱신 |
| 최종 결론 (conclusion) | 동시 | 최종 입장 정리 (`force_consensus` 시 단일 합의안 도출) |

**브레인스토밍 (`brainstorm`) — 4단계**

| 단계 | 진행 방식 | 설명 |
|------|:---:|------|
| 아이디어 발산 (diverge) | 동시 | 제약 없이 새 아이디어를 발산 |
| 상호 확장 (expand)      | 순차 | 유망한 아이디어에 '예, 그리고'로 살을 붙여 확장 |
| 수렴·선별 (converge)    | 동시 | 가장 가치 있는 아이디어를 선별 |
| 실행안 (action)         | 동시 | 선별된 아이디어의 실행 단계·제약 제시 |

**소크라테스식 문답 (`socratic`) — 가변 길이**

총 단계 수가 런타임에 결정되는 형식이다. 문답 라운드(`probe`)가 **합의 근접도**에
따라 반복된다 — 최소 2회는 진행하고, 근접도가 임계값(0.8)에 닿거나 최대 6회에
이르면 종합 단계로 넘어간다.

| 단계 | 진행 방식 | 설명 |
|------|:---:|------|
| 입장 제시 (position)  | 동시 | 각자의 입장과 그것이 기댄 핵심 전제를 제시 |
| 문답 라운드 (probe)   | 순차 · **반복(2~6회)** | 상대 전제를 캐묻고 받은 질문에 답하며 라운드마다 논점을 좁힘 |
| 종합 (synthesis)      | 동시 | 도달한 최종 입장과 남은 이견 정리 (`force_consensus` 시 단일 합의안) |

반복 단계는 `PhaseSpec(repeatable=True, min_rounds, max_rounds, converge_threshold)`
로 정의하며, 단계 진행은 `formats.plan_next()` 가 합의 근접도를 보고 결정한다.

**동시** 단계는 `asyncio.gather` 병렬 호출이며 서로의 발제를 보지 않는다.
**순차** 단계만 후순위 에이전트가 같은 단계 선행 의견을 맥락으로 받는다. 각 단계가
끝나면 파이프라인이 게이트 락(`waiting_for_user`)되어 유저 개입을 기다린다.

오래된 단계는 콘텍스트 압축(LTM)으로 원문 대신 단계 요약 메트릭스
(`phase_summaries`)를 프롬프트에 주입한다 — 최근 2개 단계는 원문을 유지한다.

### 사회자(facilitator) 에이전트

토론 생성 시 **사회자**를 선택적으로 지정할 수 있다. 사회자는 토론자가 아니다 —
입장을 갖지 않고 단계 경계에서 진행을 조율한다.

- **개회(open)** — 토론 시작 시 핵심 쟁점·첫 단계 초점 안내
- **중간 조율(between)** — 단계가 끝날 때마다 미해결 쟁점을 짚고 다음 초점 제시
- **진행 결정(decision)** — 가변 길이 형식의 반복 단계 게이트에서 라운드를 계속할지
  (`continue`)·다음 단계로 넘길지(`next`)·토론을 끝낼지(`conclude`) 판단
- **폐회(close)** — 토론 종료 시 합의점·잔존 이견 정리

사회자가 있으면 반복 단계의 종료는 합의 근접도 숫자 대신 사회자의 `decision` 이
구동한다 — 즉 사회자가 곧 소크라테스식 문답의 진행자가 된다. 사회자는 완전한
opt-in 이며(`facilitator=null` 이면 동작 불변), 호출이 실패하면 숫자 예측자로
우아하게 되돌아간다 — 그 발언은 `phase_records` 가 아닌 `facilitator_notes` 로
누적된다.

---

## 아키텍처

```
agent-agora/
├── Dockerfile               # 멀티 스테이지 프로덕션 이미지 (python:3.12-slim)
├── docker-compose.yml       # web(FastAPI) + db(PostgreSQL) 멀티 컨테이너
├── README.md                # (이 문서)
└── discussion_system/
    ├── app/
    │   ├── __init__.py       # 버전 상수
    │   ├── main.py           # FastAPI 진입점 (lifespan: DB·풀 초기화/정리)
    │   ├── schemas.py        # Pydantic v2 데이터 모델 (상태·요청·WS 메시지)
    │   ├── formats.py        # 토론 형식 정의·진행 결정 (debate · brainstorm · socratic)
    │   ├── manager.py        # 이벤트 구동형 무상태 Orchestrator + LLM 연동
    │   ├── database.py       # 비동기 영속성 레이어 (저장/조회 + 낙관적 락)
    │   ├── models.py         # SQLAlchemy ORM 모델 + 상태 <-> 행 변환
    │   ├── routers/
    │   │   └── discussion.py # REST + WebSocket 엔드포인트
    │   └── templates/
    │       └── index.html    # 단일 파일 웹 UI
    ├── tests/                # pytest 회귀 테스트 스위트
    ├── requirements.txt
    └── requirements-dev.txt  # 테스트 의존성 (pytest)
```

- **상태는 100% DB** — `Orchestrator` 는 토론 상태를 인메모리에 들지 않고, 인프라
  의존성(LLM 풀·브로드캐스트 콜백)만 보유한다.
- **인프라 객체는 `app.state`** — LLM 풀·오케스트레이터·소켓 레지스트리의 생명주기를
  FastAPI `lifespan` 이 소유하고, 라우터는 `Depends` 로 주입받는다.

**기술 스택:** Python 3.10+ · FastAPI · Pydantic v2 · SQLAlchemy 2.0(async) · uvicorn

---

## 빠른 시작 (로컬)

### 1) 설치

```bash
cd discussion_system
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) 환경 변수 설정

저장소 루트의 `.env.example` 을 `.env` 로 복사하고 필요한 키만 채우면 된다 —
앱은 기동 시 `.env` 를 자동 로드한다(`python-dotenv`). 환경 변수로 직접
export 해도 동일하게 동작한다. 모두 선택 사항이며, `manual` 공급자만 쓰면
키 없이도 토론이 굴러간다.

```bash
cp .env.example .env
# 그 다음 .env 를 편집해 필요한 키를 채운다.
```

| 변수 | 용도 |
|------|------|
| `OPENAI_API_KEY` | OpenAI(gpt-*) 공급자 사용 시 필수 |
| `ANTHROPIC_API_KEY` | Anthropic(claude-*) 공급자 사용 시 필수 |
| `GEMINI_API_KEY` | Gemini(gemini-*) 공급자 사용 시 필수 (`GOOGLE_API_KEY` 도 인식) — Google 의 OpenAI-호환 엔드포인트 사용 |
| `DEEPSEEK_API_KEY` | DeepSeek(deepseek-*) 공급자 사용 시 필수 — `api.deepseek.com` 의 OpenAI-호환 엔드포인트 사용 |
| `OLLAMA_HOST` | (선택) Ollama 호스트 주소. 기본 `http://localhost:11434` |
| `DATABASE_URL` | (선택) 비동기 DB URL. 기본 `sqlite+aiosqlite:///./agora.db`. `postgresql+asyncpg://…` 도 그대로 수용 (`AGORA_DB_URL` 은 하위 호환 폴백) |
| `AGORA_MAX_CONCURRENT_LLM` | (선택) 동시 LLM 호출 상한. 기본 `8`. 미설정·형식 오류·0 이하 값은 기본값으로 흡수 |

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

---

## Docker 로 실행 (web + PostgreSQL)

저장소 루트의 `Dockerfile`(멀티 스테이지) · `docker-compose.yml` 로 FastAPI 서버와
PostgreSQL 을 함께 띄운다.

```bash
# 저장소 루트에서
export OPENAI_API_KEY="sk-..."      # (선택) 호스트 키를 web 컨테이너로 전달
docker compose up --build
```

`db`(PostgreSQL) 헬스체크가 통과한 뒤 `web` 이 기동되며, `web` 은
`DATABASE_URL=postgresql+asyncpg://…@db:5432/agora` 로 내부 네트워크의 DB 에
접속한다. 토론 데이터는 `agora_pgdata` 영속 볼륨에 보존된다.

---

## API 레퍼런스

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET`  | `/` | 단일 파일 웹 UI |
| `GET`  | `/health` | 헬스 체크 (`{"status":"ok","version":...}`) |
| `POST` | `/discussions` | 토론 생성 + 파이프라인 기동 |
| `GET`  | `/discussions/{id}` | 토론 전체 상태 스냅샷 조회 |
| `POST` | `/discussions/{id}/advance` | 게이트 락 해제 (다음 단계 진입 승인) |
| `POST` | `/discussions/{id}/interventions` | 유저 개입 주입 |
| `POST` | `/discussions/{id}/manual-response` | 수동 에이전트 응답 주입 (복붙 터널) |
| `POST` | `/discussions/{id}/intercept` | 검토 게이트로 가로챌 에이전트 지정 |
| `POST` | `/discussions/{id}/review/question` | 검토 중인 에이전트에 질문 |
| `POST` | `/discussions/{id}/review/approve` | 검토 초안 승인 → 최종 포스팅 |
| `GET`  | `/discussions/{id}/export` | 토론 기록 마크다운 다운로드 |
| `POST` | `/discussions/{id}/archive` | 토론 기록을 로컬 `discussions/` 폴더에 저장 |
| `WS`   | `/discussions/{id}/ws` | 진행 상황 실시간 스트림 + 개입 채널 |
| `GET`  | `/formats` | 등록된 토론 형식 목록 (단계 구성 포함) |
| `POST` | `/personas/refine` | 페르소나 초안을 주제에 맞춰 LLM 으로 윤문 |

### 토론 생성 예시

```bash
curl -X POST http://127.0.0.1:8000/discussions \
  -H 'Content-Type: application/json' \
  -d '{
    "topic": "원격 근무는 조직 생산성을 높이는가?",
    "format_id": "debate",
    "agents": [
      {"agent_id": "a1", "name": "찬성 측", "model": "gpt-4o",
       "persona_prompt": "너는 원격 근무 옹호론자다.", "persona_type": "proponent"},
      {"agent_id": "a2", "name": "반대 측", "model": "claude-3-5-sonnet-20241022",
       "persona_prompt": "너는 대면 근무 옹호론자다.", "persona_type": "opponent"}
    ],
    "force_consensus": false
  }'
```

- `format_id`(기본 `debate`)·`force_consensus` 는 선택. `agents` 는 **2인 이상** 필수.
- `persona_prompt` 는 각 에이전트의 시스템 프롬프트. `temperature`(기본 0.7)·
  `max_tokens`(기본 1024)·`provider`·`persona_type` 는 선택.

### WebSocket 메시지 타입

| 방향 | 타입 | 의미 |
|:---:|------|------|
| S→C | `state_snapshot` | 접속 직후 전체 상태 스냅샷 |
| S→C | `phase_started` / `phase_completed` | 단계 시작 / 종료(+요약) |
| S→C | `agent_turn` | 에이전트 발언 1건 완료 (최종 텍스트) |
| S→C | `token_stream` | 발언 생성 중 토큰 청크 (실시간) |
| S→C | `awaiting_user` | 게이트 락 — 유저 개입 대기 알림 |
| S→C | `manual_input_required` | 수동 에이전트 입력 요청 (복사 페이로드 포함) |
| S→C | `discussion_completed` / `error` | 토론 종료 / 오류 |
| C→S | `user_intervention` | 유저 개입 주입 |
| C→S | `advance_phase` | 다음 단계 진입 승인 |

---

## 공급자 선택과 `manual` 복붙 터널

`AgentConfig.provider` 로 공급자를 명시하거나, 생략 시 `model` 명 접두사에서
추론한다 (`gpt*`·`o1/o3/o4*` → OpenAI, `claude*` → Anthropic, `gemini*` →
Gemini, `deepseek*` → DeepSeek, `llama*`·`mistral*`·`qwen*`·`gemma*` 등 →
Ollama). Gemini·DeepSeek 은 각자의 OpenAI-호환 엔드포인트를 쓰며, 로컬 Ollama
의 `gemma*` 와 Google `gemini*` 는 접두사가 달라 충돌하지 않는다.

`provider="manual"` 인 에이전트는 API 를 호출하지 않는다. 그 턴이 오면 세션이
`pending_manual_input` 으로 대기하고, 웹 UI 에 딥/일반 복사본과 붙여넣기 창이
나타난다. 유저가 외부 LLM 의 응답을 붙여넣어 `POST /discussions/{id}/manual-response`
로 제출하면 파이프라인이 재구동된다 — **API 키 없이도 토론을 끝까지 진행할 수 있다.**

---

## 동작 특성

- **이벤트 구동 / 무상태** — 단일 진입점 `Orchestrator.process_event(id, event, payload)`.
  거대한 실행 루프가 없어 단계 사이·입력 대기 중 메모리를 점유하지 않는다.
- **크래시 복구** — `lifespan` startup 의 `recover()` 가 DB 를 스캔해 `running` 세션은
  멱등 재기동, `pending_manual_input` 세션은 유실 없이 보존한다.
- **낙관적 락** — `update_state` 가 `UPDATE … WHERE id=? AND version=?` 로 동시 갱신
  충돌을 `StaleStateError` 로 검출하고 재시도한다 (SQLite·PostgreSQL 양쪽 원자적).
- **LLM 호출 백프레셔** — 모든 LLM 경로가 거치는 `_invoke_agent` 가 고정 크기
  세마포어로 동시 추론 호출 수를 제한한다. 토론 수가 폭증해도 CPU 스파이크와 락
  경합 폭증을 억제한다 (`AGORA_MAX_CONCURRENT_LLM`, 기본 8).
- **우아한 부분 실패** — LLM 키 미설정·호출 실패 시 해당 발언만 `[시스템 경고]` 턴으로
  대체되고 토론은 계속된다.
- **WS 회복력** — 웹 UI 는 연결이 끊기면 지수 백오프로 자동 재연결하고, 재연결 시
  타임라인을 스냅샷으로 재구성하면서 작성 중이던 입력 텍스트를 보존한다.

---

## 요구 사항

- **Python 3.10+** (`asyncio` 동기화 객체의 루프 비종속 생성)
- 주요 의존성: `fastapi` · `uvicorn[standard]` · `pydantic>=2.9` · `openai` ·
  `anthropic` · `ollama` · `sqlalchemy>=2.0` · `aiosqlite` · `asyncpg`
  (전체 목록은 `discussion_system/requirements.txt`)
- Docker 이미지는 `python:3.12-slim` 기반 멀티 스테이지 빌드

---

*Agent Agora · 버전 0.6.0 (v0.6.0-final)*
