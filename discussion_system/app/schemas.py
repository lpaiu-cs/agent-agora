"""Pydantic 데이터 스키마.

토론 시스템의 모든 상태/메시지 모델을 정의한다. 이 모듈은 순수 데이터
계층으로, 오케스트레이션 로직(`manager.py`)에 의존하지 않는다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .formats import PHASE_IDLE


def _utcnow() -> datetime:
    """타임존 인식(UTC) 현재 시각."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 열거형
# ---------------------------------------------------------------------------
class DiscussionStatus(str, Enum):
    """토론 세션의 실행 상태."""

    CREATED = "created"                    # 생성됨, 파이프라인 미기동
    RUNNING = "running"                    # 단계 실행 중
    WAITING_FOR_USER = "waiting_for_user"  # 단계 종료, 게이트 락 — 유저 개입 대기
    PENDING_MANUAL_INPUT = "pending_manual_input"  # 수동 에이전트 응답 입력 대기
    PENDING_REVIEW = "pending_review"      # 가로채기된 에이전트 초안 검토 대기
    COMPLETED = "completed"                # 정상 종료
    ERROR = "error"                        # 오류로 중단


class ModelProvider(str, Enum):
    """에이전트가 사용할 LLM 공급자."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"       # Google — OpenAI-호환 엔드포인트 (generativelanguage.googleapis.com)
    DEEPSEEK = "deepseek"   # OpenAI-호환 API (api.deepseek.com)
    OLLAMA = "ollama"
    MANUAL = "manual"  # API 미호출 — 유저가 웹 UI 에서 복붙으로 응답 주입


class PersonaType(str, Enum):
    """에이전트의 브레인스토밍 / 토론 성향. UI 말풍선 색상·라벨 구분에 쓰인다."""

    # --- 브레인스토밍 역할 (현행 UI 노출) ---
    IDEATOR = "ideator"          # 아이디어 발상가 — 새 아이디어를 자유롭게 발산
    BUILDER = "builder"          # 아이디어 확장가 — 남의 아이디어에 살을 붙임
    CRITIC = "critic"            # 비판적 검토자 — 약점을 건설적으로 짚음
    SYNTHESIZER = "synthesizer"  # 통합·정리자 — 흩어진 아이디어를 묶음
    PRAGMATIST = "pragmatist"    # 실용성 검토자 — 실행 가능성·제약을 따짐
    NEUTRAL = "neutral"          # 자유 참여자 — 특정 역할 없이 참여

    # --- 레거시 (구 토론 데이터 역직렬화 호환용 — UI 미노출) ---
    PROPONENT = "proponent"
    OPPONENT = "opponent"
    FACT_CHECKER = "fact_checker"
    MEDIATOR = "mediator"
    ANALYST = "analyst"


class WSMessageType(str, Enum):
    """WebSocket 메시지 타입.

    `S->C` 는 서버->클라이언트 브로드캐스트, `C->S` 는 클라이언트->서버 입력.
    """

    # --- S->C (서버 -> 클라이언트) ---
    STATE_SNAPSHOT = "state_snapshot"              # 접속 직후 전체 상태 스냅샷
    PHASE_STARTED = "phase_started"                # 단계 시작
    AGENT_TURN = "agent_turn"                      # 에이전트 발언 1건 완료(최종 텍스트)
    TOKEN_STREAM = "token_stream"                  # 발언 생성 중 토큰 청크(실시간)
    PHASE_COMPLETED = "phase_completed"            # 단계 종료 + 요약
    AWAITING_USER = "awaiting_user"                # 게이트 락 — 유저 개입 대기 알림
    DISCUSSION_COMPLETED = "discussion_completed"  # 토론 종료
    MANUAL_INPUT_REQUIRED = "manual_input_required"  # 수동 에이전트 입력 요청(복사 페이로드 포함)
    REVIEW_REQUIRED = "review_required"            # 가로채기 검토 요청(초안·사고흐름)
    REVIEW_ANSWER = "review_answer"                # 검토 문답 — 에이전트 답변
    FACILITATOR_NOTE = "facilitator_note"          # 사회자 진행 노트(개회·중간·폐회 등)
    ERROR = "error"                                # 오류 발생
    # --- C->S (클라이언트 -> 서버) ---
    USER_INTERVENTION = "user_intervention"        # 유저 개입 주입
    ADVANCE_PHASE = "advance_phase"                # 다음 단계 진입 승인


# ---------------------------------------------------------------------------
# 에이전트 모델
# ---------------------------------------------------------------------------
class AgentConfig(BaseModel):
    """토론에 참여하는 단일 에이전트의 정적 설정."""

    agent_id: str = Field(..., description="에이전트 고유 식별자")
    name: str = Field(..., description="표시 이름")
    model: str = Field(..., description="사용할 LLM 모델명 (예: claude-opus-4-7)")
    persona_prompt: str = Field(
        ..., description="에이전트의 페르소나 / 시스템 프롬프트"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="샘플링 온도"
    )
    max_tokens: int = Field(
        default=1024, ge=1, le=32768, description="응답 최대 토큰 수"
    )
    provider: Optional[ModelProvider] = Field(
        default=None,
        description="LLM 공급자. 미지정 시 model 명 접두사에서 추론한다.",
    )
    persona_type: PersonaType = Field(
        default=PersonaType.NEUTRAL,
        description="토론 성향. UI 말풍선 색상 구분에 사용된다.",
    )

    def get_provider(self) -> ModelProvider:
        """공급자를 결정한다. provider 가 없으면 model 명으로 추론한다.

        추론 불가 시 ``ValueError`` — 호출부에서 우아한 실패로 흡수된다.
        """
        if self.provider is not None:
            return self.provider
        name = self.model.lower()
        if name.startswith(("gpt", "o1", "o3", "o4", "chatgpt")):
            return ModelProvider.OPENAI
        if name.startswith("claude"):
            return ModelProvider.ANTHROPIC
        # Gemini(Google)는 OpenAI-호환 엔드포인트로 — 로컬 Ollama 의 'gemma' 와
        # 프리픽스가 달라 충돌하지 않는다.
        if name.startswith("gemini"):
            return ModelProvider.GEMINI
        # DeepSeek 모델은 api.deepseek.com 의 OpenAI-호환 엔드포인트로 — Ollama
        # 로컬에서 돌리려면 provider=ollama 를 명시.
        if name.startswith("deepseek"):
            return ModelProvider.DEEPSEEK
        if name.startswith(
            ("llama", "mistral", "mixtral", "qwen", "gemma", "phi")
        ):
            return ModelProvider.OLLAMA
        raise ValueError(
            f"모델 '{self.model}' 의 공급자를 추론할 수 없습니다. "
            "AgentConfig.provider 를 명시하세요."
        )


class AgentTurn(BaseModel):
    """한 에이전트가 특정 단계에서 생성한 단일 발언 기록."""

    agent_id: str = Field(..., description="발언 주체 에이전트 ID")
    phase: str = Field(..., description="발언이 속한 단계 (형식 내 단계 id)")
    content: str = Field(..., description="발언 본문")
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="토큰 사용량/지연시간 등 부가 정보 (phase-2 에서 채움)",
    )


# ---------------------------------------------------------------------------
# 단계 요약 메트릭스
# ---------------------------------------------------------------------------
class AgentStanceSummary(BaseModel):
    """단일 에이전트에 대한 단계별 입장 요약."""

    agent_id: str = Field(..., description="대상 에이전트 ID")
    initial_claim: str = Field(default="", description="에이전트의 초기 주장 요약")
    current_stance: str = Field(default="", description="현재 단계 기준 기조 / 입장")
    stance_shift: str = Field(default="", description="직전 단계 대비 입장 변화 요약")


class PhaseSummary(BaseModel):
    """한 단계 종료 시점의 토론 요약 메트릭스.

    유저가 게이트 락 구간에서 다음 단계 진입 여부를 판단하는 근거가 된다.
    """

    phase: str = Field(..., description="요약 대상 단계 (형식 내 단계 id)")
    agent_summaries: list[AgentStanceSummary] = Field(
        default_factory=list, description="에이전트별 입장 요약"
    )
    key_conflicts: list[str] = Field(
        default_factory=list, description="단계에서 드러난 주요 갈등 / 쟁점"
    )
    convergence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="합의 근접도 (0=완전 대립, 1=완전 합의)",
    )
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# 검토 게이트 (선택적 가로채기)
# ---------------------------------------------------------------------------
class ReviewExchange(BaseModel):
    """검토 게이트에서 진행자와 에이전트가 주고받은 문답 1쌍."""

    question: str = Field(..., description="진행자가 던진 질문")
    answer: str = Field(..., description="에이전트의 답변")
    created_at: datetime = Field(default_factory=_utcnow)


class ReviewState(BaseModel):
    """가로채기된 에이전트 턴의 검토 세션 — 초안·사고흐름·문답을 담는다."""

    agent_id: str = Field(..., description="검토 대상 에이전트 ID")
    phase: str = Field(..., description="검토 중인 단계 id")
    reasoning: str = Field(default="", description="에이전트의 사고흐름")
    draft: str = Field(default="", description="에이전트의 발언 초안")
    qa: list[ReviewExchange] = Field(
        default_factory=list, description="진행자-에이전트 문답 기록"
    )


# ---------------------------------------------------------------------------
# 사회자(facilitator) 에이전트
# ---------------------------------------------------------------------------
class FacilitatorNote(BaseModel):
    """사회자가 단계 경계에서 남긴 진행 노트.

    사회자는 토론자가 아니다 — 입장을 갖지 않고 토론을 조율하므로, 그 발언은
    ``phase_records`` 가 아닌 별도 노트로 누적된다.
    """

    phase: str = Field(..., description="노트가 속한(직전/직후) 단계 인스턴스 키")
    kind: str = Field(
        ...,
        description="노트 종류: open(개회)·between(중간)·close(폐회)·decision(진행 결정)",
    )
    content: str = Field(..., description="사회자 발언 본문")
    decision: Optional[str] = Field(
        default=None,
        description="진행 결정 (kind=decision 일 때): continue·next·conclude",
    )
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="토큰 사용량 등 부가 정보"
    )


# ---------------------------------------------------------------------------
# 유저 개입
# ---------------------------------------------------------------------------
class UserIntervention(BaseModel):
    """단계 사이 게이트 락 구간에서 유저가 주입한 개입 기록."""

    message: str = Field(..., description="토론에 주입할 지시 / 코멘트")
    after_phase: Optional[str] = Field(
        default=None, description="개입이 발생한 직전 단계 id (None=시작 전)"
    )
    target_agent_id: Optional[str] = Field(
        default=None, description="특정 에이전트 지정 시 대상 ID (None=전체)"
    )
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# 토론 중심 상태
# ---------------------------------------------------------------------------
class DiscussionState(BaseModel):
    """토론 세션 전체의 중심 상태 객체.

    파이프라인의 모든 단계가 이 객체를 단일 진실 공급원(SSOT)으로 읽고 쓴다.
    """

    # --- 식별 / 정적 설정 ---
    discussion_id: str = Field(..., description="토론 세션 고유 ID")
    topic: str = Field(..., description="토론 주제")
    format_id: str = Field(
        default="debate", description="토론 형식 id (formats.FORMATS 의 키)"
    )
    agents: list[AgentConfig] = Field(
        ..., min_length=2, description="참여 에이전트 설정 목록 (2인 이상)"
    )

    # --- 실행 상태 ---
    status: DiscussionStatus = Field(default=DiscussionStatus.CREATED)
    current_phase: str = Field(
        default=PHASE_IDLE,
        description="현재 단계 id. 'idle'(시작 전)·'completed'(종료)는 예약값.",
    )

    # --- 단계별 발언 기록 — 단계 id -> 그 단계의 발언 턴 목록 ---
    phase_records: dict[str, list[AgentTurn]] = Field(
        default_factory=dict, description="단계 id 별 발언 기록"
    )

    # --- 단계 요약 메트릭스 ---
    phase_summaries: list[PhaseSummary] = Field(
        default_factory=list, description="단계 종료마다 누적되는 요약"
    )

    # --- 유저 개입 기록 ---
    user_interventions: list[UserIntervention] = Field(
        default_factory=list, description="누적 유저 개입 기록"
    )

    # --- 검토 게이트 (선택적 가로채기) ---
    intercept_agents: list[str] = Field(
        default_factory=list,
        description="다음 턴을 검토 게이트로 가로챌 에이전트 ID 목록",
    )
    review: Optional[ReviewState] = Field(
        default=None,
        description="진행 중인 검토 세션 (status=PENDING_REVIEW 일 때만 채워짐)",
    )

    # --- 사회자(facilitator) 에이전트 ---
    facilitator: Optional[AgentConfig] = Field(
        default=None,
        description="사회자 에이전트 — 단계 경계에서 진행을 조율한다. None=사회자 없음.",
    )
    facilitator_notes: list[FacilitatorNote] = Field(
        default_factory=list,
        description="사회자가 남긴 진행 노트(개회·중간·폐회 등) 누적",
    )

    # --- 옵션 플래그 ---
    force_consensus: bool = Field(
        default=False,
        description="True 시 5단계에서 합의를 강제 (미합의 에이전트도 합의안 수렴 유도)",
    )
    reference_materials: Optional[str] = Field(
        default=None,
        description=(
            "선택 참고 자료 (텍스트·발췌·URL 목록). 지정 시 모든 에이전트 "
            "프롬프트에 공통 자료로 주입되고 인용 규칙이 활성화된다. "
            "None 이면 동작 불변 — 주제에 따라 강제하지 않는다."
        ),
    )

    # --- 타임스탬프 / 오류 ---
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    version: int = Field(
        default=0,
        description="낙관적 락(optimistic lock) 버전 — DB 저장에 성공할 때마다 +1",
    )
    error: Optional[str] = Field(
        default=None, description="status=ERROR 일 때의 오류 메시지"
    )

    # --- 5단계 합의 산출물 ---
    final_joint_agreement: Optional[str] = Field(
        default=None,
        description="force_consensus=True 일 때 5단계에서 도출된 단일 합의안 문서",
    )

    # -- 헬퍼 --------------------------------------------------------------
    def touch(self) -> None:
        """`updated_at` 을 현재 시각으로 갱신한다."""
        self.updated_at = _utcnow()

    def record_for_phase(self, phase: str) -> list[AgentTurn]:
        """주어진 단계 id 의 발언 기록 리스트(참조)를 반환한다.

        해당 단계의 칸이 아직 없으면 빈 리스트를 만들어 등록한 뒤 반환한다 —
        호출부는 이 참조에 발언을 append 한다.
        """
        return self.phase_records.setdefault(phase, [])


# ---------------------------------------------------------------------------
# API 요청 / 응답 모델
# ---------------------------------------------------------------------------
class CreateDiscussionRequest(BaseModel):
    """토론 생성 요청 본문."""

    topic: str = Field(..., min_length=1, description="토론 주제")
    format_id: str = Field(
        default="debate", description="토론 형식 id (기본 debate)"
    )
    agents: list[AgentConfig] = Field(
        ..., min_length=2, description="참여 에이전트 목록 (2인 이상)"
    )
    facilitator: Optional[AgentConfig] = Field(
        default=None,
        description="사회자 에이전트 (선택). 지정하면 단계 경계마다 진행을 조율한다.",
    )
    force_consensus: bool = Field(
        default=False, description="마지막 단계 합의 강제 여부 (형식이 지원할 때만)"
    )
    reference_materials: Optional[str] = Field(
        default=None,
        description="선택 참고 자료 — 지정 시에만 에이전트가 인용 규칙을 적용한다",
    )


class CreateDiscussionResponse(BaseModel):
    """토론 생성 응답."""

    discussion_id: str
    status: DiscussionStatus
    current_phase: str


class ManualResponseRequest(BaseModel):
    """수동(manual) 에이전트의 응답 주입 요청 본문."""

    agent_id: str = Field(..., description="응답을 주입할 수동 에이전트 ID")
    phase: str = Field(..., description="응답이 속한 단계 id")
    content: str = Field(..., min_length=1, description="유저가 붙여넣은 응답 본문")


class SetInterceptsRequest(BaseModel):
    """검토 게이트로 가로챌 에이전트 지정 요청 (빈 목록이면 가로채기 해제)."""

    agent_ids: list[str] = Field(
        default_factory=list, description="가로챌 API 에이전트 ID 목록"
    )


class ReviewQuestionRequest(BaseModel):
    """검토 중인 에이전트에게 던지는 진행자 질문."""

    question: str = Field(..., min_length=1, description="진행자의 질문")


class RefinePersonaRequest(BaseModel):
    """페르소나 초안 윤문 요청.

    사용자가 대강 쓴 페르소나 초안을 토론 주제 맥락에 맞춰 다듬는다. 윤문은
    해당 에이전트 슬롯에 설정된 ``provider``/``model`` 로 수행한다.
    """

    topic: str = Field(..., min_length=1, description="토론 주제 (윤문 맥락)")
    draft: str = Field(..., min_length=1, description="사용자가 대강 쓴 페르소나 초안")
    provider: ModelProvider = Field(..., description="윤문에 사용할 LLM 공급자")
    model: str = Field(..., min_length=1, description="윤문에 사용할 모델명")
    name: str = Field(default="", description="에이전트 표시 이름 (맥락 보강용, 선택)")
    persona_type: Optional[PersonaType] = Field(
        default=None, description="에이전트 역할 (맥락 보강용, 선택)"
    )


class RefinePersonaResponse(BaseModel):
    """페르소나 윤문 결과."""

    refined: str = Field(..., description="주제에 맞춰 윤문된 페르소나 프롬프트")


class WSMessage(BaseModel):
    """WebSocket 으로 오가는 단일 메시지 봉투(envelope)."""

    type: WSMessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)
