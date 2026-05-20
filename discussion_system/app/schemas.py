"""Pydantic 데이터 스키마.

토론 시스템의 모든 상태/메시지 모델을 정의한다. 이 모듈은 순수 데이터
계층으로, 오케스트레이션 로직(`manager.py`)에 의존하지 않는다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """타임존 인식(UTC) 현재 시각."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 열거형
# ---------------------------------------------------------------------------
class DiscussionPhase(str, Enum):
    """5단계 토론 프로토콜의 단계 식별자."""

    IDLE = "idle"                              # 시작 전
    PHASE_1_OPINION = "phase_1_opinion"        # 1단계: 초기 주장 제시
    PHASE_2_CRITIQUE = "phase_2_critique"      # 2단계: 상호 비판
    PHASE_3_REBUTTAL = "phase_3_rebuttal"      # 3단계: 반론 및 방어
    PHASE_4_REVISION = "phase_4_revision"      # 4단계: 입장 수정
    PHASE_5_CONCLUSION = "phase_5_conclusion"  # 5단계: 최종 입장 / 합의
    COMPLETED = "completed"                    # 종료


class DiscussionStatus(str, Enum):
    """토론 세션의 실행 상태."""

    CREATED = "created"                    # 생성됨, 파이프라인 미기동
    RUNNING = "running"                    # 단계 실행 중
    WAITING_FOR_USER = "waiting_for_user"  # 단계 종료, 게이트 락 — 유저 개입 대기
    PENDING_MANUAL_INPUT = "pending_manual_input"  # 수동 에이전트 응답 입력 대기
    COMPLETED = "completed"                # 5단계까지 정상 종료
    ERROR = "error"                        # 오류로 중단


class ModelProvider(str, Enum):
    """에이전트가 사용할 LLM 공급자."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    MANUAL = "manual"  # API 미호출 — 유저가 웹 UI 에서 복붙으로 응답 주입


class PersonaType(str, Enum):
    """에이전트의 토론 성향. UI 말풍선 색상 구분에 사용된다."""

    PROPONENT = "proponent"        # 찬성론자
    OPPONENT = "opponent"          # 반대론자
    FACT_CHECKER = "fact_checker"  # 팩트체커
    MEDIATOR = "mediator"          # 중재자
    ANALYST = "analyst"            # 분석가
    NEUTRAL = "neutral"            # 중립


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
    ERROR = "error"                                # 오류 발생
    # --- C->S (클라이언트 -> 서버) ---
    USER_INTERVENTION = "user_intervention"        # 유저 개입 주입
    ADVANCE_PHASE = "advance_phase"                # 다음 단계 진입 승인


# ---------------------------------------------------------------------------
# 단계 메타데이터
# ---------------------------------------------------------------------------
#: 파이프라인이 순차 실행하는 단계 순서.
PHASE_SEQUENCE: list[DiscussionPhase] = [
    DiscussionPhase.PHASE_1_OPINION,
    DiscussionPhase.PHASE_2_CRITIQUE,
    DiscussionPhase.PHASE_3_REBUTTAL,
    DiscussionPhase.PHASE_4_REVISION,
    DiscussionPhase.PHASE_5_CONCLUSION,
]

#: 각 단계 -> `DiscussionState` 의 발언 기록 필드명 매핑.
PHASE_RECORD_FIELDS: dict[DiscussionPhase, str] = {
    DiscussionPhase.PHASE_1_OPINION: "phase_1_opinions",
    DiscussionPhase.PHASE_2_CRITIQUE: "phase_2_critiques",
    DiscussionPhase.PHASE_3_REBUTTAL: "phase_3_rebuttals",
    DiscussionPhase.PHASE_4_REVISION: "phase_4_revisions",
    DiscussionPhase.PHASE_5_CONCLUSION: "phase_5_conclusions",
}


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
        if name.startswith(
            ("llama", "mistral", "mixtral", "qwen", "gemma", "phi", "deepseek")
        ):
            return ModelProvider.OLLAMA
        raise ValueError(
            f"모델 '{self.model}' 의 공급자를 추론할 수 없습니다. "
            "AgentConfig.provider 를 명시하세요."
        )


class AgentTurn(BaseModel):
    """한 에이전트가 특정 단계에서 생성한 단일 발언 기록."""

    agent_id: str = Field(..., description="발언 주체 에이전트 ID")
    phase: DiscussionPhase = Field(..., description="발언이 속한 단계")
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

    phase: DiscussionPhase = Field(..., description="요약 대상 단계")
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
# 유저 개입
# ---------------------------------------------------------------------------
class UserIntervention(BaseModel):
    """단계 사이 게이트 락 구간에서 유저가 주입한 개입 기록."""

    message: str = Field(..., description="토론에 주입할 지시 / 코멘트")
    after_phase: Optional[DiscussionPhase] = Field(
        default=None, description="개입이 발생한 직전 단계 (None=시작 전)"
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
    agents: list[AgentConfig] = Field(
        ..., min_length=2, description="참여 에이전트 설정 목록 (2인 이상)"
    )

    # --- 실행 상태 ---
    status: DiscussionStatus = Field(default=DiscussionStatus.CREATED)
    current_phase: DiscussionPhase = Field(default=DiscussionPhase.IDLE)

    # --- 단계별 발언 기록 공간 ---
    phase_1_opinions: list[AgentTurn] = Field(
        default_factory=list, description="1단계: 초기 주장"
    )
    phase_2_critiques: list[AgentTurn] = Field(
        default_factory=list, description="2단계: 상호 비판"
    )
    phase_3_rebuttals: list[AgentTurn] = Field(
        default_factory=list, description="3단계: 반론 및 방어"
    )
    phase_4_revisions: list[AgentTurn] = Field(
        default_factory=list, description="4단계: 입장 수정"
    )
    phase_5_conclusions: list[AgentTurn] = Field(
        default_factory=list, description="5단계: 최종 입장 / 합의"
    )

    # --- 단계 요약 메트릭스 ---
    phase_summaries: list[PhaseSummary] = Field(
        default_factory=list, description="단계 종료마다 누적되는 요약"
    )

    # --- 유저 개입 기록 ---
    user_interventions: list[UserIntervention] = Field(
        default_factory=list, description="누적 유저 개입 기록"
    )

    # --- 옵션 플래그 ---
    force_consensus: bool = Field(
        default=False,
        description="True 시 5단계에서 합의를 강제 (미합의 에이전트도 합의안 수렴 유도)",
    )

    # --- 타임스탬프 / 오류 ---
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
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

    def record_for_phase(self, phase: DiscussionPhase) -> list[AgentTurn]:
        """주어진 단계의 발언 기록 리스트(참조)를 반환한다.

        IDLE/COMPLETED 처럼 기록 공간이 없는 단계는 `ValueError`.
        """
        field = PHASE_RECORD_FIELDS.get(phase)
        if field is None:
            raise ValueError(f"단계 '{phase}' 에는 발언 기록 공간이 없습니다.")
        return getattr(self, field)


# ---------------------------------------------------------------------------
# API 요청 / 응답 모델
# ---------------------------------------------------------------------------
class CreateDiscussionRequest(BaseModel):
    """토론 생성 요청 본문."""

    topic: str = Field(..., min_length=1, description="토론 주제")
    agents: list[AgentConfig] = Field(
        ..., min_length=2, description="참여 에이전트 목록 (2인 이상)"
    )
    force_consensus: bool = Field(default=False, description="5단계 합의 강제 여부")


class CreateDiscussionResponse(BaseModel):
    """토론 생성 응답."""

    discussion_id: str
    status: DiscussionStatus
    current_phase: DiscussionPhase


class ManualResponseRequest(BaseModel):
    """수동(manual) 에이전트의 응답 주입 요청 본문."""

    agent_id: str = Field(..., description="응답을 주입할 수동 에이전트 ID")
    phase: DiscussionPhase = Field(..., description="응답이 속한 단계")
    content: str = Field(..., min_length=1, description="유저가 붙여넣은 응답 본문")


class WSMessage(BaseModel):
    """WebSocket 으로 오가는 단일 메시지 봉투(envelope)."""

    type: WSMessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=_utcnow)
