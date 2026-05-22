"""1단계 동시 발제 격리 — 회귀 방지의 핵심 테스트.

1단계는 동시 발제 단계라 후순위 에이전트가 선행 발제를 보면 안 된다.
2단계는 순차라 같은 단계 선행 의견을 맥락으로 받아야 한다.
"""
from app import database
from app.manager import PipelineEvent
from app.schemas import ModelProvider

SEQ_MARKER = "이번 단계 선행 의견"   # 순차 단계에서만 프롬프트에 들어가는 머리말
P1_MARK = "[1단계 · 초기주장]"        # 1단계 지시문(대괄호) — 1단계 에이전트 프롬프트만 매칭
P2_MARK = "[2단계 · 상호비판]"        # 2단계 지시문


async def test_phase1_concurrent_isolated_phase2_sequential_keeps_context(
    orchestrator, make_state, patch_llm,
):
    """API 경로 — 1단계 프롬프트엔 선행 발제 없음, 2단계엔 있음."""
    prompts: list[str] = []

    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        prompts.append(user)
        return "테스트 응답 본문."

    patch_llm(fake)
    await database.insert_state(make_state(discussion_id="api"))   # 기본 2인 API

    await orchestrator.process_event("api", PipelineEvent.START)
    st = await database.load_state("api")
    if st.status.value == "waiting_for_user":
        await orchestrator.process_event("api", PipelineEvent.ADVANCE)

    p1 = [p for p in prompts if P1_MARK in p]
    p2 = [p for p in prompts if P2_MARK in p]
    assert len(p1) >= 2, "1단계 에이전트 프롬프트가 2건 미만"
    assert not any(SEQ_MARKER in p for p in p1), "1단계 프롬프트에 선행 발제 누수"
    assert any(SEQ_MARKER in p for p in p2), "2단계(순차)에 선행 의견 누락 — 회귀"


async def test_phase1_manual_copy_has_no_peer_posting(
    orchestrator, make_state, make_agent,
):
    """Manual 경로 — m1 발제 후에도 m2 복사본에 m1 발제가 새지 않는다."""
    state = make_state(
        discussion_id="man",
        agents=[make_agent("m1", "감마", provider=ModelProvider.MANUAL),
                make_agent("m2", "델타", provider=ModelProvider.MANUAL)])
    await database.insert_state(state)

    await orchestrator.process_event("man", PipelineEvent.START)
    mir = [m for _, m in orchestrator.broadcasts
           if m.type.value == "manual_input_required"]
    assert len(mir) == 2, "동시 단계 — 두 수동 에이전트가 모두 대기 진입해야 함"

    secret = "SECRET_감마의_1단계_발제_내용"
    orchestrator.broadcasts.clear()
    await orchestrator.process_event(
        "man", PipelineEvent.MANUAL_RESPONSE,
        {"agent_id": "m1", "phase": "phase_1_opinion", "content": secret})

    m2 = [m for _, m in orchestrator.broadcasts
          if m.type.value == "manual_input_required"
          and m.payload.get("agent_id") == "m2"]
    assert m2, "m1 발제 후 m2 복붙 패널이 재발행되지 않음"
    blob = ((m2[-1].payload.get("deep_copy") or "")
            + (m2[-1].payload.get("general_copy") or ""))
    assert secret not in blob, "m2 복사본에 m1 의 1단계 발제 누수"
    assert SEQ_MARKER not in blob, "m2 복사본에 '선행 의견' 섹션 누수"
