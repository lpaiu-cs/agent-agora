"""오케스트레이터 — 전원 manual 폴백, 세마포어 백프레셔, 크래시 복구."""
import asyncio
import json

from app import database
from app.manager import PipelineEvent
from app.schemas import DiscussionPhase, DiscussionStatus, ModelProvider


async def test_all_manual_discussion_summary_uses_fallback_model(
    orchestrator, make_state, make_agent, patch_llm,
):
    """전원 manual 토론도 폴백 모델(gpt-4o-mini)로 요약·합의 근접도를 계산한다."""
    calls: list[str] = []

    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        calls.append(model)
        return json.dumps({"agent_summaries": [], "key_conflicts": [],
                           "convergence_score": 0.42})

    patch_llm(fake)
    state = make_state(
        discussion_id="allman",
        agents=[make_agent("m1", "감마", provider=ModelProvider.MANUAL),
                make_agent("m2", "델타", provider=ModelProvider.MANUAL)])
    await database.insert_state(state)

    await orchestrator.process_event("allman", PipelineEvent.START)
    for aid, name in (("m1", "감마"), ("m2", "델타")):
        await orchestrator.process_event(
            "allman", PipelineEvent.MANUAL_RESPONSE,
            {"agent_id": aid, "phase": "phase_1_opinion",
             "content": f"{name}의 1단계 발제"})

    assert calls, "전원 manual 인데 요약 LLM 호출이 없었다 — 폴백 미작동"
    assert all(m == "gpt-4o-mini" for m in calls), f"폴백 모델이 아님: {calls}"
    st = await database.load_state("allman")
    summ = [p for p in st.phase_summaries
            if p.phase is DiscussionPhase.PHASE_1_OPINION]
    assert summ, "1단계 요약이 생성되지 않음"
    assert summ[0].convergence_score == 0.42


async def test_llm_semaphore_caps_concurrency(orchestrator, make_agent, patch_llm):
    """동시 LLM 호출 수가 세마포어 상한을 넘지 않는다."""
    orchestrator._llm_semaphore = asyncio.Semaphore(2)
    live = {"cur": 0, "peak": 0}

    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        live["cur"] += 1
        live["peak"] = max(live["peak"], live["cur"])
        await asyncio.sleep(0.02)
        live["cur"] -= 1
        return "ok"

    patch_llm(fake)
    agent = make_agent("a1", "알파")   # gpt-4o-mini → openai
    await asyncio.gather(
        *(orchestrator._invoke_agent(agent, "sys", "user") for _ in range(20)))
    assert live["peak"] == 2, f"세마포어 상한 위반 또는 미포화 — peak={live['peak']}"


async def test_recover_reruns_running_and_preserves_pending(
    orchestrator, make_state, make_agent, patch_llm,
):
    """크래시 복구 — RUNNING 은 멱등 재기동, PENDING_MANUAL_INPUT 은 그대로 보존."""
    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        return "복구 후 응답."

    patch_llm(fake)

    running = make_state(discussion_id="run")   # 기본 2인 API
    running.status = DiscussionStatus.RUNNING
    running.current_phase = DiscussionPhase.PHASE_1_OPINION
    await database.insert_state(running)

    pending = make_state(
        discussion_id="pend",
        agents=[make_agent("m1", "감마", provider=ModelProvider.MANUAL),
                make_agent("m2", "델타", provider=ModelProvider.MANUAL)])
    pending.status = DiscussionStatus.PENDING_MANUAL_INPUT
    pending.current_phase = DiscussionPhase.PHASE_1_OPINION
    await database.insert_state(pending)

    result = await orchestrator.recover()
    assert result == {"running_recovered": 1, "pending_preserved": 1}

    # recover() 는 trigger()로 백그라운드 재기동한다 — 완료까지 기다린다.
    inflight = list(orchestrator._inflight)
    if inflight:
        await asyncio.gather(*inflight)

    run_after = await database.load_state("run")
    assert run_after.status is not DiscussionStatus.RUNNING   # 재기동돼 진행됨
    pend_after = await database.load_state("pend")
    assert pend_after.status is DiscussionStatus.PENDING_MANUAL_INPUT   # 보존
