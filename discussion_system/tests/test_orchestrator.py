"""오케스트레이터 — 폴백 모델, 세마포어 백프레셔, 크래시 복구, 형식 진행."""
import asyncio
import json

from app import database
from app.manager import PipelineEvent
from app.schemas import DiscussionStatus, ModelProvider


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
            {"agent_id": aid, "phase": "opinion",
             "content": f"{name}의 발제"})

    assert calls, "전원 manual 인데 요약 LLM 호출이 없었다 — 폴백 미작동"
    assert all(m == "gpt-4o-mini" for m in calls), f"폴백 모델이 아님: {calls}"
    st = await database.load_state("allman")
    summ = [p for p in st.phase_summaries if p.phase == "opinion"]
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

    running = make_state(discussion_id="run")   # 기본 2인 API · debate
    running.status = DiscussionStatus.RUNNING
    running.current_phase = "opinion"
    await database.insert_state(running)

    pending = make_state(
        discussion_id="pend",
        agents=[make_agent("m1", "감마", provider=ModelProvider.MANUAL),
                make_agent("m2", "델타", provider=ModelProvider.MANUAL)])
    pending.status = DiscussionStatus.PENDING_MANUAL_INPUT
    pending.current_phase = "opinion"
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


async def test_token_usage_recorded_in_turn_metadata(
    orchestrator, make_state, patch_llm,
):
    """LLM 호출의 토큰 사용량이 발언 메타데이터에 기록되고 기록 문서에 합산된다."""
    from app.manager import render_transcript

    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        return ("테스트 발언.", {"prompt_tokens": 10, "completion_tokens": 20})

    patch_llm(fake)
    await database.insert_state(make_state(discussion_id="usage"))
    await orchestrator.process_event("usage", PipelineEvent.START)

    st = await database.load_state("usage")
    turns = st.phase_records.get("opinion", [])
    assert turns
    assert turns[0].metadata.get("usage") == {
        "prompt_tokens": 10, "completion_tokens": 20}
    # 기록 문서(내보내기)에 토큰 합계가 들어간다
    assert "토큰" in render_transcript(st)


async def test_brainstorm_format_runs_its_four_phases(
    orchestrator, make_state, patch_llm,
):
    """brainstorm 형식 — 발산·확장·수렴·실행 4단계를 끝까지 진행한다."""
    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        return "브레인스토밍 발언."

    patch_llm(fake)
    await database.insert_state(
        make_state(discussion_id="bs", format_id="brainstorm"))

    await orchestrator.process_event("bs", PipelineEvent.START)
    for _ in range(6):   # 단계 수보다 넉넉한 상한
        st = await database.load_state("bs")
        if st.status is not DiscussionStatus.WAITING_FOR_USER:
            break
        await orchestrator.process_event("bs", PipelineEvent.ADVANCE)

    st = await database.load_state("bs")
    assert st.status is DiscussionStatus.COMPLETED
    assert st.current_phase == "completed"
    assert set(st.phase_records) == {"diverge", "expand", "converge", "action"}


async def test_socratic_probe_stops_at_min_rounds_when_converged(
    orchestrator, make_state, patch_llm,
):
    """socratic 가변 길이 — 합의 근접도가 높으면 문답 라운드가 최소 라운드에서 멈춘다."""
    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        # 요약 호출이 JSON 을 파싱할 수 있도록 — 합의 근접도 0.95(>= 0.8 임계).
        return json.dumps({"agent_summaries": [], "key_conflicts": [],
                           "convergence_score": 0.95})

    patch_llm(fake)
    await database.insert_state(
        make_state(discussion_id="soc", format_id="socratic"))

    await orchestrator.process_event("soc", PipelineEvent.START)
    for _ in range(12):   # 라운드 수보다 넉넉한 상한
        st = await database.load_state("soc")
        if st.status is not DiscussionStatus.WAITING_FOR_USER:
            break
        await orchestrator.process_event("soc", PipelineEvent.ADVANCE)

    st = await database.load_state("soc")
    assert st.status is DiscussionStatus.COMPLETED
    assert st.current_phase == "completed"
    # 근접도 0.95 ≥ 0.8 이지만 min_rounds=2 라 probe 는 2라운드까지만.
    assert set(st.phase_records) == {
        "position", "probe#1", "probe#2", "synthesis"}


async def test_socratic_probe_caps_at_max_rounds_when_not_converging(
    orchestrator, make_state, patch_llm,
):
    """socratic 가변 길이 — 합의가 안 되면 문답 라운드가 max_rounds 상한에서 멈춘다."""
    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        return json.dumps({"agent_summaries": [], "key_conflicts": [],
                           "convergence_score": 0.1})

    patch_llm(fake)
    await database.insert_state(
        make_state(discussion_id="soc2", format_id="socratic"))

    await orchestrator.process_event("soc2", PipelineEvent.START)
    for _ in range(15):
        st = await database.load_state("soc2")
        if st.status is not DiscussionStatus.WAITING_FOR_USER:
            break
        await orchestrator.process_event("soc2", PipelineEvent.ADVANCE)

    st = await database.load_state("soc2")
    assert st.status is DiscussionStatus.COMPLETED
    # 근접도 0.1 — 6라운드(max_rounds)까지 반복 후 종합 단계로.
    probe_rounds = {k for k in st.phase_records if k.startswith("probe#")}
    assert probe_rounds == {f"probe#{n}" for n in range(1, 7)}
    assert {"position", "synthesis"} <= set(st.phase_records)


async def test_gate_payload_reports_token_budget_state(
    orchestrator, make_state, patch_llm,
):
    """AWAITING_USER 페이로드 — 누적 사용량·예산·초과 여부를 싣는다 (소프트 상한)."""
    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        return ("발언.", {"prompt_tokens": 400, "completion_tokens": 200})

    patch_llm(fake)
    state = make_state(discussion_id="bud", token_budget=1000)
    await database.insert_state(state)
    await orchestrator.process_event("bud", PipelineEvent.START)

    gates = [m for _, m in orchestrator.broadcasts
             if m.type.value == "awaiting_user"]
    assert gates, "게이트 메시지가 없다"
    p = gates[-1].payload
    # 에이전트 2명 × (400+200) = 1200 ≥ 예산 1000 → 초과.
    assert p["tokens_used"] == 1200
    assert p["token_budget"] == 1000
    assert p["budget_exceeded"] is True
    # 예산 미지정 토론은 초과 플래그가 항상 False.
    await database.insert_state(make_state(discussion_id="nobud"))
    orchestrator.broadcasts.clear()
    await orchestrator.process_event("nobud", PipelineEvent.START)
    p2 = [m for _, m in orchestrator.broadcasts
          if m.type.value == "awaiting_user"][-1].payload
    assert p2["token_budget"] is None
    assert p2["budget_exceeded"] is False


async def test_end_event_completes_discussion_at_gate(
    orchestrator, make_state, patch_llm,
):
    """END — WAITING_FOR_USER 게이트에서 조기 종료하면 남은 단계는 진행하지 않는다."""
    from app.manager import InvalidStateTransition

    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        return "발언."

    patch_llm(fake)
    await database.insert_state(make_state(discussion_id="early"))

    await orchestrator.process_event("early", PipelineEvent.START)
    gate = await database.load_state("early")
    assert gate.status is DiscussionStatus.WAITING_FOR_USER   # 1단계 후 게이트

    await orchestrator.process_event("early", PipelineEvent.END)

    st = await database.load_state("early")
    assert st.status is DiscussionStatus.COMPLETED
    assert st.current_phase == "completed"
    assert set(st.phase_records) == {"opinion"}   # 남은 단계는 실행되지 않음

    # 이미 종료된 토론에 END 재요청 → 거부
    try:
        await orchestrator.process_event("early", PipelineEvent.END)
        raise AssertionError("종료된 토론에 END 가 거부되지 않음")
    except InvalidStateTransition:
        pass
