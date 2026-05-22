"""사회자(facilitator) 에이전트 — 단계 경계 진행 조율.

사회자는 토론자가 아닌 부가 레이어다 — 지정 시 개회·중간·폐회 노트를 남기고,
지정하지 않으면 동작은 완전히 종전과 같다. 호출 실패도 토론을 막지 않는다.
가변 길이 형식에서는 사회자의 'decision' 훅이 문답 라운드 루프를 구동한다.
"""
import json

from app import database
from app.manager import PipelineEvent, render_transcript
from app.schemas import DiscussionStatus, FacilitatorNote


async def _drive_to_completion(orchestrator, did, limit=16):
    """START 후 게이트마다 ADVANCE 해 토론을 끝까지 진행한다."""
    await orchestrator.process_event(did, PipelineEvent.START)
    for _ in range(limit):
        st = await database.load_state(did)
        if st.status is not DiscussionStatus.WAITING_FOR_USER:
            break
        await orchestrator.process_event(did, PipelineEvent.ADVANCE)
    return await database.load_state(did)


async def test_facilitator_notes_at_phase_boundaries(
    orchestrator, make_state, make_agent, patch_llm,
):
    """사회자 — 개회 1회·중간 조율(단계 사이)·폐회 1회 노트가 생성된다."""
    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        if "사회자 작업" in user:   # _build_facilitator_prompt 의 머리말
            return "사회자의 진행 코멘트."
        return "에이전트 발언."

    patch_llm(fake)
    await database.insert_state(make_state(
        discussion_id="fac", facilitator=make_agent("f1", "사회자")))

    await orchestrator.process_event("fac", PipelineEvent.START)
    for _ in range(8):   # debate 5단계 — 넉넉한 상한
        st = await database.load_state("fac")
        if st.status is not DiscussionStatus.WAITING_FOR_USER:
            break
        await orchestrator.process_event("fac", PipelineEvent.ADVANCE)

    st = await database.load_state("fac")
    assert st.status is DiscussionStatus.COMPLETED
    kinds = [n.kind for n in st.facilitator_notes]
    assert kinds.count("open") == 1
    assert kinds.count("close") == 1
    assert kinds.count("between") == 4   # debate 5단계 — 단계 사이 4회


async def test_facilitator_failure_does_not_block_discussion(
    orchestrator, make_state, make_agent, patch_llm,
):
    """사회자 LLM 호출이 실패해도 토론은 끝까지 진행된다 (비치명적 레이어)."""
    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        if "사회자 작업" in user:
            raise RuntimeError("사회자 모델 장애")
        return "에이전트 발언."

    patch_llm(fake)
    await database.insert_state(make_state(
        discussion_id="facfail", facilitator=make_agent("f1", "사회자")))

    await orchestrator.process_event("facfail", PipelineEvent.START)
    for _ in range(8):
        st = await database.load_state("facfail")
        if st.status is not DiscussionStatus.WAITING_FOR_USER:
            break
        await orchestrator.process_event("facfail", PipelineEvent.ADVANCE)

    st = await database.load_state("facfail")
    assert st.status is DiscussionStatus.COMPLETED   # 사회자 실패에도 정상 종료
    assert st.facilitator_notes == []                # 실패한 노트는 적재 안 됨


async def test_no_facilitator_yields_no_notes(
    orchestrator, make_state, patch_llm,
):
    """사회자 미지정 — facilitator_notes 는 비어 있다 (완전 opt-in)."""
    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        return "발언."

    patch_llm(fake)
    await database.insert_state(make_state(discussion_id="nofac"))

    await orchestrator.process_event("nofac", PipelineEvent.START)
    st = await database.load_state("nofac")
    assert st.facilitator_notes == []


def test_render_transcript_includes_facilitator_notes(make_state, make_agent):
    """기록 문서(내보내기)에 사회자의 개회·폐회 노트와 헤더 표기가 들어간다."""
    state = make_state(facilitator=make_agent("f1", "진행자"))
    state.facilitator_notes = [
        FacilitatorNote(phase="opinion", kind="open", content="개회사 본문"),
        FacilitatorNote(phase="completed", kind="close", content="폐회사 본문"),
    ]
    doc = render_transcript(state)
    assert "사회자: **진행자**" in doc        # 헤더에 사회자 표기
    assert "사회자 — 개회" in doc and "개회사 본문" in doc
    assert "사회자 — 폐회" in doc and "폐회사 본문" in doc


# ---------------------------------------------------------------------------
# 증분 C — 사회자 'decision' 훅이 가변 길이(반복 단계) 루프를 구동한다
# ---------------------------------------------------------------------------
def _decisive_fake(decision: str, convergence: float):
    """문답 라운드마다 사회자가 ``decision`` 을 내리는 가짜 LLM 을 만든다."""
    async def fake(client, model, system, user, temperature, max_tokens, on_token):
        if "한 번 더 진행할지" in user:        # 사회자 decision 훅
            return f"[결정] {decision}\n사회자의 판단 근거."
        if "사회자 작업" in user:               # 사회자 open/between/close
            return "사회자 코멘트."
        if "convergence_score" in user:         # 단계 요약(JSON 요청)
            return json.dumps({"agent_summaries": [], "key_conflicts": [],
                               "convergence_score": convergence})
        return "에이전트 발언."
    return fake


async def test_facilitator_continue_overrides_high_convergence(
    orchestrator, make_state, make_agent, patch_llm,
):
    """사회자 'continue' — 합의 근접도가 높아도 사회자가 라운드를 더 끌고 간다."""
    # 근접도 0.99 면 숫자 예측자는 2라운드에서 멈추지만, 사회자가 우선한다.
    patch_llm(_decisive_fake("continue", 0.99))
    await database.insert_state(make_state(
        discussion_id="fc", format_id="socratic",
        facilitator=make_agent("f1", "사회자")))

    st = await _drive_to_completion(orchestrator, "fc")
    assert st.status is DiscussionStatus.COMPLETED
    probe_rounds = {k for k in st.phase_records if k.startswith("probe#")}
    assert probe_rounds == {f"probe#{n}" for n in range(1, 7)}   # max_rounds 까지
    assert "synthesis" in st.phase_records


async def test_facilitator_conclude_ends_discussion_early(
    orchestrator, make_state, make_agent, patch_llm,
):
    """사회자 'conclude' — 문답 라운드 도중 사회자가 토론을 조기 종료한다."""
    patch_llm(_decisive_fake("conclude", 0.3))
    await database.insert_state(make_state(
        discussion_id="fc2", format_id="socratic",
        facilitator=make_agent("f1", "사회자")))

    st = await _drive_to_completion(orchestrator, "fc2")
    assert st.status is DiscussionStatus.COMPLETED
    # min_rounds=2 는 지키고, 그 후 첫 게이트(probe#2)에서 conclude → 종합 없이 종료.
    assert "synthesis" not in st.phase_records
    assert {k for k in st.phase_records if k.startswith("probe#")} == {
        "probe#1", "probe#2"}


async def test_facilitator_next_moves_to_following_phase(
    orchestrator, make_state, make_agent, patch_llm,
):
    """사회자 'next' — min_rounds 이후 사회자가 다음 단계로 넘긴다."""
    patch_llm(_decisive_fake("next", 0.3))
    await database.insert_state(make_state(
        discussion_id="fc3", format_id="socratic",
        facilitator=make_agent("f1", "사회자")))

    st = await _drive_to_completion(orchestrator, "fc3")
    assert st.status is DiscussionStatus.COMPLETED
    # 근접도 0.3 이면 숫자 예측자는 라운드를 계속하지만, 사회자 next 가 종합으로.
    assert set(st.phase_records) == {
        "position", "probe#1", "probe#2", "synthesis"}
    # decision 노트가 게이트마다 남는다.
    decisions = [n for n in st.facilitator_notes if n.kind == "decision"]
    assert {n.phase for n in decisions} == {"probe#1", "probe#2"}
    assert all(n.decision == "next" for n in decisions)
