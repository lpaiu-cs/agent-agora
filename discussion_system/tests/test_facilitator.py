"""사회자(facilitator) 에이전트 — 단계 경계 진행 조율.

사회자는 토론자가 아닌 부가 레이어다 — 지정 시 개회·중간·폐회 노트를 남기고,
지정하지 않으면 동작은 완전히 종전과 같다. 호출 실패도 토론을 막지 않는다.
"""
from app import database
from app.manager import PipelineEvent, render_transcript
from app.schemas import DiscussionStatus, FacilitatorNote


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
