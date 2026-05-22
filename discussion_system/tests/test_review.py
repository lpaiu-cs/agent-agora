"""선택적 가로채기 검토 게이트 — 초안·사고흐름 검토, 문답 개입, 승인 흐름."""
from app import database
from app.manager import PipelineEvent
from app.schemas import DiscussionStatus


async def _fake(client, model, system, user, temperature, max_tokens, on_token):
    """프롬프트 종류별로 다른 응답을 돌려주는 가짜 LLM."""
    if "[검토 모드]" in user:
        return "[사고흐름]\n나는 이렇게 판단했다.\n[초안]\nDRAFT_초안_발언"
    if "[진행자와의 문답]" in user:
        return "FINAL_문답_반영_최종발언"
    if "[진행자 질문]" in user:
        return "질문에 대한 답변입니다."
    return "일반 발언."


async def test_intercept_review_question_approve_flow(
    orchestrator, make_state, make_agent, patch_llm,
):
    """가로채기 → 초안·사고흐름 검토 → 문답 → 승인 → 최종 포스팅 전 과정."""
    patch_llm(_fake)
    state = make_state(
        discussion_id="rv",
        agents=[make_agent("a1", "알파"), make_agent("a2", "베타")])
    await database.insert_state(state)

    await orchestrator.set_intercepts("rv", ["a1"])   # a1 만 가로채기
    await orchestrator.process_event("rv", PipelineEvent.START)

    st = await database.load_state("rv")
    assert st.status is DiscussionStatus.PENDING_REVIEW
    assert st.review is not None and st.review.agent_id == "a1"
    assert st.review.draft == "DRAFT_초안_발언"
    assert st.review.reasoning             # 사고흐름이 채워짐
    # 가로채기 안 된 a2 는 이미 자동 포스팅됨
    assert any(t.agent_id == "a2" for t in st.phase_records.get("opinion", []))
    assert any(m.type.value == "review_required"
               for _, m in orchestrator.broadcasts)

    # 문답 개입
    await orchestrator.process_event(
        "rv", PipelineEvent.REVIEW_QUESTION, {"question": "왜 그렇게 보나?"})
    st = await database.load_state("rv")
    assert len(st.review.qa) == 1
    assert st.review.qa[0].question == "왜 그렇게 보나?"
    assert st.review.qa[0].answer == "질문에 대한 답변입니다."

    # 승인 → 문답 반영 최종 발언 확정
    await orchestrator.process_event("rv", PipelineEvent.REVIEW_APPROVE)
    st = await database.load_state("rv")
    assert st.review is None
    a1_turns = [t for t in st.phase_records["opinion"] if t.agent_id == "a1"]
    assert a1_turns and a1_turns[0].content == "FINAL_문답_반영_최종발언"
    assert a1_turns[0].metadata.get("reviewed") is True
    # 두 에이전트 모두 포스팅 → 1단계 종료, 게이트 락
    assert st.status is DiscussionStatus.WAITING_FOR_USER


async def test_review_approve_without_questions_uses_draft(
    orchestrator, make_state, make_agent, patch_llm,
):
    """문답 없이 곧바로 승인하면 초안이 그대로 최종 발언이 된다."""
    patch_llm(_fake)
    state = make_state(
        discussion_id="rv2",
        agents=[make_agent("a1", "알파"), make_agent("a2", "베타")])
    await database.insert_state(state)
    await orchestrator.set_intercepts("rv2", ["a1"])
    await orchestrator.process_event("rv2", PipelineEvent.START)

    await orchestrator.process_event("rv2", PipelineEvent.REVIEW_APPROVE)
    st = await database.load_state("rv2")
    a1 = [t for t in st.phase_records["opinion"] if t.agent_id == "a1"][0]
    assert a1.content == "DRAFT_초안_발언"


async def test_non_intercepted_discussion_runs_without_review(
    orchestrator, make_state, patch_llm,
):
    """가로채기 미지정 — 검토 게이트 없이 현행대로 자동 진행한다."""
    patch_llm(_fake)
    await database.insert_state(make_state(discussion_id="auto"))
    await orchestrator.process_event("auto", PipelineEvent.START)
    st = await database.load_state("auto")
    assert st.status is not DiscussionStatus.PENDING_REVIEW
    assert st.review is None


async def test_intercept_in_sequential_phase(
    orchestrator, make_state, make_agent, patch_llm,
):
    """순차 단계(2단계 상호비판)에서의 가로채기 — 순차 분기 검토 진입 경로.

    debate 2단계는 순차 단계라 _advance_phase_progress 의 순차 분기를 탄다.
    선행 에이전트는 자동 포스팅되고, 가로채기된 후순위 에이전트만 검토로 진입한다.
    """
    patch_llm(_fake)
    state = make_state(
        discussion_id="seq",
        agents=[make_agent("a1", "알파"), make_agent("a2", "베타")])
    await database.insert_state(state)

    # 가로채기 없이 1단계(동시)를 끝낸다
    await orchestrator.process_event("seq", PipelineEvent.START)
    st = await database.load_state("seq")
    assert st.status is DiscussionStatus.WAITING_FOR_USER

    # 2단계 진입 전 a2 만 가로채기로 지정
    await orchestrator.set_intercepts("seq", ["a2"])
    await orchestrator.process_event("seq", PipelineEvent.ADVANCE)

    # 2단계(critique, 순차): a1 자동 포스팅 → a2 는 순차 분기로 검토 진입
    st = await database.load_state("seq")
    assert st.status is DiscussionStatus.PENDING_REVIEW
    assert st.review is not None
    assert st.review.agent_id == "a2"
    assert st.review.phase == "critique"
    assert any(t.agent_id == "a1"
               for t in st.phase_records.get("critique", []))

    # 승인 → a2 최종 발언 확정, 2단계 종료
    await orchestrator.process_event("seq", PipelineEvent.REVIEW_APPROVE)
    st = await database.load_state("seq")
    assert st.review is None
    assert any(t.agent_id == "a2" for t in st.phase_records["critique"])
    assert st.status is DiscussionStatus.WAITING_FOR_USER
