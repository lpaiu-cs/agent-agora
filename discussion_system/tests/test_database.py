"""database.py — 영속성 라운드트립 + 낙관적 락(StaleStateError)."""
import pytest

from app import database
from app.database import StaleStateError
from app.schemas import DiscussionStatus


async def test_insert_and_load(fresh_db, make_state):
    await database.insert_state(make_state(discussion_id="d1", topic="저장 테스트"))
    loaded = await database.load_state("d1")
    assert loaded is not None
    assert loaded.discussion_id == "d1"
    assert loaded.topic == "저장 테스트"
    assert loaded.version == 0


async def test_load_missing_returns_none(fresh_db):
    assert await database.load_state("does-not-exist") is None


async def test_update_bumps_version(fresh_db, make_state):
    state = make_state(discussion_id="d1")
    await database.insert_state(state)
    state.topic = "수정된 주제"
    await database.update_state(state)
    assert state.version == 1
    loaded = await database.load_state("d1")
    assert loaded.topic == "수정된 주제"
    assert loaded.version == 1


async def test_optimistic_lock_rejects_stale_write(fresh_db, make_state):
    await database.insert_state(make_state(discussion_id="d1"))
    # 두 워커가 같은 버전(0)을 로드한다.
    first = await database.load_state("d1")
    second = await database.load_state("d1")
    # 먼저 커밋한 쪽은 성공.
    first.topic = "A 가 먼저 수정"
    await database.update_state(first)
    # 뒤늦은 쪽은 버전이 어긋나 거부된다.
    second.topic = "B 의 늦은 수정"
    with pytest.raises(StaleStateError):
        await database.update_state(second)
    # 최종 상태는 먼저 커밋한 쪽.
    loaded = await database.load_state("d1")
    assert loaded.topic == "A 가 먼저 수정"
    assert loaded.version == 1


async def test_list_states_by_status(fresh_db, make_state):
    await database.insert_state(
        make_state(discussion_id="r", status=DiscussionStatus.RUNNING))
    await database.insert_state(
        make_state(discussion_id="p", status=DiscussionStatus.PENDING_MANUAL_INPUT))
    await database.insert_state(
        make_state(discussion_id="c", status=DiscussionStatus.COMPLETED))

    rows = await database.list_states_by_status(("running", "pending_manual_input"))
    assert {s.discussion_id for s in rows} == {"r", "p"}
