"""REST API 엔드포인트 — 상태 코드·검증·라우팅 (백그라운드 파이프라인 비활성).

client 픽스처가 Orchestrator.trigger 를 no-op 으로 막으므로, 생성된 토론은
created 상태에 머문다 — HTTP 계층을 결정론적으로 검증할 수 있다.
"""


def _manual_agents():
    return [
        {"agent_id": "m1", "name": "감마", "model": "manual-x",
         "persona_prompt": "감마 페르소나", "provider": "manual"},
        {"agent_id": "m2", "name": "델타", "model": "manual-y",
         "persona_prompt": "델타 페르소나", "provider": "manual"},
    ]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_discussion(client):
    r = client.post("/discussions", json={
        "topic": "API 테스트 주제", "agents": _manual_agents(),
        "force_consensus": False})
    assert r.status_code == 201
    body = r.json()
    assert body["discussion_id"]
    assert body["status"] == "created"


def test_create_discussion_rejects_single_agent(client):
    r = client.post("/discussions", json={
        "topic": "t", "agents": _manual_agents()[:1]})
    assert r.status_code == 422


def test_get_discussion_roundtrip(client):
    did = client.post("/discussions", json={
        "topic": "조회 주제", "agents": _manual_agents()}).json()["discussion_id"]
    r = client.get(f"/discussions/{did}")
    assert r.status_code == 200
    assert r.json()["topic"] == "조회 주제"


def test_get_unknown_discussion_returns_404(client):
    assert client.get("/discussions/does-not-exist").status_code == 404


def test_advance_rejects_non_waiting_state(client):
    did = client.post("/discussions", json={
        "topic": "t", "agents": _manual_agents()}).json()["discussion_id"]
    # 갓 생성된 토론은 created 상태 — advance 는 waiting_for_user 에서만 가능.
    assert client.post(f"/discussions/{did}/advance").status_code == 409


def test_manual_response_rejects_non_pending_state(client):
    did = client.post("/discussions", json={
        "topic": "t", "agents": _manual_agents()}).json()["discussion_id"]
    r = client.post(f"/discussions/{did}/manual-response", json={
        "agent_id": "m1", "phase": "opinion", "content": "응답"})
    assert r.status_code == 409


def test_refine_persona_rejects_manual_provider(client):
    r = client.post("/personas/refine", json={
        "topic": "주제", "draft": "대강 쓴 초안",
        "provider": "manual", "model": "manual-x"})
    assert r.status_code == 400


def test_refine_persona_validates_required_fields(client):
    r = client.post("/personas/refine", json={"topic": "주제만 있음"})
    assert r.status_code == 422


def test_list_formats(client):
    r = client.get("/formats")
    assert r.status_code == 200
    formats = {f["id"]: f for f in r.json()["formats"]}
    assert {"debate", "brainstorm"} <= set(formats)
    assert [p["id"] for p in formats["brainstorm"]["phases"]] == [
        "diverge", "expand", "converge", "action"]
    assert formats["debate"]["supports_consensus"] is True


def test_export_discussion(client):
    did = client.post("/discussions", json={
        "topic": "내보내기 주제", "agents": _manual_agents()}).json()["discussion_id"]
    r = client.get(f"/discussions/{did}/export")
    assert r.status_code == 200
    assert "text/markdown" in r.headers["content-type"]
    assert "attachment" in r.headers.get("content-disposition", "")
    assert "내보내기 주제" in r.text


def test_export_unknown_discussion_returns_404(client):
    assert client.get("/discussions/does-not-exist/export").status_code == 404


def test_set_intercepts(client):
    did = client.post("/discussions", json={
        "topic": "t", "agents": _manual_agents()}).json()["discussion_id"]
    r = client.post(f"/discussions/{did}/intercept", json={"agent_ids": ["m1"]})
    assert r.status_code == 200
    assert r.json()["agent_ids"] == ["m1"]


def test_review_endpoints_reject_non_review_state(client):
    did = client.post("/discussions", json={
        "topic": "t", "agents": _manual_agents()}).json()["discussion_id"]
    # 갓 생성된 토론은 PENDING_REVIEW 가 아니므로 409
    assert client.post(f"/discussions/{did}/review/question",
                       json={"question": "q"}).status_code == 409
    assert client.post(f"/discussions/{did}/review/approve").status_code == 409
