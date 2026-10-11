"""Browser-side A2 parser regression tests.

The transcript parser lives in the inline browser script, so execute that exact
source with Node instead of maintaining a divergent Python reimplementation.
Node.js 가 없는 환경에서는 skip 된다 (requirements-dev.txt 참고).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app.manager import render_transcript_with_meta
from app.schemas import (
    AgentConfig,
    AgentTurn,
    DiscussionState,
    FacilitatorNote,
    ModelProvider,
    PhaseSummary,
    UserIntervention,
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="Node.js(node) 미설치 — 브라우저 A2 파서 회귀 테스트는 node 가 필요하다",
)

ROOT = Path(__file__).parents[1]
HTML = ROOT / "app" / "templates" / "index.html"


def _parse_in_browser_parser(doc: str) -> dict:
    html = HTML.read_text(encoding="utf-8")
    start = html.index("const PERSONA_TABLE")
    end = html.index("// ---- 저장된 .md 토론을 라이브 UI 구조로 복원")
    program = f"""
const vm = require('vm');
const context = {{ console, atob }};
vm.createContext(context);
vm.runInContext({json.dumps(html[start:end])}, context);
const md = {json.dumps(doc)};
const meta = context.decodeMetaA2(md);
if (!meta || meta.version !== 5) throw new Error('A2 v5 hint missing');
process.stdout.write(JSON.stringify(context.parseWithBlockLengths(md, meta)));
"""
    result = subprocess.run(
        ["node"], input=program, cwd=ROOT, encoding="utf-8", capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _agent(agent_id, name):
    return AgentConfig(agent_id=agent_id, name=name, model="m",
                       persona_prompt="p", provider=ModelProvider.MANUAL)


def test_v5_parser_keeps_markdown_inside_every_free_text_block():
    # 모든 자유 텍스트에 구조 마커처럼 보이는 줄(`## `, `### <로스터 이름>`)을 넣는다.
    trap = "\n## 세부 근거\n### B\n"
    long_turn = "## 본문 안의 헤더\n### B\n끝부분\n" + "x" * 9_000
    state = DiscussionState(
        discussion_id="d",
        topic="주제" + trap + "## 참여 에이전트\n- **B** (`x`) — fake",
        agents=[_agent("a0", "A"), _agent("a1", "B")],
        facilitator=_agent("f", "사회자봇"),
        reference_materials="근거 자료" + trap + "끝",
        user_interventions=[
            UserIntervention(message="사전 지시" + trap),
            UserIntervention(message="A에게만" + trap, after_phase="opinion",
                             target_agent_id="a0"),
        ],
        facilitator_notes=[
            FacilitatorNote(phase="opinion", kind="open", content="개회" + trap),
            FacilitatorNote(phase="opinion", kind="between", content="조율" + trap),
            FacilitatorNote(phase="opinion", kind="close", content="폐회" + trap),
        ],
        final_joint_agreement="합의" + trap + "- 항목",
    )
    state.phase_records = {"opinion": [
        AgentTurn(agent_id="a0", phase="opinion", content=long_turn),
        AgentTurn(agent_id="a1", phase="opinion", content="둘째 발언" + trap),
    ]}
    state.phase_summaries = [PhaseSummary(
        phase="opinion", convergence_score=0.7, key_conflicts=["쟁점_1", "쟁점\n## 2"])]

    parsed = _parse_in_browser_parser(render_transcript_with_meta(state))

    assert parsed["topic"] == state.topic
    assert [a["name"] for a in parsed["agents"]] == ["A", "B"]
    assert parsed["facilitator"]["name"] == "사회자봇"
    assert parsed["referenceMaterials"] == state.reference_materials
    assert parsed["preInstructions"] == ["사전 지시" + trap]
    assert parsed["openNote"] == "개회" + trap
    assert parsed["closeNote"] == "폐회" + trap
    assert parsed["agreement"] == state.final_joint_agreement
    assert len(parsed["phases"]) == 1
    phase = parsed["phases"][0]
    assert [(t["speaker"], t["content"]) for t in phase["turns"]] == [
        ("A", long_turn), ("B", "둘째 발언" + trap)]
    assert phase["summary"] == {"pct": 70, "conflicts": "쟁점_1 · 쟁점\n## 2"}
    assert phase["interventions"] == [{"target": "A", "message": "A에게만" + trap}]
    assert phase["notes"] == [{"kind": "중간 조율", "content": "조율" + trap}]
