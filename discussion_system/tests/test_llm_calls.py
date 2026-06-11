"""LLM 호출 레이어 — reasoning 모델 파라미터 호환·스트리밍 추출.

OpenAI o-시리즈·gpt-5 계열은 max_tokens/temperature 를 거부하고, DeepSeek
reasoner 는 temperature 를 지원하지 않으며 사고 과정을 reasoning_content 로
보낸다 — 가짜 클라이언트로 실제 전송 kwargs 와 추출 결과를 검증한다.
"""
from types import SimpleNamespace

import pytest

from app.manager import _call_openai, _openai_sampling_kwargs


@pytest.mark.parametrize("model,expected", [
    # 일반 모델 — 종전 그대로 temperature + max_tokens.
    ("gpt-4o-mini", {"temperature": 0.7, "max_tokens": 1024}),
    ("gpt-4.1", {"temperature": 0.7, "max_tokens": 1024}),
    ("deepseek-chat", {"temperature": 0.7, "max_tokens": 1024}),
    # reasoning 모델 — max_completion_tokens, temperature 생략.
    ("gpt-5-mini", {"max_completion_tokens": 1024}),
    ("gpt-5", {"max_completion_tokens": 1024}),
    ("o3", {"max_completion_tokens": 1024}),
    ("o4-mini", {"max_completion_tokens": 1024}),
    ("o1-preview", {"max_completion_tokens": 1024}),
    # deepseek-reasoner — temperature 미지원, max_tokens 는 유지.
    ("deepseek-reasoner", {"max_tokens": 1024}),
])
def test_openai_sampling_kwargs_by_model_family(model, expected):
    assert _openai_sampling_kwargs(model, 0.7, 1024) == expected


def _chunk(content=None, reasoning=None, usage=None):
    """Chat Completions 스트리밍 청크 모형."""
    delta = SimpleNamespace(content=content)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    choices = [SimpleNamespace(delta=delta)] if (
        content is not None or reasoning is not None) else []
    u = (SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1])
         if usage else None)
    return SimpleNamespace(choices=choices, usage=u)


class _FakeOpenAI:
    """create() 에 전달된 kwargs 를 캡처하고 지정된 청크를 흘리는 가짜 클라이언트."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.captured = None
        outer = self

        async def create(**kwargs):
            outer.captured = kwargs

            async def gen():
                for c in outer._chunks:
                    yield c
            return gen()

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=create))


async def test_call_openai_reasoning_model_sends_adapted_params():
    fake = _FakeOpenAI([_chunk(content="답변."), _chunk(usage=(11, 22))])
    text, usage = await _call_openai(
        fake, "gpt-5-mini", "sys", "user", 0.7, 512, None)
    assert text == "답변."
    assert usage == {"prompt_tokens": 11, "completion_tokens": 22}
    sent = fake.captured
    assert sent["max_completion_tokens"] == 512
    assert "max_tokens" not in sent and "temperature" not in sent


async def test_call_openai_regular_model_keeps_legacy_params():
    fake = _FakeOpenAI([_chunk(content="ok")])
    await _call_openai(fake, "gpt-4o-mini", "sys", "user", 0.3, 256, None)
    sent = fake.captured
    assert sent["temperature"] == 0.3 and sent["max_tokens"] == 256
    assert "max_completion_tokens" not in sent


async def test_call_openai_streams_reasoning_but_excludes_from_text():
    """deepseek-reasoner — 사고 과정은 콜백으로만, 반환 발언에는 미포함."""
    fake = _FakeOpenAI([
        _chunk(reasoning="(생각 중…)"),
        _chunk(content="최종 발언."),
    ])
    seen: list[str] = []

    async def on_token(t):
        seen.append(t)

    text, _ = await _call_openai(
        fake, "deepseek-reasoner", "sys", "user", 0.7, 1024, on_token)
    assert text == "최종 발언."                  # 사고 과정은 발언이 아니다
    assert seen == ["(생각 중…)", "최종 발언."]   # 라이브 스트림에는 둘 다 보인다
    assert "temperature" not in fake.captured     # reasoner 는 temperature 미지원
