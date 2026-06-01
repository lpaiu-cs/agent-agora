"""Agent Agora — 다중 에이전트 토론 시스템.

5단계 동시 턴 파이프라인 기반 LLM 에이전트 토론 오케스트레이터.
"""

# .env 를 가장 먼저 로드한다 — database 등 하위 모듈이 임포트 시점에 환경
# 변수를 읽어가므로, 그 전에 적재해야 한다. python-dotenv 미설치 시(개발 환경에
# 따라)에도 동작하도록 ImportError 를 흡수한다 — 실제 환경 변수 그대로 사용.
# 기존 환경 변수는 덮어쓰지 않는다(override=False, 기본) — 테스트 conftest.py
# 가 setdefault 로 미리 박아둔 더미 키가 .env 로 교란되지 않게.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

__version__ = "0.6.0"
