# Agent Agora — 멀티 스테이지 프로덕션 이미지
# 빌드 컨텍스트: 저장소 루트 (`docker build .`)

# ---- 1) builder: 의존성을 가상환경에 설치 ----------------------------------
FROM python:3.12-slim AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# requirements 만 먼저 복사 — 코드 변경이 의존성 레이어 캐시를 무효화하지 않게.
COPY discussion_system/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---- 2) runtime: 가상환경 + 앱 코드만 (빌드 도구·캐시 제외) -----------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# 비루트 사용자로 구동 (프로덕션 보안 관행).
RUN useradd --create-home --uid 10001 agora
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=agora:agora discussion_system/app ./app

USER agora
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
