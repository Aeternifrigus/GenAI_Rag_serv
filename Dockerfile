# Multi-stage: build deps once, ship only what the runtime needs.
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY app/ app/
COPY data/ data/

# Cloud Run injects $PORT; 8080 is the local default.
ENV PORT=8080
EXPOSE 8080

# Non-root, because the container has no reason to run privileged.
RUN useradd --create-home --uid 1001 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
