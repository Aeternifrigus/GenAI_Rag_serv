.PHONY: install install-training test lint run smoke train deps-up deps-down docker-build clean

install:
	pip install -r requirements.txt

install-training:
	pip install -r requirements.txt -r requirements-training.txt

test:
	pytest -v

lint:
	ruff check app tests scripts

run:
	uvicorn app.main:app --reload --port 8080

smoke:
	python scripts/smoke.py

# Trains the reranker and measures it against fusion-only on the held-out
# questions. Exits non-zero when the candidate fails the promotion gate.
train:
	python scripts/train_reranker.py

# optional backing services
deps-up:
	docker compose up -d

deps-down:
	docker compose down

docker-build:
	docker build -t genai-rag-service .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
