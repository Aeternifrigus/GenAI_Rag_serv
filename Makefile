.PHONY: install test lint run smoke deps-up deps-down docker-build clean

install:
	pip install -r requirements.txt

test:
	pytest -v

lint:
	ruff check app tests scripts

run:
	uvicorn app.main:app --reload --port 8080

smoke:
	python scripts/smoke.py

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
