.PHONY: install check format lint typecheck test run clean help

help:
	@echo "Available targets:"
	@echo "  make install       - Install dependencies"
	@echo "  make check         - Run ruff, mypy, and pytest"
	@echo "  make format        - Format code with ruff"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make typecheck     - Type check with mypy"
	@echo "  make test          - Run pytest"
	@echo "  make run           - Run the investment meeting simulator"
	@echo "  make clean         - Remove cache and build artifacts"

install:
	pip install -r requirements.txt

format:
	ruff format src/ tests/

lint:
	ruff check --fix src/ tests/

typecheck:
	mypy src/

test:
	pytest tests/ -v --cov=src

check: lint typecheck test
	@echo "✅ All checks passed!"

run:
	python -m src.main

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov/ dist/ build/
