.PHONY: install check format lint typecheck test run clean help

help:
	@echo "Available targets:"
	@echo "  make install       - Install dependencies"
	@echo "  make check         - Run ruff, mypy, and pytest"
	@echo "  make format        - Format code with ruff"
	@echo "  make lint          - Lint code with ruff"
	@echo "  make typecheck     - Type check with mypy"
	@echo "  make test          - Run pytest"
	@echo "  make run           - Run the arXiv importer"
	@echo "  make clean         - Remove cache and build artifacts"

install:
	pip install -r requirements.txt
	pip install ruff mypy pytest pytest-cov

format:
	ruff format . --exclude downloads/,generated_models/

lint:
	ruff check --fix . --exclude downloads/,generated_models/

typecheck:
	mypy . --ignore-missing-imports --exclude generated_models/

test:
	pytest -v --tb=short

check: lint typecheck test
	@echo "All checks passed!"
	@echo "✅ All checks passed!"

run:
	python -m src.main

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".mypy_cache" -exec rm -rf {} + || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + || true
	rm -rf .coverage htmlcov/

.PHONY: install check format lint typecheck test run clean help
