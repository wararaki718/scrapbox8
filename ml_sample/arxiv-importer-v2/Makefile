.PHONY: install lint typecheck test check run

# ==============================================================================
# Dependency Management
# ==============================================================================
install:
	@echo "Installing dependencies from requirements.txt..."
	pip install --upgrade pip
	pip install -r requirements.txt

# ==============================================================================
# Quality Assurance
# ==============================================================================
lint:
	@echo "Running linter (ruff)..."
	ruff check .

typecheck:
	@echo "Running type checker (mypy)..."
	mypy .

test:
	@echo "Running tests (pytest)..."
	pytest

check: lint typecheck test
	@echo "All checks passed!"

# ==============================================================================
# Execution
# ==============================================================================
run:
	@echo "Running arXiv importer..."
	python -m src.main

# ==============================================================================
# Help
# =================================e==============================================
help:
	@echo "Available commands:"
	@echo "  make install    - Install project dependencies"
	@echo "  make lint       - Run ruff linter"
	@echo "  make typecheck  - Run mypy type checker"
	@echo "  make test       - Run pytest tests"
	@echo "  make check      - Run all quality checks (lint, typecheck, test)"
	@echo "  make run        - Run the main application"
