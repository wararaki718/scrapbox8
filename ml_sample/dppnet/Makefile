.PHONY: install lint typecheck test check run

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt

lint:
	@echo "Running ruff..."
	ruff check .

typecheck:
	@echo "Running mypy..."
	PYTHONPATH=src mypy src tests

test:
	@echo "Running pytest..."
	PYTHONPATH=src pytest -q

check: lint typecheck test
	@echo "All checks passed."

run:
	@echo "Running DPPNet demo..."
	PYTHONPATH=src python -m main
