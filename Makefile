.PHONY: install lint type test eval check serve docker clean

VENV ?= .venv
PY := $(VENV)/bin/python
export VERITY_OTEL_CONSOLE_EXPORT ?= false

install:  ## Create venv and install with dev extras
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

lint:  ## Ruff lint
	$(VENV)/bin/ruff check src evals tests

type:  ## mypy --strict
	$(VENV)/bin/mypy

test:  ## Unit tests
	$(PY) -m pytest

eval:  ## Offline eval suite (the CI quality gate)
	$(PY) -m evals.run_evals

check: lint type test eval  ## Everything CI runs

serve:  ## Run the API (deterministic backend; use real models via Ollama otherwise)
	$(PY) -m verity.cli serve

docker:  ## Bring up the full local stack
	docker compose up --build

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache **/__pycache__
