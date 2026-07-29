.PHONY: install test lint format demo app

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests app
	mypy src

format:
	ruff format src tests app
	ruff check --fix src tests app

demo:
	bess-opt run-all --config configs/example.yaml

app:
	streamlit run app/streamlit_app.py

