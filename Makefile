.PHONY: install train test lint serve build

install:
	pip install -r requirements.txt -r requirements-dev.txt

train:
	python scripts/train.py

test: train
	pytest -v

lint:
	ruff check .

serve: train
	uvicorn app.main:app --reload

build:
	docker build -t iris-classifier:dev .
