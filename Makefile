.PHONY: install test run-sim clean

install:
	pip install -r requirements.txt

test:
	python3 -m pytest -q

run-sim:
	python3 simulate.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf data/logs/*.jsonl
