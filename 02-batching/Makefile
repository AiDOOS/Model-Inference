.PHONY: setup benchmark

setup:
	python -m venv .venv
	. .venv/bin/activate; pip install --upgrade pip
	. .venv/bin/activate; pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
	. .venv/bin/activate; pip install -r requirements.txt

benchmark:
	python benchmark.py
