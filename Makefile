.PHONY: setup test lint data build-sample-data retrieve-eval train-deberta serve ui cli export-demo-corpus

setup:
	python3 -m pip install -r requirements.txt

test:
	python3 -m pytest

lint:
	python3 -m compileall data retrieval ranking models training rag agent evaluation serving ui tests

data:
	@echo "Phase 1 will add the data pipeline."

build-sample-data:
	python3 scripts/build_sample_datasets.py

retrieve-eval:
	@echo "Phase 2 will add retrieval evaluation."

train-deberta:
	@echo "Phase 4 will add DeBERTa training."

serve:
	python3 -m uvicorn serving.api:app --reload

ui:
	python3 -m ui.app

cli:
	python3 cli.py "Paris is in France"

export-demo-corpus:
	python3 -c "from data.export_demo_corpus import export_demo_corpus; export_demo_corpus('data/demo_corpus.jsonl')"
