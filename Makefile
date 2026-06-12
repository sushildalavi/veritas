.PHONY: setup test lint data build-sample-data eval-retrieval eval-ranking train-verifier train-verifier-smoke serve serve-real ui cli export-demo-corpus

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

eval-retrieval:
	python3 scripts/run_retrieval_eval.py

eval-ranking:
	python3 scripts/run_ranking_eval.py

retrieve-eval: eval-retrieval

train-verifier:
	python3 scripts/train_verifier.py

train-verifier-smoke:
	python3 scripts/train_verifier.py --checkpoint-dir /tmp/veritas-verifier-checkpoint

train-deberta: train-verifier

serve:
	python3 -m uvicorn serving.api:app --reload

serve-real:
	VERITAS_VERIFIER_CHECKPOINT=checkpoints/verifier python3 -m uvicorn serving.api:app --reload

ui:
	python3 -m ui.app

cli:
	python3 cli.py "Paris is in France"

export-demo-corpus:
	python3 -c "from data.export_demo_corpus import export_demo_corpus; export_demo_corpus('data/demo_corpus.jsonl')"
