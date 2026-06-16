.PHONY: setup test lint data build-sample-data build-large-sample-data eval-retrieval eval-ranking eval-retrieval-large eval-ranking-large eval-faithfulness error-analysis pareto-analysis train-verifier train-verifier-smoke serve serve-real serve-vllm-explanations cli export-demo-corpus audit manifest all-evals verify-local build-mlx-lora-data train-mlx-lora eval-mlx-lora eval-topk-verifier install api frontend-install frontend final-reports all-local

## ── Developer workflow ─────────────────────────────────────────────────────

# Install all Python dependencies
install:
	python3 -m pip install -r requirements.txt

# Install frontend dependencies
frontend-install:
	cd frontend && npm install

# Run Python test suite
test:
	python3 -m pytest

# Start FastAPI backend (auto-reloads)
api:
	VERITAS_VERIFIER_CHECKPOINT=checkpoints/transformer_verifier_clean python3 -m uvicorn serving.api:app --reload --port 8000

# Start React frontend dev server (requires backend running)
frontend:
	cd frontend && npm run dev

# Start both backend and frontend simultaneously (requires tmux or two terminals)
# Usage: make api (terminal 1), make frontend (terminal 2)
all-local: api

# Build frontend production bundle
frontend-build:
	cd frontend && npm run build

# Print final research report summaries
final-reports:
	@echo "=== Final Veritas Metrics ==="
	@cat reports/final_veritas_metrics_summary.md

## ── Existing targets ───────────────────────────────────────────────────────

setup:
	python3 -m pip install -r requirements.txt

test:
	python3 -m pytest

lint:
	python3 -m compileall data retrieval ranking models training rag agent evaluation serving tests

data:
	@echo "Phase 1 will add the data pipeline."

build-sample-data:
	python3 scripts/build_sample_datasets.py

build-large-sample-data:
	python3 scripts/build_large_sample_datasets.py --fever-train 2000 --fever-val 500 --fever-test 500

eval-retrieval-large:
	python3 scripts/run_retrieval_eval.py --split val --max-queries 20 --dense-backend sentence-transformers --embedding-model sentence-transformers/all-MiniLM-L6-v2 --file-suffix _large --output-json reports/retrieval_eval_neural_large.json --output-md reports/retrieval_eval_neural_large.md

eval-ranking-large:
	python3 scripts/run_ranking_eval.py --split val --max-queries 2 --candidate-k 5 --train-query-cap 40 --use-cross-encoder --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 --cross-encoder-batch-size 4 --file-suffix _large --output-json reports/ranking_eval_cross_encoder_large.json --output-md reports/ranking_eval_cross_encoder_large.md

manifest:
	python3 scripts/write_artifact_manifest.py

audit: manifest

all-evals: eval-retrieval eval-ranking eval-faithfulness error-analysis pareto-analysis

verify-local: test lint manifest

eval-retrieval:
	python3 scripts/run_retrieval_eval.py

eval-ranking:
	python3 scripts/run_ranking_eval.py

eval-topk-verifier:
	python3 scripts/eval_topk_verifier.py

eval-faithfulness:
	python3 scripts/eval_faithfulness_final.py

error-analysis:
	python3 scripts/error_analysis.py

pareto-analysis:
	python3 scripts/pareto_analysis_final.py

retrieve-eval: eval-retrieval

train-verifier:
	python3 scripts/train_verifier_clean.py

train-verifier-smoke:
	python3 scripts/train_verifier_clean.py --checkpoint-dir /tmp/veritas-verifier-checkpoint

train-deberta:
	python3 scripts/train_deberta_challenger.py

serve:
	python3 -m uvicorn serving.api:app --reload

serve-real:
	VERITAS_VERIFIER_CHECKPOINT=checkpoints/transformer_verifier_clean python3 -m uvicorn serving.api:app --reload

serve-vllm-explanations:
	python3 scripts/serve_vllm_explanations.py

cli:
	python3 cli.py "Paris is in France"

export-demo-corpus:
	python3 -c "from data.export_demo_corpus import export_demo_corpus; export_demo_corpus('data/demo_corpus.jsonl')"

build-mlx-lora-data:
	python3 scripts/build_mlx_lora_dataset.py

train-mlx-lora:
	python3 scripts/train_mlx_lora.py

eval-mlx-lora:
	python3 scripts/train_mlx_lora.py --skip-train
