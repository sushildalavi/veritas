# Veritas

Veritas | LLM Fact Verification with Hybrid Retrieval, Learned Ranking, QLoRA, and DPO Alignment

Veritas is a focused factual claim verification project:
claim -> evidence retrieval -> evidence ranking -> claim verification -> grounded explanation -> citation and faithfulness evaluation -> deployment.

## Live Demo

TODO: add the public Hugging Face Spaces URL after deployment.

## Architecture

```mermaid
flowchart LR
    A[Claim] --> B[BM25 + Dense Retrieval]
    B --> C[Hybrid RRF]
    C --> D[Learned Ranker]
    D --> E[Verifier]
    E --> F[Grounded Explanation]
    F --> G[Citation Checker]
    G --> H[Faithfulness Evaluation]
    H --> I[FastAPI + Gradio Demo]
```

## Results

No fabricated results are reported here. Populate this table only after running the matching experiment scripts.

| Component | Metric | Value |
| --- | --- | --- |
| Retrieval | Recall@k | TODO |
| Retrieval | MRR | TODO |
| Retrieval | nDCG@k | TODO |
| Ranking | nDCG@10 | TODO |
| Ranking | MAP | TODO |
| Verification | Macro-F1 | TODO |
| Verification | Per-class F1 | TODO |
| Faithfulness | Citation precision | TODO |
| Faithfulness | Unsupported sentence rate | TODO |
| Calibration | ECE | TODO |

## Why These Choices

- Hybrid retrieval improves recall when lexical and semantic signals disagree.
- Learned evidence ranking can exploit retrieval features without forcing a heavyweight model into the free demo.
- DeBERTa is a practical CPU-friendly verifier checkpoint path when a local checkpoint is available.
- QLoRA is used for offline fine-tuning because it reduces memory pressure during adaptation.
- DPO is aimed at explanation quality and alignment, not just classification accuracy.
- The public demo does not host a live 7B model because the project must remain free to run.

## Data Quality

- Exact deduplication is implemented for claims and evidence spans.
- Near-duplicate detection is exposed with a dependency-light fallback.
- Chunking is configurable by window size and overlap.
- Quality audits report label distribution, duplicate count, missing evidence count, and length statistics.

## Training

- `training/train_deberta.py` is the offline verifier fine-tuning entry point.
- `training/train_qlora.py` is the offline QLoRA path.
- `training/train_dpo.py` is the offline DPO alignment path.
- `training/train_ranker.py` is the learned ranker training entry point.
- These scripts are intentionally lightweight in CI and do not download large models there.

## Evaluation

- Retrieval metrics live in `evaluation/retrieval_metrics.py`.
- Ranking metrics live in `evaluation/ranking_metrics.py`.
- Classification metrics live in `evaluation/classification_metrics.py`.
- Calibration and error analysis live in `evaluation/confidence_analysis.py` and `evaluation/error_analysis.py`.
- Faithfulness helpers live in `evaluation/faithfulness_metrics.py`.
- BEIR wrappers are exposed in `evaluation/beir_benchmark.py`.

## Deployment

- Free public demo target: Hugging Face Spaces + Gradio.
- Local API: `make serve`
- Local UI: `make ui`
- Local CLI: `make cli`
- Export the demo corpus: `make export-demo-corpus`
- Docker deployment: `docker build -t veritas . && docker run -p 8000:8000 veritas`
- The demo uses a lightweight path by default: BM25 retrieval, fallback verifier, template explanation, and citation checking.
- Environment variables are documented in [.env.example](/Users/sushildalavi/Desktop/Github/Veritas/.env.example).

## Service Endpoints

- `GET /health`
- `POST /verify`
- `GET /metrics`

## Makefile

- `make setup`: install Python dependencies
- `make test`: run the test suite
- `make lint`: run a lightweight compile check
- `make data`: placeholder for data pipeline commands
- `make retrieve-eval`: placeholder for retrieval evaluation commands
- `make train-deberta`: placeholder for verifier training commands
- `make serve`: run the FastAPI app with Uvicorn
- `make ui`: launch the Gradio UI

## Limitations

- TODO: fill in all numeric claims after experiments.
- The free demo uses fallback components when checkpoints are unavailable.
- Offline QLoRA and DPO flows are not required for the CPU demo or CI.
- Retrieval corpora in the demo path are intentionally small and should be replaced with project data.

## Resume Bullets

- TODO: add verified metrics after experiments.
- TODO: describe deployment and evaluation outcomes after the benchmark runs.

## CI

![CI badge placeholder](TODO)

## Notes

- No fake metrics are reported in this repository.
- If a table or bullet is not backed by a script run, it should stay `TODO`.
