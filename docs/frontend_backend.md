# Frontend / Backend Architecture

## Overview

Veritas uses a FastAPI backend and a React + Vite + TypeScript frontend.

```
frontend/ (React + Vite + TypeScript)
  └── src/
      ├── App.tsx           — tab router, metadata fetch
      ├── api.ts            — typed fetch wrappers
      ├── types.ts          — TypeScript interfaces
      ├── styles.css        — design tokens, layout, components
      └── components/
          ├── Overview.tsx          — metrics cards, architecture, negative results
          ├── VerifyClaim.tsx       — full pipeline UI
          ├── EvidenceExplorer.tsx  — retrieval-only UI
          ├── TrainingArtifacts.tsx — artifact table, MLX LoRA results
          └── ResearchResults.tsx   — eval tables, resume summary

serving/ (FastAPI)
  ├── api.py          — all endpoints + CORS
  ├── schemas.py      — Pydantic request/response models
  ├── errors.py       — structured error types
  ├── startup.py      — artifact existence checks
  ├── dependencies.py — lru_cache singletons
  ├── model_loader.py — pipeline loading
  ├── cache.py        — response cache
  └── monitoring.py   — metrics tracker
```

## Backend Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service liveness, pipeline metadata |
| GET | `/metadata` | Project info, measured metrics, artifact checks |
| GET | `/metrics/summary` | Runtime stats snapshot |
| GET | `/metrics` | Legacy detailed metrics |
| POST | `/verify` | NLI verifier with cache |
| POST | `/retrieve` | BM25 retrieval only |
| POST | `/explain` | Explanation generation (template or MLX) |
| POST | `/pipeline` | Full pipeline with latency breakdown |
| GET | `/reports/{name}` | Allowlisted report file serving |

## Report Allowlist

Only these named reports are served via `/reports/{name}`:

| Name | File |
|------|------|
| `final-results` | `reports/final_veritas_metrics_summary.md` |
| `training-artifacts` | `docs/training_artifacts.md` |
| `error-analysis` | `reports/error_analysis_650.md` |
| `retrieval-ceiling` | `docs/retrieval_ceiling.md` |
| `inference-performance` | `docs/inference_performance.md` |

No arbitrary path traversal is possible — all other names return 404.

## CORS Configuration

Default allowed origins:
- `http://localhost:5173` (Vite frontend)
- `http://localhost:3000` (alternative)
- `http://localhost:8080` (alternative)

Override with `VERITAS_CORS_ORIGINS` env var (comma-separated).

## Startup Checks

On startup, `serving/startup.py` checks for:
- `checkpoints/transformer_verifier_clean` (production verifier)
- `checkpoints/transformer_verifier_clean_onnx` (ONNX export)
- `checkpoints/mlx_lora_verifier` (MLX LoRA verdict-prediction adapter)
- `adapters/mlx_qwen_veritas_lora` (explanation SFT adapter)
- `reports/final_veritas_metrics_summary.json` (final results report)
- `data/demo_corpus.jsonl` (demo corpus for retrieval)

Missing artifacts log warnings but do not crash the server. The `/metadata`
endpoint exposes the artifact check results.

## Frontend ↔ Backend Communication

The frontend reads `VITE_API_BASE_URL` (default: `http://localhost:8000`) to
determine the backend base URL. All requests are typed against `src/types.ts`.

In development, Vite also proxies `/api/*` → backend for convenience,
but the components use the full URL directly via `api.ts`.

## Running Tests

Backend only:
```bash
make test
```

Frontend type check:
```bash
cd frontend && npm run typecheck
```

Frontend production build:
```bash
cd frontend && npm run build
```
