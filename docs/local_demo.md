# Local Demo Guide

Run Veritas end-to-end locally: FastAPI backend + React research dashboard.

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm 9+
- Verifier checkpoint at `checkpoints/transformer_verifier_clean`
  (train with `python3 scripts/train_transformer_verifier_clean.py` if missing)

## Quick Start

### Terminal 1 — Backend

```bash
make api
```

This starts the FastAPI server at `http://localhost:8000`.
Equivalent to:
```bash
VERITAS_VERIFIER_CHECKPOINT=checkpoints/transformer_verifier_clean \
  python3 -m uvicorn serving.api:app --reload --port 8000
```

### Terminal 2 — Frontend

```bash
make frontend
```

This starts the Vite dev server at `http://localhost:5173`.
Equivalent to:
```bash
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173` in your browser.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| GET | `/metadata` | Project metadata and measured metrics |
| GET | `/metrics/summary` | Runtime metrics snapshot |
| POST | `/verify` | Run NLI verifier on claim + evidence |
| POST | `/retrieve` | BM25 evidence retrieval only |
| POST | `/explain` | Generate explanation for a claim |
| POST | `/pipeline` | Full pipeline with latency breakdown |
| GET | `/reports/{name}` | Serve allowlisted research reports |

Interactive docs: `http://localhost:8000/docs`

## Example API Calls

```bash
# Health check
curl http://localhost:8000/health

# Full pipeline
curl -X POST http://localhost:8000/pipeline \
  -H "Content-Type: application/json" \
  -d '{"claim": "Marie Curie won the Nobel Prize", "top_k": 5}'

# Evidence retrieval only
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"claim": "The Eiffel Tower is in Paris", "top_k": 3}'

# Final results report
curl http://localhost:8000/reports/final-results
```

## Frontend Tabs

1. **Overview** — measured metrics cards, architecture diagram, negative results
2. **Verify Claim** — full pipeline with verdict, explanation, evidence, latency
3. **Evidence Explorer** — retrieval-only with score ranking
4. **Training Artifacts** — all artifact statuses, MLX LoRA results, what not to claim
5. **Research Results** — full evaluation tables, error analysis, inference benchmarks

## Production Build

```bash
cd frontend && npm run build
```

Output in `frontend/dist/`. Serve with any static file server.

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

Key variables:
- `VERITAS_VERIFIER_CHECKPOINT` — path to verifier checkpoint
- `VITE_API_BASE_URL` — backend URL for frontend (default: `http://localhost:8000`)
- `VERITAS_CORS_ORIGINS` — allowed CORS origins (default: localhost:5173, localhost:3000)
