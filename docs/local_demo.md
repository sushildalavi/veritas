# Local Demo Guide

Run Veritas end to end locally: FastAPI backend plus the React claim-verification workspace.

## Prerequisites

- Python 3.11+
- Node.js 18+ and npm 9+
- Verifier checkpoint at `checkpoints/deberta_verifier_clean`

## Quick Start

### Terminal 1 - Backend

```bash
make api
```

This starts the FastAPI server at `http://localhost:8000`.

### Terminal 2 - Frontend

```bash
make frontend
```

This starts the Vite dev server at `http://localhost:5173`.

Open `http://localhost:5173` and submit a claim.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status and active backend metadata |
| `GET` | `/metadata` | Project metadata and measured validation metrics |
| `GET` | `/metrics/summary` | Runtime metrics snapshot |
| `POST` | `/verify` | Run retrieval + verifier + explanation |
| `POST` | `/retrieve` | BM25 evidence retrieval only |
| `POST` | `/explain` | Generate a citation-grounded explanation |
| `POST` | `/pipeline` | Full pipeline with latency breakdown |
| `GET` | `/reports/{name}` | Allowlisted research reports |

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

## Browser Workflow

1. Enter or paste a claim
2. Select an evidence depth
3. Verify the claim
4. Review the verdict, confidence, explanation, evidence, and latency

## Production Build

```bash
cd frontend && npm run build
```

The output lands in `frontend/dist/`.

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

Key variables:

- `VERITAS_VERIFIER_CHECKPOINT` - path to the verifier checkpoint if you want to override the default
- `VITE_API_BASE_URL` - backend URL for the frontend
- `VERITAS_CORS_ORIGINS` - allowed browser origins if you use a custom frontend port
