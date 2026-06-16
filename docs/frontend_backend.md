# Frontend / Backend Contract

## Overview

Veritas now presents a single product workflow in the browser: paste a claim, verify it against the live backend, and review the verdict, evidence, explanation, and latency.

```
frontend/ (React + Vite + TypeScript)
  └── src/
      ├── App.tsx           — loads service status and metadata, renders the workspace
      ├── api.ts            — typed fetch wrappers for the live API
      ├── types.ts          — shared request/response interfaces
      ├── styles.css        — product styling, layout, and responsive behavior
      └── components/
          └── VerifyClaim.tsx — main claim-verification workspace

serving/ (FastAPI)
  ├── api.py          — API endpoints, CORS, and report allowlist
  ├── schemas.py      — Pydantic request/response models
  ├── errors.py       — structured error helpers
  ├── startup.py      — artifact existence checks
  ├── model_loader.py — retrieval/verifier/explanation pipeline loading
  ├── cache.py        — response cache
  └── monitoring.py   — runtime metrics tracker
```

## Backend Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status and active backend metadata |
| `GET` | `/metadata` | Project summary, measured validation metrics, artifact checks |
| `GET` | `/metrics/summary` | Runtime snapshot of requests, latency, cache usage, and verdicts |
| `GET` | `/metrics` | Legacy detailed metrics snapshot |
| `POST` | `/verify` | Retrieval + verifier + explanation with cache |
| `POST` | `/retrieve` | BM25 evidence retrieval only |
| `POST` | `/explain` | Citation-grounded explanation generation |
| `POST` | `/pipeline` | Full pipeline with latency breakdown |
| `GET` | `/reports/{name}` | Allowlisted research report files |

## Report Allowlist

Only these report names are exposed through `/reports/{name}`:

| Name | File |
|------|------|
| `final-results` | `reports/final_veritas_metrics_summary.md` |
| `training-artifacts` | `docs/training_artifacts.md` |
| `error-analysis` | `reports/error_analysis_650.md` |
| `retrieval-ceiling` | `docs/retrieval_ceiling.md` |
| `inference-performance` | `docs/inference_performance.md` |

## CORS

The backend allows local browser origins by default:

- `http://localhost:5173`
- `http://localhost:3000`
- `http://localhost:8080`

If you run the frontend on a different port, add it to `VERITAS_CORS_ORIGINS`.

## Frontend Runtime Notes

- `VITE_API_BASE_URL` controls the backend origin used by the browser app.
- `VerifyClaim.tsx` submits claims to `/pipeline` and renders the live result.
- The UI disables the main action when the claim exceeds the configured length limit.
- Service health and metadata are fetched on load so the UI can show actual backend status.

## Validation

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
