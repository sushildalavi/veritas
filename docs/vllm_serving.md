# vLLM Serving Path

Veritas now supports a vLLM-backed explanation generator without changing the verifier decision path.

## Scope

- Verifier labels still come from the compact transformer verifier.
- vLLM is used only for explanation generation.
- Template explanations remain the fallback when vLLM is unavailable.

## Config

Use `configs/vllm_serving.yaml` or set:

- `VERITAS_CONFIG=configs/vllm_serving.yaml`
- `VERITAS_VLLM_BASE_URL`
- `VERITAS_VLLM_MODEL`
- `VERITAS_VLLM_API_KEY`
- `VERITAS_VLLM_TIMEOUT_SECONDS`

## Launch

Dry run:

```bash
python3 scripts/serve_vllm_explanations.py --dry-run
```

Start the OpenAI-compatible vLLM server:

```bash
make serve-vllm-explanations
```

Then start the API:

```bash
VERITAS_CONFIG=configs/vllm_serving.yaml python3 -m uvicorn serving.api:app --reload
```

## Failure mode

If the vLLM endpoint is missing or errors, explanation generation falls back to the template path rather than changing verifier behavior.
