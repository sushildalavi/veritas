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
- `VERITAS_VLLM_MAX_RETRIES` (default `2`) -- request retries on connection/timeout errors
- `VERITAS_VLLM_MAX_NEW_TOKENS` (default `256`) -- sent as `max_tokens` in the chat-completions payload
- `VERITAS_EXPLANATION_BACKEND` (default `vllm`) -- set to `template` to force template explanations
  even when `explanation_mode: vllm` is configured (useful for CPU-only/local runs)

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

- If `VllmExplanationGenerator` cannot be constructed (e.g. import error), explanation generation
  falls back to the template path.
- If the vLLM endpoint is unreachable at request time, the generator retries up to
  `vllm_max_retries` times, then returns a JSON fallback (`{"explanation": "Explanation
  unavailable: vLLM request failed (...)", "citations": []}`) so the response is well-formed
  instead of raising.
- Verifier behavior is unaffected in either case -- only explanation text changes.
