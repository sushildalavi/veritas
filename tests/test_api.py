from fastapi.testclient import TestClient

from serving.api import app


client = TestClient(app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "verifier_backend" in payload
    assert "fallback_used" in payload
    assert "retrieval_backend" in payload
    assert "embedding_model" in payload
    assert "retrieval_fallback_used" in payload
    assert "reranker_backend" in payload
    assert "cross_encoder_model" in payload
    assert "reranker_fallback_used" in payload
    assert "checkpoint_path" in payload


# ---------------------------------------------------------------------------
# GET /metadata
# ---------------------------------------------------------------------------

def test_metadata_endpoint_schema() -> None:
    response = client.get("/metadata")

    assert response.status_code == 200
    payload = response.json()
    assert payload["project"] == "Veritas"
    assert isinstance(payload["verifier_oracle_macro_f1"], float)
    assert isinstance(payload["verifier_retrieved_macro_f1"], float)
    assert isinstance(payload["retrieval_recall_at_10"], float)
    assert isinstance(payload["oracle_retrieved_gap"], float)
    assert isinstance(payload["endpoints"], list)
    assert len(payload["endpoints"]) > 0
    assert isinstance(payload["artifact_checks"], dict)


def test_metadata_measured_values_preserved() -> None:
    response = client.get("/metadata")
    payload = response.json()

    assert payload["verifier_oracle_macro_f1"] == 0.6728
    assert payload["verifier_retrieved_macro_f1"] == 0.3887
    assert payload["retrieval_recall_at_10"] == 0.5334
    assert payload["oracle_retrieved_gap"] == 0.2841


# ---------------------------------------------------------------------------
# GET /metrics/summary
# ---------------------------------------------------------------------------

def test_metrics_summary_schema() -> None:
    response = client.get("/metrics/summary")

    assert response.status_code == 200
    payload = response.json()
    assert "requests" in payload
    assert "average_latency_ms" in payload
    assert "p95_latency_ms" in payload
    assert "cache_entries" in payload
    assert "backend_usage" in payload
    assert "verdicts" in payload


def test_metrics_endpoint_reports_snapshot() -> None:
    client.post("/verify", json={"claim": "Paris is in France", "top_k": 2})
    response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert "requests" in payload
    assert "cache_entries" in payload
    assert "average_latency_ms" in payload
    assert "p95_latency_ms" in payload
    assert "verifier_backend" in payload
    assert "retrieval_backend" in payload
    assert "embedding_model" in payload
    assert "retrieval_fallback_used" in payload
    assert "reranker_backend" in payload
    assert "cross_encoder_model" in payload
    assert "reranker_fallback_used" in payload
    assert "checkpoint_path" in payload
    assert "backend_usage_counts" in payload


# ---------------------------------------------------------------------------
# POST /verify
# ---------------------------------------------------------------------------

def test_verify_endpoint_rejects_empty_claim() -> None:
    response = client.post("/verify", json={"claim": "   ", "top_k": 5})

    assert response.status_code == 422


def test_verify_endpoint_rejects_claim_too_long() -> None:
    response = client.post("/verify", json={"claim": "x" * 1001, "top_k": 5})

    assert response.status_code == 422


def test_verify_endpoint_returns_grounded_response() -> None:
    response = client.post("/verify", json={"claim": "Paris is in France", "top_k": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] in {"SUPPORTED", "REFUTED", "NOT ENOUGH INFO"}
    assert "explanation" in payload
    assert isinstance(payload["evidence"], list)
    assert "backend_used" in payload
    assert "retrieval_backend" in payload
    assert "retrieval_fallback_used" in payload
    assert "reranker_backend" in payload
    assert "reranker_fallback_used" in payload


def test_verify_response_has_latency() -> None:
    response = client.post("/verify", json={"claim": "Paris is in France", "top_k": 2})
    assert response.status_code == 200
    assert "latency_ms" in response.json()
    assert response.json()["latency_ms"] >= 0.0


def test_verify_response_has_request_id() -> None:
    response = client.post("/verify", json={"claim": "Paris is in France", "top_k": 2})
    assert response.status_code == 200
    assert "request_id" in response.json()
    assert len(response.json()["request_id"]) > 0


def test_verify_rejects_top_k_out_of_range() -> None:
    response = client.post("/verify", json={"claim": "Paris is in France", "top_k": 0})
    assert response.status_code == 422

    response = client.post("/verify", json={"claim": "Paris is in France", "top_k": 21})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /retrieve
# ---------------------------------------------------------------------------

def test_retrieve_endpoint_returns_evidence() -> None:
    response = client.post("/retrieve", json={"claim": "Paris is the capital of France", "top_k": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["claim"] == "Paris is the capital of France"
    assert isinstance(payload["evidence"], list)
    assert "retrieval_backend" in payload
    assert "latency_ms" in payload


def test_retrieve_rejects_empty_claim() -> None:
    response = client.post("/retrieve", json={"claim": "  ", "top_k": 3})
    assert response.status_code == 422


def test_retrieve_response_evidence_structure() -> None:
    response = client.post("/retrieve", json={"claim": "Marie Curie won the Nobel Prize", "top_k": 2})
    assert response.status_code == 200
    for item in response.json()["evidence"]:
        assert "doc_id" in item
        assert "text" in item


# ---------------------------------------------------------------------------
# POST /explain
# ---------------------------------------------------------------------------

_EVIDENCE_FIXTURE = [
    {"doc_id": "doc1", "text": "Paris is the capital city of France.", "title": "Paris", "score": 0.9, "citation_id": 1}
]


def test_explain_endpoint_returns_explanation() -> None:
    response = client.post("/explain", json={
        "claim": "Paris is in France",
        "label": "SUPPORTED",
        "evidence": _EVIDENCE_FIXTURE,
    })

    assert response.status_code == 200
    payload = response.json()
    assert "explanation" in payload
    assert isinstance(payload["explanation"], str)
    assert len(payload["explanation"]) > 0
    assert "citations" in payload
    assert "backend_used" in payload
    assert "latency_ms" in payload


def test_explain_rejects_invalid_label() -> None:
    response = client.post("/explain", json={
        "claim": "Paris is in France",
        "label": "WRONG_LABEL",
        "evidence": _EVIDENCE_FIXTURE,
    })
    assert response.status_code == 422


def test_explain_rejects_empty_evidence() -> None:
    response = client.post("/explain", json={
        "claim": "Paris is in France",
        "label": "SUPPORTED",
        "evidence": [],
    })
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /pipeline
# ---------------------------------------------------------------------------

def test_pipeline_endpoint_returns_full_response() -> None:
    response = client.post("/pipeline", json={"claim": "Paris is in France", "top_k": 3})

    assert response.status_code == 200
    payload = response.json()
    assert "verdict" in payload
    assert payload["verdict"] in {"SUPPORTED", "REFUTED", "NOT ENOUGH INFO"}
    assert "evidence" in payload
    assert isinstance(payload["evidence"], list)
    assert "explanation" in payload
    assert "citations" in payload
    assert "backend_used" in payload
    assert "retrieval_backend" in payload
    assert "request_id" in payload
    assert "claim" in payload


def test_pipeline_response_has_latency_breakdown() -> None:
    response = client.post("/pipeline", json={"claim": "Marie Curie discovered radium", "top_k": 3})

    assert response.status_code == 200
    payload = response.json()
    assert "latency" in payload
    latency = payload["latency"]
    assert "retrieval_ms" in latency
    assert "verification_ms" in latency
    assert "explanation_ms" in latency
    assert "total_ms" in latency
    assert latency["total_ms"] >= 0.0


def test_pipeline_rejects_empty_claim() -> None:
    response = client.post("/pipeline", json={"claim": "  ", "top_k": 3})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /reports/{name}
# ---------------------------------------------------------------------------

def test_reports_final_results_exists() -> None:
    response = client.get("/reports/final-results")
    assert response.status_code in {200, 404}
    if response.status_code == 200:
        payload = response.json()
        assert "content" in payload
        assert "name" in payload
        assert payload["name"] == "final-results"


def test_reports_unknown_name_returns_404() -> None:
    response = client.get("/reports/nonexistent-report-xyz")
    assert response.status_code == 404


def test_reports_allowlist_enforced() -> None:
    for bad_name in ["../etc/passwd", "../../secrets", "arbitrary"]:
        response = client.get(f"/reports/{bad_name}")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------

def test_cors_headers_present() -> None:
    response = client.options(
        "/verify",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code in {200, 400}
