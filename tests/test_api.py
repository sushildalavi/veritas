from fastapi.testclient import TestClient

from serving.api import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_verify_endpoint_rejects_empty_claim() -> None:
    response = client.post("/verify", json={"claim": "   ", "top_k": 5})

    assert response.status_code == 422


def test_verify_endpoint_returns_grounded_response() -> None:
    response = client.post("/verify", json={"claim": "Paris is in France", "top_k": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] in {"SUPPORTED", "REFUTED", "NOT ENOUGH INFO"}
    assert "explanation" in payload
    assert isinstance(payload["evidence"], list)
