"""FastAPI application stub for the Veritas service."""

from fastapi import FastAPI

app = FastAPI(title="Veritas", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
