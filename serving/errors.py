"""Structured error types for the Veritas serving layer."""

from __future__ import annotations

from fastapi import HTTPException


def claim_too_long(max_length: int) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={"code": "CLAIM_TOO_LONG", "message": f"Claim exceeds {max_length} character limit."},
    )


def claim_empty() -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={"code": "CLAIM_EMPTY", "message": "Claim cannot be empty."},
    )


def artifact_missing(name: str, setup_hint: str = "") -> HTTPException:
    detail: dict = {"code": "ARTIFACT_MISSING", "artifact": name}
    if setup_hint:
        detail["setup_hint"] = setup_hint
    return HTTPException(status_code=503, detail=detail)


def backend_unavailable(name: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={"code": "BACKEND_UNAVAILABLE", "backend": name, "message": f"{name} backend is not available in this environment."},
    )


def report_not_found(name: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"code": "REPORT_NOT_FOUND", "report": name, "message": f"Report '{name}' does not exist."},
    )
