"""Helpers for grounded explanation training artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CANONICAL_LABELS = ("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO")
PROMPT_VERSION = "v1"
COMPLETION_VERSION = "v1"
DEFAULT_CITATION_PREFIX = "E"


@dataclass(frozen=True)
class ExplanationPassage:
    """A lightweight JSON-serializable evidence passage."""

    doc_id: str
    text: str
    title: str | None = None
    score: float | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata or {})
        return payload


def normalize_label(raw: Any) -> str:
    text = str(raw).strip().upper().replace("-", "_").replace(" ", "_")
    if text in {"SUPPORTED", "SUPPORTS"}:
        return "SUPPORTED"
    if text in {"REFUTED", "REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    return "NOT_ENOUGH_INFO"


def build_evidence_passages(row: dict[str, Any]) -> list[ExplanationPassage]:
    evidence_text = str(row.get("evidence", "")).strip()
    if not evidence_text:
        return []
    metadata = {
        "source": row.get("source"),
        "evidence_type": row.get("evidence_type"),
    }
    return [
        ExplanationPassage(
            doc_id=f"{DEFAULT_CITATION_PREFIX}1",
            text=evidence_text,
            metadata={key: value for key, value in metadata.items() if value is not None},
        )
    ]


def build_prompt(claim: str, verifier_label: str, evidence_passages: list[ExplanationPassage]) -> str:
    evidence_block = _format_evidence_block(evidence_passages)
    return (
        "You are a fact-verification assistant. Use only the provided evidence.\n\n"
        f"Claim:\n{claim.strip()}\n\n"
        f"Verifier label:\n{verifier_label}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts."
    )


def build_completion(verifier_label: str, evidence_passages: list[ExplanationPassage], claim: str) -> tuple[str, list[str]]:
    citations = [passage.doc_id for passage in evidence_passages]
    explanation = _build_explanation(verifier_label, evidence_passages, claim)
    completion = (
        f"Decision: {verifier_label}\n"
        f"Explanation: {explanation}\n"
        f"Citations: {json_list(citations)}"
    )
    return completion, citations


def build_explanation_record(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    claim = str(row.get("claim", "")).strip()
    verifier_label = normalize_label(row.get("label", "NOT_ENOUGH_INFO"))
    evidence_passages = build_evidence_passages(row)
    prompt = build_prompt(claim, verifier_label, evidence_passages)
    completion, citations = build_completion(verifier_label, evidence_passages, claim)
    metadata = {
        "split": split,
        "source": row.get("source"),
        "evidence_type": row.get("evidence_type"),
        "prompt_version": PROMPT_VERSION,
        "completion_version": COMPLETION_VERSION,
        "evidence_count": len(evidence_passages),
    }
    return {
        "claim_id": str(row.get("claim_id", "")),
        "claim": claim,
        "verifier_label": verifier_label,
        "evidence_passages": [passage.to_dict() for passage in evidence_passages],
        "source": row.get("source"),
        "prompt": prompt,
        "completion": completion,
        "citations": citations,
        "metadata": metadata,
    }


def validate_explanation_record(record: dict[str, Any]) -> None:
    required_fields = {
        "claim_id",
        "claim",
        "verifier_label",
        "evidence_passages",
        "source",
        "prompt",
        "completion",
        "citations",
        "metadata",
    }
    missing = sorted(required_fields.difference(record))
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if normalize_label(record["verifier_label"]) not in CANONICAL_LABELS:
        raise ValueError("invalid verifier_label")
    if not isinstance(record["evidence_passages"], list):
        raise ValueError("evidence_passages must be a list")
    for passage in record["evidence_passages"]:
        if not isinstance(passage, dict):
            raise ValueError("evidence_passages entries must be dicts")
        if not str(passage.get("doc_id", "")).strip():
            raise ValueError("evidence passage missing doc_id")
        if "text" not in passage:
            raise ValueError("evidence passage missing text")
    if not str(record["prompt"]).strip():
        raise ValueError("empty prompt")
    if not str(record["completion"]).strip():
        raise ValueError("empty completion")
    if not isinstance(record["citations"], list):
        raise ValueError("citations must be a list")
    if record["verifier_label"] not in str(record["completion"]):
        raise ValueError("completion must include the verifier label")


def build_dpo_pair(record: dict[str, Any], *, rejection_type: str) -> dict[str, Any]:
    verifier_label = normalize_label(record["verifier_label"])
    citations = list(record.get("citations", []))
    prompt = str(record["prompt"])
    chosen = str(record["completion"])
    rejected = _build_rejected_completion(verifier_label, record, rejection_type)
    return {
        "claim_id": record.get("claim_id"),
        "source": record.get("source"),
        "verifier_label": verifier_label,
        "rejection_type": rejection_type,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "citations": citations,
        "metadata": {
            **dict(record.get("metadata", {})),
            "synthetic_rejection": True,
            "rejection_type": rejection_type,
        },
    }


def validate_dpo_pair(pair: dict[str, Any]) -> None:
    required_fields = {"prompt", "chosen", "rejected", "metadata"}
    missing = sorted(required_fields.difference(pair))
    if missing:
        raise ValueError(f"missing fields: {missing}")
    for field in ("prompt", "chosen", "rejected"):
        if not str(pair[field]).strip():
            raise ValueError(f"empty {field}")
    if pair["chosen"] == pair["rejected"]:
        raise ValueError("chosen and rejected must differ")
    if "synthetic_rejection" not in dict(pair["metadata"]):
        raise ValueError("missing synthetic_rejection metadata")


def _build_explanation(verifier_label: str, evidence_passages: list[ExplanationPassage], claim: str) -> str:
    if not evidence_passages:
        if verifier_label == "NOT_ENOUGH_INFO":
            return "The provided evidence does not establish the claim."
        return "The claim cannot be grounded because no evidence passages were provided."

    lead = evidence_passages[0].text.rstrip(".!?")
    if verifier_label == "SUPPORTED":
        return f"{lead} therefore the claim is supported."
    if verifier_label == "REFUTED":
        return f"{lead} therefore the claim is refuted."
    return f"{lead} does not establish the claim, so the correct decision is not enough info."


def _build_rejected_completion(verifier_label: str, record: dict[str, Any], rejection_type: str) -> str:
    evidence_ids = list(record.get("citations", [])) or ["E1"]
    wrong_label = "NOT_ENOUGH_INFO"
    if verifier_label == "NOT_ENOUGH_INFO":
        wrong_label = "SUPPORTED"
    elif verifier_label == "SUPPORTED":
        wrong_label = "REFUTED"

    if rejection_type == "wrong_label":
        return _format_completion(
            wrong_label,
            "The decision is inconsistent with the evidence.",
            evidence_ids,
        )
    if rejection_type == "missing_citation":
        return _format_completion(
            verifier_label,
            "The answer is plausible but omits citations.",
            [],
        )
    if rejection_type == "hallucinated_fact":
        return _format_completion(
            verifier_label,
            "The evidence says the opposite, and this answer adds unsupported details.",
            ["E99"],
        )
    if rejection_type == "vague_answer":
        return _format_completion(
            verifier_label,
            "The answer is too vague to be useful.",
            evidence_ids,
        )
    if rejection_type == "wrong_citation":
        return _format_completion(
            verifier_label,
            "The evidence is cited incorrectly.",
            ["E999"],
        )
    if rejection_type == "overclaiming":
        return _format_completion(
            verifier_label,
            "The answer overstates what the evidence proves.",
            evidence_ids,
        )
    if rejection_type == "insufficient_claim":
        return _format_completion(
            "NOT_ENOUGH_INFO",
            "The answer incorrectly claims support despite insufficient evidence.",
            evidence_ids,
        )
    return _format_completion(
        wrong_label,
        "This answer does not follow the evidence.",
        evidence_ids,
    )


def _format_completion(verdict: str, explanation: str, citations: list[str]) -> str:
    return f"Decision: {verdict}\nExplanation: {explanation}\nCitations: {json_list(citations)}"


def _format_evidence_block(evidence_passages: list[ExplanationPassage]) -> str:
    if not evidence_passages:
        return "(no evidence provided)"
    lines = []
    for passage in evidence_passages:
        title = f" ({passage.title})" if passage.title else ""
        lines.append(f"[{passage.doc_id}]{title} {passage.text.strip()}")
    return "\n".join(lines)


def json_list(items: list[str]) -> str:
    return "[" + ", ".join(f'"{item}"' for item in items) + "]"
