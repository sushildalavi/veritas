"""Sample dataset building helpers for FEVER and SciFact."""

from __future__ import annotations

from dataclasses import asdict
import io
import json
import random
import re
import tarfile
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from data.quality_audit import quality_audit
from data.schemas import ClaimEvidenceRecord, EvidenceSpan

FEVER_URLS = {
    "train": "https://fever.ai/download/fever/train.jsonl",
    "val": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "test": "https://fever.ai/download/fever/paper_test.jsonl",
}

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def normalize_sample_label(label: str) -> str:
    normalized = label.strip().upper()
    if normalized in {"SUPPORTS", "SUPPORT"}:
        return "SUPPORTED"
    if normalized in {"REFUTES", "CONTRADICT", "CONTRADICTS"}:
        return "REFUTED"
    if normalized in {"NOT ENOUGH INFO", "NOT_ENOUGH_INFO", "NEI", "UNKNOWN"}:
        return "NOT_ENOUGH_INFO"
    return normalized.replace(" ", "_")


def _read_jsonl_from_url(url: str) -> list[dict[str, Any]]:
    with urlopen(url, timeout=120) as response:
        return [json.loads(line) for line in response.read().decode("utf-8").splitlines() if line.strip()]


def _download_bytes(url: str) -> bytes:
    with urlopen(url, timeout=180) as response:
        return response.read()


@lru_cache(maxsize=256)
def _fetch_wikipedia_extract(page_title: str) -> str:
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1"
        f"&titles={quote(page_title)}&format=json&formatversion=2&redirects=1"
    )
    request = Request(url, headers={"User-Agent": "VeritasSampleBuilder/1.0 (local research pipeline)"})
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except URLError:
        return ""
    pages = payload.get("query", {}).get("pages", [])
    if not pages:
        return ""
    extract = pages[0].get("extract") or ""
    return str(extract)


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return parts


def _extract_sentence_from_page(page_title: str, sentence_id: int) -> str:
    extract = _fetch_wikipedia_extract(page_title.replace("_", " "))
    sentences = _split_sentences(extract)
    if 0 <= sentence_id < len(sentences):
        return sentences[sentence_id]
    return sentences[0] if sentences else page_title.replace("_", " ")


def _sample_rows(rows: list[dict[str, Any]], target_size: int, seed: int) -> list[dict[str, Any]]:
    if target_size <= 0 or not rows:
        return []
    rng = random.Random(seed)
    if target_size >= len(rows):
        return list(rows)
    return rng.sample(rows, target_size)


def build_fever_sample(
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
) -> tuple[list[ClaimEvidenceRecord], list[ClaimEvidenceRecord], list[ClaimEvidenceRecord]]:
    train_rows = _sample_rows(_read_jsonl_from_url(FEVER_URLS["train"]), train_size, seed)
    val_rows = _sample_rows(_read_jsonl_from_url(FEVER_URLS["val"]), val_size, seed + 1)
    test_rows = _sample_rows(_read_jsonl_from_url(FEVER_URLS["test"]), test_size, seed + 2)
    return (
        [_normalize_fever_row(row) for row in train_rows],
        [_normalize_fever_row(row) for row in val_rows],
        [_normalize_fever_row(row) for row in test_rows],
    )


def _normalize_fever_row(row: dict[str, Any]) -> ClaimEvidenceRecord:
    evidence_groups = row.get("evidence") or []
    evidence_spans: list[EvidenceSpan] = []
    for group_index, group in enumerate(evidence_groups):
        for item_index, item in enumerate(group):
            page_title = item[2] or ""
            sentence_id = int(item[3]) if item[3] is not None else -1
            sentence = _extract_sentence_from_page(page_title, sentence_id) if page_title else ""
            evidence_spans.append(
                EvidenceSpan(
                    doc_id=f"{page_title}:{sentence_id}:{group_index}:{item_index}",
                    text=sentence,
                    title=page_title,
                    metadata={"source": "fever", "evidence_annotation_id": item[0], "evidence_id": item[1]},
                )
            )
    label = normalize_sample_label(str(row.get("label", "NOT ENOUGH INFO")))
    return ClaimEvidenceRecord(
        claim_id=str(row.get("id", "")),
        claim=str(row.get("claim", "")),
        label=label,
        evidence=tuple(evidence_spans),
        split=str(row.get("split", "")) or None,
        metadata={"source": "fever", "verifiable": row.get("verifiable")},
    )


def build_scifact_sample(
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
) -> tuple[list[ClaimEvidenceRecord], list[ClaimEvidenceRecord], list[ClaimEvidenceRecord], list[EvidenceSpan]]:
    archive_bytes = _download_bytes(SCIFACT_URL)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        corpus = _load_scifact_corpus(tar)
        train = _sample_rows(_load_scifact_claims(tar, "data/claims_train.jsonl"), train_size, seed)
        val = _sample_rows(_load_scifact_claims(tar, "data/claims_dev.jsonl"), val_size, seed + 1)
        test = _sample_rows(_load_scifact_claims(tar, "data/claims_test.jsonl"), test_size, seed + 2)
    return (
        [_normalize_scifact_row(row, corpus) for row in train],
        [_normalize_scifact_row(row, corpus) for row in val],
        [_normalize_scifact_row(row, corpus) for row in test],
        corpus,
    )


def _load_scifact_corpus(tar: tarfile.TarFile) -> list[EvidenceSpan]:
    rows = _read_tar_jsonl(tar, "data/corpus.jsonl")
    corpus: list[EvidenceSpan] = []
    for row in rows:
        abstract = row.get("abstract") or []
        text = " ".join(str(sentence) for sentence in abstract if sentence)
        corpus.append(
            EvidenceSpan(
                doc_id=str(row.get("doc_id", "")),
                text=text,
                title=str(row.get("title", "")),
                metadata={"source": "scifact", "structured": row.get("structured", False)},
            )
        )
    return corpus


def _load_scifact_claims(tar: tarfile.TarFile, filepath: str) -> list[dict[str, Any]]:
    return _read_tar_jsonl(tar, filepath)


def _read_tar_jsonl(tar: tarfile.TarFile, filepath: str) -> list[dict[str, Any]]:
    member = tar.getmember(filepath)
    extracted = tar.extractfile(member)
    if extracted is None:
        return []
    return [json.loads(line) for line in extracted.read().decode("utf-8").splitlines() if line.strip()]


def _normalize_scifact_row(row: dict[str, Any], corpus: list[EvidenceSpan]) -> ClaimEvidenceRecord:
    evidence_spans: list[EvidenceSpan] = []
    doc_lookup = {span.doc_id: span for span in corpus}
    evidence = row.get("evidence") or {}
    for doc_id, evidence_groups in evidence.items():
        span = doc_lookup.get(str(doc_id))
        if span is None:
            continue
        for group_index, group in enumerate(evidence_groups):
            sentences = group.get("sentences") or []
            text = _sentences_from_abstract(span.text, sentences)
            evidence_spans.append(
                EvidenceSpan(
                    doc_id=str(doc_id),
                    text=text,
                    title=span.title,
                    metadata={
                        "source": "scifact",
                        "evidence_label": group.get("label"),
                        "sentences": sentences,
                        "group_index": group_index,
                    },
                )
            )

    label = _scifact_label_from_row(row)
    return ClaimEvidenceRecord(
        claim_id=str(row.get("id", "")),
        claim=str(row.get("claim", "")),
        label=label,
        evidence=tuple(evidence_spans),
        split=str(row.get("split", "")) or None,
        metadata={"source": "scifact", "cited_doc_ids": row.get("cited_doc_ids", [])},
    )


def _scifact_label_from_row(row: dict[str, Any]) -> str:
    evidence = row.get("evidence") or {}
    labels = {str(group.get("label", "")).upper() for groups in evidence.values() for group in groups}
    if any(label.startswith("SUPPORT") for label in labels):
        return "SUPPORTED"
    if any(label.startswith("CONTRADICT") or label.startswith("REFUTE") for label in labels):
        return "REFUTED"
    return "NOT_ENOUGH_INFO"


def _sentences_from_abstract(abstract_text: str, sentence_ids: Iterable[int]) -> str:
    sentences = _split_sentences(abstract_text)
    selected = [sentences[index] for index in sentence_ids if 0 <= index < len(sentences)]
    if selected:
        return " ".join(selected)
    return abstract_text


def write_sample_jsonl(records: Iterable[ClaimEvidenceRecord], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_record_to_json(record), ensure_ascii=False) + "\n")
    return output


def _record_to_json(record: ClaimEvidenceRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["evidence"] = [asdict(span) for span in record.evidence]
    return payload


def build_quality_report(records: list[ClaimEvidenceRecord]) -> dict[str, object]:
    return quality_audit(records)


def build_data_quality_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Data Quality Report",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key in [
        "label_distribution",
        "duplicate_count",
        "missing_evidence_count",
        "average_claim_length",
        "average_evidence_length",
        "split_stats",
    ]:
        lines.append(f"| {key} | {report.get(key)} |")
    return "\n".join(lines) + "\n"
