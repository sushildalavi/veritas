"""Core project utilities for Veritas."""

from .config import ProjectSettings, load_project_settings, load_yaml_config
from .evidence_formatting import (
    EvidenceStyle,
    canonicalize_evidence_blocks,
    choose_evidence_style,
    format_verifier_text,
    render_evidence,
    sanitize_evidence_text,
    split_evidence_blocks,
)

__all__ = [
    "EvidenceStyle",
    "ProjectSettings",
    "canonicalize_evidence_blocks",
    "choose_evidence_style",
    "format_verifier_text",
    "load_project_settings",
    "load_yaml_config",
    "render_evidence",
    "sanitize_evidence_text",
    "split_evidence_blocks",
]
