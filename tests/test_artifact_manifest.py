import json
from pathlib import Path

from scripts.write_artifact_manifest import _git_commit_hash, _list_artifacts


def test_list_artifacts_skips_gitkeep(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    (root / ".gitkeep").write_text("", encoding="utf-8")
    (root / "sample.json").write_text("{}", encoding="utf-8")

    items = _list_artifacts(root)

    assert len(items) == 1
    assert items[0]["path"].endswith("sample.json")


def test_git_commit_hash_is_optional(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("scripts.write_artifact_manifest.subprocess.run", boom)

    assert _git_commit_hash() is None
