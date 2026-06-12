from pathlib import Path


def test_repository_skeleton_exists() -> None:
    assert Path("README.md").exists()
    assert Path("pyproject.toml").exists()
    assert Path("Makefile").exists()
