from app import app


def test_app_entrypoint_builds_demo() -> None:
    assert app is not None
