def test_core_modules_import() -> None:
    import agent
    import cli
    import data
    import evaluation
    import models
    import rag
    import ranking
    import retrieval
    import serving
    import training

    assert agent is not None
    assert cli is not None
    assert data is not None
    assert evaluation is not None
    assert models is not None
    assert rag is not None
    assert ranking is not None
    assert retrieval is not None
    assert serving is not None
    assert training is not None
