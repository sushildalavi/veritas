from pathlib import Path

from data.build_preference_pairs import CandidateExplanation, build_preference_pairs
from training.train_dpo import load_config as load_dpo_config
from training.train_qlora import load_config as load_qlora_config


def test_preference_pairs_require_quality_gap() -> None:
    pairs = build_preference_pairs(
        [
            [
                CandidateExplanation("1", "good answer", 0.9, 0.9, 0.9),
                CandidateExplanation("1", "bad answer", 0.1, 0.1, 0.1),
            ],
            [
                CandidateExplanation("2", "close answer a", 0.6, 0.6, 0.6),
                CandidateExplanation("2", "close answer b", 0.55, 0.55, 0.55),
            ],
        ],
        quality_gap_threshold=0.2,
    )

    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "good answer"
    assert pairs[0]["rejected"] == "bad answer"


def test_alignment_configs_parse(tmp_path: Path) -> None:
    qlora = load_qlora_config("configs/qlora_phi.yaml")
    dpo = load_dpo_config("configs/dpo.yaml")

    assert qlora["base_model"] == "microsoft/phi-2"
    assert dpo["base_model"] == "microsoft/phi-2"
