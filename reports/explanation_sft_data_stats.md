# Explanation SFT Data Stats

- output_prefix: data/processed/explanation_sft
- total_examples: 4109
- strict_json_output: True

| Split | Examples |
| --- | ---: |
| train | 2809 |
| val | 650 |
| test | 650 |

## Label distribution

{
  "train": {
    "SUPPORTED": 1417,
    "REFUTED": 604,
    "NOT_ENOUGH_INFO": 788
  },
  "val": {
    "SUPPORTED": 237,
    "REFUTED": 176,
    "NOT_ENOUGH_INFO": 237
  },
  "test": {
    "SUPPORTED": 226,
    "REFUTED": 200,
    "NOT_ENOUGH_INFO": 224
  }
}

## Source distribution

{
  "train": {
    "unknown": 2809
  },
  "val": {
    "unknown": 650
  },
  "test": {
    "unknown": 650
  }
}

## Average lengths

{
  "average_claim_length": {
    "train": 9.33,
    "val": 9.18,
    "test": 9.31
  },
  "average_evidence_length": {
    "train": 35.23,
    "val": 30.76,
    "test": 30.84
  }
}
