# Explanation SFT Dataset Stats

- output_dir: data/explanations
- total_examples: 4099
- prompt_version: v1
- completion_version: v1

## Split sizes

| Split | Examples | Empty evidence |
| --- | ---: | ---: |
| train | 2808 | 415 |
| val | 649 | 117 |
| test | 642 | 102 |

## Label distribution

{
  "train": {
    "SUPPORTED": 1416,
    "REFUTED": 604,
    "NOT_ENOUGH_INFO": 788
  },
  "val": {
    "SUPPORTED": 237,
    "REFUTED": 176,
    "NOT_ENOUGH_INFO": 236
  },
  "test": {
    "REFUTED": 195,
    "SUPPORTED": 224,
    "NOT_ENOUGH_INFO": 223
  }
}

## Source distribution

{
  "train": {
    "fever": 2000,
    "scifact": 808
  },
  "val": {
    "fever": 500,
    "scifact": 149
  },
  "test": {
    "fever": 493,
    "scifact": 149
  }
}

## Coverage checks

- supported: True
- refuted: True
- not_enough_info: True
- fever: True
- scifact: True
