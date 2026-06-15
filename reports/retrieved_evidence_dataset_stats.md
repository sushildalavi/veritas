# Retrieved-Evidence Per-Passage Dataset Stats

| split | total pairs | claims | SUPPORTS | REFUTES | NEI | fever | scifact | positive | hard_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | 3663 | 650 | 499 | 395 | 2769 | 2826 | 837 | 894 | 2769 |
| holdout | 3676 | 649 | 491 | 426 | 2759 | 2825 | 851 | 917 | 2759 |

## calibration: pair_type distribution

- irrelevant: 1439
- near_miss: 622
- positive_oracle: 413
- positive_retrieved: 481
- same_entity_insufficient: 101
- same_topic_missing_fact: 607

## holdout: pair_type distribution

- irrelevant: 1493
- near_miss: 601
- positive_oracle: 426
- positive_retrieved: 491
- same_entity_insufficient: 127
- same_topic_missing_fact: 538
