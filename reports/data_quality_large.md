# Data Quality Report

| Metric | Value |
| --- | --- |
| label_distribution | {'NOT_ENOUGH_INFO': 1249, 'SUPPORTED': 1880, 'REFUTED': 980} |
| duplicate_count | 19 |
| missing_evidence_count | 416 |
| average_claim_length | 9.301289851545388 |
| average_evidence_length | 8.629406160685932 |
| split_stats | {'unspecified': 4109} |

## Sample scale

- Evidence corpus passages: 9804
- FEVER sizes (train/val/test): {'train': 2000, 'val': 500, 'test': 500}
- SciFact sizes (train/val/test): {'train': 809, 'val': 150, 'test': 150}
- Build runtime seconds: 202.113

FEVER evidence text is fetched from Wikipedia and is bounded for runtime;
SciFact ships claims + abstract corpus self-contained and is used at full scale.
SciFact's blind test set carries no labels, so the held-out SciFact test split
is carved deterministically from the labelled dev pool.
