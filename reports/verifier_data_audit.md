# Verifier Data Audit

| split | examples | gold | no_evidence | retrieved_negative | skipped (no gold evidence) |
| --- | --- | --- | --- | --- | --- |
| train | 2809 | 2021 | 415 | 373 | 0 |
| val | 650 | 413 | 118 | 119 | 0 |
| test | 650 | 426 | 103 | 121 | 0 |

## train

- label distribution: {'SUPPORTED': 1417, 'REFUTED': 604, 'NOT_ENOUGH_INFO': 788}
- source distribution: {'fever': 2000, 'scifact': 809}
- duplicate claims: 5
- average claim length (words): 9.33
- average evidence length (words): 41.33

## val

- label distribution: {'SUPPORTED': 237, 'REFUTED': 176, 'NOT_ENOUGH_INFO': 237}
- source distribution: {'fever': 500, 'scifact': 150}
- duplicate claims: 0
- average claim length (words): 9.18
- average evidence length (words): 37.59

## test

- label distribution: {'SUPPORTED': 226, 'REFUTED': 200, 'NOT_ENOUGH_INFO': 224}
- source distribution: {'fever': 500, 'scifact': 150}
- duplicate claims: 0
- average claim length (words): 9.31
- average evidence length (words): 36.65

