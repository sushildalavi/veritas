# Verifier Data Audit

| split | examples | gold | no_evidence | retrieved_negative | skipped (no gold evidence) |
| --- | --- | --- | --- | --- | --- |
| train | 2808 | 2020 | 415 | 373 | 0 |
| val | 649 | 413 | 117 | 119 | 0 |
| test | 642 | 419 | 102 | 121 | 0 |

## train

- label distribution: {'SUPPORTED': 1416, 'REFUTED': 604, 'NOT_ENOUGH_INFO': 788}
- source distribution: {'fever': 2000, 'scifact': 808}
- duplicate claims: 4
- average claim length (words): 9.33
- average evidence length (words): 41.33

## val

- label distribution: {'SUPPORTED': 237, 'REFUTED': 176, 'NOT_ENOUGH_INFO': 236}
- source distribution: {'fever': 500, 'scifact': 149}
- duplicate claims: 0
- average claim length (words): 9.19
- average evidence length (words): 37.59

## test

- label distribution: {'REFUTED': 195, 'SUPPORTED': 224, 'NOT_ENOUGH_INFO': 223}
- source distribution: {'fever': 493, 'scifact': 149}
- duplicate claims: 0
- average claim length (words): 9.33
- average evidence length (words): 36.49

