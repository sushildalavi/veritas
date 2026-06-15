# DPO Preference Pair Stats

- input_file: data/processed/verifier_train.jsonl
- output_jsonl: data/dpo_preferences/preferences.jsonl
- sample_size: 1382
- chosen_citation_valid_rate: 1.0 (1382/1382)
- avg_prompt_words: 72.1
- avg_chosen_words: 28.69
- avg_rejected_words: 23.76

## Label distribution

| Label | Count |
| --- | ---: |
| SUPPORTED | 829 |
| REFUTED | 340 |
| NOT ENOUGH INFO | 213 |

## Source distribution

| Source | Count |
| --- | ---: |
| fever | 1090 |
| scifact | 292 |

## Rejection type distribution

| Type | Count |
| --- | ---: |
| wrong_verdict | 461 |
| invalid_citation | 461 |
| unsupported_explanation | 460 |
