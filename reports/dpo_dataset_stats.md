# DPO Dataset Stats

- output_train: data/explanations/dpo_train.jsonl
- output_val: data/explanations/dpo_val.jsonl
- total_examples: 3457
- synthetic_rejection: True
- prompt_source: data/explanations/sft_{train,val}.jsonl

| Split | Examples |
| --- | ---: |
| train | 2808 |
| val | 649 |

## Label distribution

{
  "SUPPORTED": 1653,
  "REFUTED": 780,
  "NOT_ENOUGH_INFO": 1024
}

## Source distribution

{
  "fever": 2500,
  "scifact": 957
}

## Rejection type distribution

{
  "wrong_label": 495,
  "missing_citation": 494,
  "hallucinated_fact": 494,
  "vague_answer": 494,
  "wrong_citation": 494,
  "overclaiming": 493,
  "insufficient_claim": 493
}
