# MLX LoRA Verifier Evaluation

- base_model: mlx-community/Qwen2.5-1.5B-Instruct-4bit
- adapter_path: checkpoints/mlx_lora_verifier_300
- eval_file: data/processed/verifier_val.jsonl
- sample_size: 200

| Metric | Value |
| --- | ---: |
| Verdict accuracy | 0.52 |
| Macro F1 | 0.24 |
| Citation valid rate | 0.545 |
| Unsupported sentence rate | 0.3021 |
| Mean latency (s/example) | 1.3534 |
| Parseable verdicts | 200 / 200 (1.0) |

## Per-class F1

| Label | F1 |
| --- | ---: |
| SUPPORTED | 0.68 |
| REFUTED | 0.04 |
| NOT ENOUGH INFO | 0.0 |

## Top error examples

- claim: Catherine Hardwicke is exclusively Irish.
  - gold: REFUTED, predicted: SUPPORTED
  - response: Verdict: SUPPORTED
Explanation: [1] Catherine Hardwicke therefore the claim is supported.
Citation: [1]
- claim: Henri Kontinen's male sibling is a retired tennis player.
  - gold: REFUTED, predicted: SUPPORTED
  - response: Verdict: SUPPORTED
Explanation: [1] Henri Kontinen (Finnish pronunciation: [ˈhenri ˈkontinen]; born 19 June 1990) is a Finnish former professional tennis player who is a former world No. 1 in doubles in 2012 and 2013. Therefore, the claim is supported.
Citation: [1]
- claim: The Winds of Winter is the sixth season in a series.
  - gold: REFUTED, predicted: SUPPORTED
  - response: Verdict: SUPPORTED
Explanation: [1] The Winds of Winter is the planned sixth and penultimate novel of the epic fantasy series A Song of Ice and Fire by American writer George R therefore the claim is supported.
Citation: [1]
- claim: Rachel Green appeared in every episode of Friends until the final episode in 2002.
  - gold: REFUTED, predicted: SUPPORTED
  - response: Verdict: SUPPORTED
Explanation: [1] Portrayed by actress Jennifer Aniston, she was created for the show by its creators, David Crane and Marta Kauffman, and appears in all 236 episodes across 10 seasons. Her on-again, off-again relationship with Monica's brother, Ross, forms a central storyline of t
- claim: The Little Prince (2015 film) is directed by Michael Bay only.
  - gold: REFUTED, predicted: SUPPORTED
  - response: Verdict: SUPPORTED
Explanation: [1] The Little Prince (French: Le Petit Prince; Italian: Il piccolo principe) is a 2015 English-language French-Italian animated fantasy adventure comedy-drama film directed by Mark Osborne and based on the 1943 novella of the same name by Antoine de Saint-Exupéry the
