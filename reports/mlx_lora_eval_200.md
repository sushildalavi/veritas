# MLX LoRA Verifier Evaluation

- base_model: mlx-community/Qwen2.5-1.5B-Instruct-4bit
- adapter_path: checkpoints/mlx_lora_verifier
- eval_file: data/processed/verifier_val.jsonl
- sample_size: 200

| Metric | Value |
| --- | ---: |
| Verdict accuracy | 0.695 |
| Macro F1 | 0.4632 |
| Citation valid rate | 0.6 |
| Unsupported sentence rate | 0.2537 |
| Mean latency (s/example) | 1.2984 |
| Parseable verdicts | 200 / 200 (1.0) |

## Per-class F1

| Label | F1 |
| --- | ---: |
| SUPPORTED | 0.7024 |
| REFUTED | 0.6872 |
| NOT ENOUGH INFO | 0.0 |

## Top error examples

- claim: Henri Kontinen's male sibling is a retired tennis player.
  - gold: REFUTED, predicted: SUPPORTED
  - response: Verdict: SUPPORTED
Explanation: [1] Henri Kontinen is a Finnish former professional tennis player, which the claim is supported.
Citation: [1]
- claim: The Winds of Winter is the sixth season in a series.
  - gold: REFUTED, predicted: SUPPORTED
  - response: Verdict: SUPPORTED
Explanation: [1] The Winds of Winter is the planned sixth and penultimate novel of the epic fantasy series A Song of Ice and Fire by American writer George R.
Citation: [1] The Winds of Winter is the planned sixth and penultimate novel of the epic fantasy series A Song of Ice and 
- claim: Rachel Green appeared in every episode of Friends until the final episode in 2002.
  - gold: REFUTED, predicted: SUPPORTED
  - response: Verdict: SUPPORTED
Explanation: [1] Portrayed by actress Jennifer Aniston, she was created for the show by its creators, David Crane and Marta Kauffman, and appears in all 236 episodes across 10 seasons. Her on-again, off-again relationship with Monica's brother, Ross, forms a central storyline of t
- claim: One of the major features of Zoroastrianism is messianism.
  - gold: SUPPORTED, predicted: REFUTED
  - response: Verdict: REFUTED
Explanation: [1] Opposed to Ahura Mazda is Angra Mainyu (𐬀𐬢𐬭𐬀⸱ 𐬨𐬀𐬌𐬥𐬌𐬌𐬎), who is personified as a destructive spirit and the adversary of all things that are good. Opposed to Ahura Mazda is Angra Mainyu (𐬀𐬢𐬭𐬀⸱ 𐬨𐬀𐬌𐬥𐬌𐬌𐬎), who is personified as a destructive spirit and the adversary of 
- claim: Stanley Williams died in 2007.
  - gold: REFUTED, predicted: SUPPORTED
  - response: Verdict: SUPPORTED
Explanation: [1] Stanley Tookie Williams III (December 29, 1953 – December 13, 2005) was an American gangster who co-founded and led the Crips gang in Los Angeles. The highly publicized trial of Williams and extensive appeals for clemency sparked debate on the status of the death 
