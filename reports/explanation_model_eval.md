# Explanation Model Evaluation

- input_file: data/explanations/sft_test.jsonl
- sample_size: 5

| Backend | Status | Model | Adapter | Format | Decision | Citations | Precision | Unsupported | Length |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mlx_base | measured | mlx-community/Qwen2.5-1.5B-Instruct-4bit | - | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## mlx_base Sample Outputs

- claim_id: 204557
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim that "Commodore is only a land rank" is refuted by the evidence provided. The evidence explicitly states "Commodore -LRB-rank-RRB- Commodore -LRB-rank-RRB-", which indicates that Commodore is a rank, not limited to land. This directly contradicts the claim that it is only a land rank. Therefore, the claim is refuted by the evidence. The evidence provided is sufficient to disprove the claim. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the
```
- claim_id: 35882
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim that "Pearl (Steven Universe) exists as a magical dog" is **REFUTED** based on the evidence provided. The evidence states that Pearl is described as a Gem, an alien being, and that she exists as a magical gemstone. This description does not align with the claim that she is a magical dog. The evidence does not provide any information that supports the claim that Pearl is a magical dog. Therefore, the claim is refuted by the evidence. [E1] This evidence ID is used to reference the source of the information provided. The claim is refuted by the evidence, which states that Pearl is a Gem, an alien being, and that she exists as a magical gemstone. This information does not support the claim that Pearl is a magical dog.
```
- claim_id: 127777
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim "Exotic Birds achieved mostly local failure" is refuted by the evidence provided. The evidence simply states "Exotic Birds" without any additional context or information about their success or failure. There is no information provided to support or refute the specific claim about local failure. Therefore, the claim cannot be verified or refuted based on the evidence alone. The evidence does not provide enough information to either support or contradict the specific claim about local failure. Thus, the claim is refuted by the lack of supporting evidence. The evidence ID used is [E1] "Exotic Birds". The claim is refuted by the evidence. The claim is not supported by the evidence provided. The claim is not refuted by the evidence provided. The claim is not supported by the evidence provided. The
```
- claim_id: 88346
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim is false because there is no evidence provided that Claire Danes plays a cartoon penguin in the film "Temple Grandin". The claim is unsupported by the evidence provided. The evidence ID [E1] is used to reference the claim that "Temple Grandin" is a film. This information alone is not sufficient to verify or refute the specific claim about Claire Danes playing a cartoon penguin. Therefore, the claim is refuted by the lack of supporting evidence. [E1] is the only ID used in this explanation. The claim is refuted by the absence of supporting evidence. [E1] is the only ID used in this explanation. The claim is refuted by the absence of supporting evidence. [E1] is the only ID used in this explanation
```
- claim_id: 14610
  - verifier_label: SUPPORTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim "James Brolin is an actor" is supported by the evidence provided. The evidence explicitly states that James Brolin is an American actor, producer, and director, which directly confirms the claim. Therefore, the claim is supported by the evidence. [E1]<|endoftext|>Human: You are a fact-verification assistant. Use only the provided evidence.

Claim:
James Brolin is an actor.

Verifier label:
SUPPORTED

Evidence:
[E1] James Brolin (born Craig Kenneth Bruderlin; July 18, 1940) is an American actor, producer, and director.

Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts. The claim "James Brolin is an actor" is supported
```
| mlx_adapter | trained | mlx-community/Qwen2.5-1.5B-Instruct-4bit | adapters/mlx_qwen_veritas_lora | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 23.0 |

## mlx_adapter Sample Outputs

- claim_id: 204557
  - verifier_label: REFUTED
  - format_correct: True
  - citation_precision: 0.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
Decision: NOT_ENOUGH_INFO
Explanation: generation failed: [matmul] Last dimension of first input with shape (1,77,1536) must match second to last dimension of second input with shape (1024,8).
Citations: []
```
- claim_id: 35882
  - verifier_label: REFUTED
  - format_correct: True
  - citation_precision: 0.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
Decision: NOT_ENOUGH_INFO
Explanation: generation failed: [matmul] Last dimension of first input with shape (1,86,1536) must match second to last dimension of second input with shape (1024,8).
Citations: []
```
- claim_id: 127777
  - verifier_label: REFUTED
  - format_correct: True
  - citation_precision: 0.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
Decision: NOT_ENOUGH_INFO
Explanation: generation failed: [matmul] Last dimension of first input with shape (1,58,1536) must match second to last dimension of second input with shape (1024,8).
Citations: []
```
- claim_id: 88346
  - verifier_label: REFUTED
  - format_correct: True
  - citation_precision: 0.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
Decision: NOT_ENOUGH_INFO
Explanation: generation failed: [matmul] Last dimension of first input with shape (1,72,1536) must match second to last dimension of second input with shape (1024,8).
Citations: []
```
- claim_id: 14610
  - verifier_label: SUPPORTED
  - format_correct: True
  - citation_precision: 0.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
Decision: NOT_ENOUGH_INFO
Explanation: generation failed: [matmul] Last dimension of first input with shape (1,86,1536) must match second to last dimension of second input with shape (1024,8).
Citations: []
```
