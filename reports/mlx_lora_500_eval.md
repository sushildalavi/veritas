# Explanation Model Evaluation

- input_file: data/explanations/sft_test.jsonl
- sample_size: 25

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
The claim that "Commodore is only a land rank" is refuted by the evidence provided. The evidence explicitly states "Commodore -LRB-rank-RRB- Commodore -LRB-rank-RRB-", which indicates that Commodore is a rank, not limited to land. This directly contradicts the claim that it is only a land rank. Therefore, the claim is refuted by the evidence. The evidence provided is sufficient to disprove the claim. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim is refuted by the evidence. The claim
```
- claim_id: 35882
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim that "Pearl (Steven Universe) exists as a magical dog" is **REFUTED** based on the evidence provided. The evidence states that Pearl is described as a Gem, an alien being, and that she exists as a magical gemstone. This description does not align with the claim that she is a magical dog. The evidence does not provide any information that supports the claim that Pearl is a magical dog. Therefore, the claim is refuted by the evidence. [E1] This evidence ID is used to reference the source of the information provided. The claim is refuted by the evidence, which states that Pearl is a Gem, an alien being, and that she exists as a magical gemstone. This information does not support the claim that Pearl is a magical dog. [E1] This evidence ID is used to reference the source of the information provided. The claim is refuted by the evidence, which states that Pearl is a Gem, an alien being, and
```
- claim_id: 127777
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim "Exotic Birds achieved mostly local failure" is refuted by the evidence provided. The evidence simply states "Exotic Birds" without any additional context or information about their success or failure. There is no information provided to support or refute the specific claim about local failure. Therefore, the claim cannot be verified or refuted based on the evidence alone. The evidence does not provide enough information to either support or contradict the specific claim about local failure. Thus, the claim is refuted by the lack of supporting evidence. The evidence ID used is [E1] "Exotic Birds". The claim is refuted by the evidence. The claim is not supported by the evidence provided. The claim is not refuted by the evidence provided. The claim is not supported by the evidence provided. The claim is not refuted by the evidence provided. The claim is not supported by the evidence provided. The claim is not refuted by the evidence provided. The claim is not supported by the evidence provided
```
- claim_id: 88346
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim is false because there is no evidence provided that Claire Danes plays a cartoon penguin in the film "Temple Grandin". The claim is unsupported by the evidence provided. The evidence ID [E1] is used to reference the claim that "Temple Grandin" is a film. This information alone is not sufficient to verify or refute the specific claim about Claire Danes playing a cartoon penguin. Therefore, the claim is refuted by the lack of supporting evidence. [E1] is the only ID used in this explanation. The claim is refuted by the absence of supporting evidence. [E1] is the only ID used in this explanation. The claim is refuted by the absence of supporting evidence. [E1] is the only ID used in this explanation. The claim is refuted by the absence of supporting evidence. [E1] is the only ID used in this explanation. The claim is refuted by the absence of supporting evidence. [E
```
- claim_id: 14610
  - verifier_label: SUPPORTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim "James Brolin is an actor" is supported by the evidence provided. The evidence explicitly states that James Brolin is an American actor, producer, and director, which directly confirms the claim. Therefore, the claim is supported by the evidence. [E1]
```
- claim_id: 193837
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim that Ed Gagliardi died on October 13, 2005 is refuted by the evidence provided. The evidence states that Edward John Gagliardi, the subject of the claim, died on May 11, 2014, not October 13, 2005. The date of death is explicitly mentioned in the evidence, which provides a complete lifespan of Gagliardi from February 13, 1952 to May 11, 2014. This information directly contradicts the claim, thus the claim is refuted. The evidence IDs used are [E1], which refers to the statement about Edward John Gagliardi's lifespan. [E1] explicitly states the date of death as May 11, 2014, which is not October 13, 2005. Therefore, the claim is refuted by the evidence. [E1]
```
- claim_id: 145707
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim that the great white shark does not prefer to prey on humans because it's a herbivore is refuted by the evidence provided. The evidence states that the great white shark is a carnivorous species, not a herbivore, which contradicts the claim. Therefore, the claim is false. [E1]
```
- claim_id: 188967
  - verifier_label: SUPPORTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim "William Cohen is a politician" is supported by the evidence provided. The evidence explicitly states that William Cohen is an "American attorney, author, and politician from Maine." This information directly confirms that Cohen holds political positions, thus supporting the claim that he is a politician. The evidence ID used is [E1]. The claim is therefore supported by the evidence.
```
- claim_id: 148376
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim that Google Search is not the product of an American multinational technology company is refuted by the evidence provided. The evidence explicitly states that Google LLC, the parent company of Google Search, is an American multinational technology corporation. This directly contradicts the claim that Google Search is not the product of an American company. Therefore, the claim is refuted by the evidence. The evidence ID used is [E1]. The claim is refuted by the evidence provided. The claim that Google Search is not the product of an American multinational technology company is refuted by the evidence provided. The evidence explicitly states that Google LLC, the parent company of Google Search, is an American multinational technology corporation. This directly contradicts the claim that Google Search is not the product of an American company. Therefore, the claim is refuted by the evidence. The evidence ID used is [E1]. The claim that Google Search is not the product of an American multinational technology company is refuted by the evidence provided. The
```
- claim_id: 166925
  - verifier_label: SUPPORTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 0.0
  - output:
```text
The claim that "Baadshah had at least three languages dubbed over it" is supported by the evidence provided. Specifically, the evidence states that T. Brahmanandam won the SIIMA Award for Best Comedian (Telugu) for his comic performance in the film, which implies that the film was dubbed in Telugu. Additionally, the evidence mentions that Brahmanandam won the SIIMA Award for Best Comedian (Telugu) for his comic performance in the film, further supporting the claim. Therefore, the claim is supported by the evidence. [E1]
```
| mlx_adapter | trained | mlx-community/Qwen2.5-1.5B-Instruct-4bit | adapters/mlx_qwen_veritas_lora | 0.28 | 0.24 | 0.72 | 0.72 | 0.92 | 33.24 |

## mlx_adapter Sample Outputs

- claim_id: 204557
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 1.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
The claim is refuted.

Explanation: Commodore -LRB-rank-RRB- Commodore -LRB-rank-RRB- therefore the claim is refuted.

Citations: ["E1"]
```
- claim_id: 35882
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 1.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
Pearl (Steven Universe) exists as a Gem, an alien being that is a gemstone projecting a holographic body, based on the gemstone pearl therefore the claim is refuted.

Explanation: Pearl (Steven Universe) exists as a Gem, an alien being that is a gemstone projecting a holographic body, based on the gemstone pearl therefore the claim is refuted.

Citations: ["E1"]
```
- claim_id: 127777
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 1.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
Support your explanation with the evidence IDs you used. Exotic Birds therefore the claim is refuted.

Explanation: Exotic Birds therefore the claim is refuted.

Citations: ["E1"]
```
- claim_id: 88346
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 0.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
Temple Grandin -LRB-film-RRB-

Write a concise explanation. Cite the evidence IDs you used. REFUTED

Explanation: Temple Grandin -LRB-film-RRB-

Write a concise explanation. Cite the evidence IDs you used. REFUTED

Explanation: Temple Grandin -LRB-film-RRB-

Write a concise explanation. Cite the evidence IDs you used. REFUTED

Explanation: Temple Grandin -LRB-film-RRB-

Write a concise explanation. Cite the evidence IDs you used. REFUTED

Explanation: Temple Grandin -LRB-film-RRB-

Write a concise explanation. Cite the evidence IDs you used. REFUTED

Explanation: Temple Grandin -LRB-film-RRB-

Write a concise explanation. Cite the evidence IDs you used. REFUTED

Explanation: Temple Grandin -LRB-film-RRB-

Write a
```
- claim_id: 14610
  - verifier_label: SUPPORTED
  - format_correct: False
  - citation_precision: 1.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
Support the claim accordingly.

Explanation:
James Brolin (born Craig Kenneth Bruderlin; July 18, 1940) is an American actor, producer, and director therefore the claim is supported.

Citations: ["E1"]
```
- claim_id: 193837
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 1.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
Support your explanation. CLAIM: Edward John Gagliardi died on October 13, 2005.

Explanation: Edward John Gagliardi (February 13, 1952 – May 11, 2014) was an American bass guitarist best known as the original bass player for the 1970s rock band Foreigner therefore the claim is refuted.

Citations: ["E1"]
```
- claim_id: 145707
  - verifier_label: REFUTED
  - format_correct: False
  - citation_precision: 1.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
The claim is refuted.

Explanation: Great white sharks therefore the claim is refuted.

Citations: ["E1"]
```
- claim_id: 188967
  - verifier_label: SUPPORTED
  - format_correct: False
  - citation_precision: 1.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
CLAIM: William Cohen is a politician.

Explanation: William Sebastian Cohen (born August 28, 1940) is an American attorney, author, and politician from Maine therefore the claim is supported.

Citations: ["E1"]
```
- claim_id: 148376
  - verifier_label: REFUTED
  - format_correct: True
  - citation_precision: 1.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
Support your explanation.

Decision: REFUTED

Explanation: Google Search (also known as Google or google.com) is a search engine operated by Google LLC (GOO-gəl), which is an American multinational technology corporation focused on information technology, online advertising, search engine technology, email, cloud computing, software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI). Google LLC therefore the claim is refuted.

Citations: ["E1"]
```
- claim_id: 166925
  - verifier_label: SUPPORTED
  - format_correct: True
  - citation_precision: 1.0
  - unsupported_sentence_rate: 1.0
  - output:
```text
Support your explanation.

Decision: SUPPORTED

Explanation: Brahmanandam won the SIIMA Award for Best Comedian (Telugu) for his comic performance in the film. Brahmanandam therefore the claim is supported.

Citations: ["E1"]
```
