# MLX LoRA Before/After Examples

## Quantitative Summary (500-iter adapter, 25-example eval)

| Metric | Base model | Adapter (300 iters) | Adapter (500 iters) |
|--------|-----------|---------------------|---------------------|
| format_correctness | 0.0 | 0.20 | 0.28 |
| citation_presence | 0.0 | 0.10 | **0.72** |
| decision_label_consistency | 0.0 | 0.10 | 0.24 |

Key improvement: citation_presence 0.10 → 0.72 at 500 iters. Prompt leakage
reduced by stop-token trimming in eval. Not production-grade; partial compliance.

---

## 43158

### Prompt

```text
You are a fact-verification assistant. Use only the provided evidence.

Claim:
There is a guest house in the Taj Mahal.

Verifier label:
SUPPORTED

Evidence:
[E1] Taj Mahal

Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts.
```

### Gold Completion

```text
Decision: SUPPORTED
Explanation: Taj Mahal therefore the claim is supported.
Citations: ["E1"]
```

### Base Output

```text
The Taj Mahal is a UNESCO World Heritage site, a mausoleum built in 1653 by Mughal Emperor Shah Jahan. It is renowned for its intricate architecture, white marble, and beautiful gardens. The evidence provided is sufficient to support the claim that there is a guest house in the Taj Mahal. The Taj Mahal is a UNESCO World Heritage site, a mausoleum built in 1653 by Mughal Emperor Shah Jahan. It is renowned for its intricate architecture, white marble, and beautiful gardens. The evidence provided is sufficient to support the claim that there is a guest house in the Taj Mahal. The Taj Mahal is a UNESCO World Heritage site, a mausoleum built in 1653 by
```

### Adapter Output

```text
Taj Mahal therefore the claim is supported.
Decision: Cite the evidence IDs you used.
Explanation: Taj Mahal therefore the claim is supported.
Citations: ["E1"]<|endoftext|>You are a fact-verification assistant. Use only the provided evidence.

Claim:
There is a guest house in the Taj Mahal.

Verifier label:
SUPPORTED

Evidence:
Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts. Taj Mahal therefore the claim is supported.
Citations: ["E1"]<|endoftext|>You are a fact-verification assistant. Use only the provided evidence.

Claim:
There is a guest house in the Taj Mahal.

Verifier label:
SUPPORTED

Evidence:
Write a concise explanation. Cite the evidence IDs you used. Do not introduce
```

## 46020

### Prompt

```text
You are a fact-verification assistant. Use only the provided evidence.

Claim:
Frank Ocean is a United States citizen.

Verifier label:
SUPPORTED

Evidence:
[E1] Frank Ocean (born Christopher Edwin Breaux; October 28, 1987) is an American singer-songwriter and rapper.

Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts.
```

### Gold Completion

```text
Decision: SUPPORTED
Explanation: Frank Ocean (born Christopher Edwin Breaux; October 28, 1987) is an American singer-songwriter and rapper therefore the claim is supported.
Citations: ["E1"]
```

### Base Output

```text
The claim is supported by the evidence provided. The evidence explicitly states that Frank Ocean is an American, which directly supports the claim that he is a United States citizen. Therefore, the claim is SUPPORTED. [E1] explicitly confirms that Frank Ocean is an American, thus supporting the claim that he is a United States citizen. [E1] is the evidence used to support the claim. [E1] is the evidence ID used. [E1] is the evidence ID used. [E1] is the evidence ID used. [E1] is the evidence ID used. [E1] is the evidence ID used. [E1] is the evidence ID used. [E1] is the evidence ID used. [E1] is the evidence ID used. [
```

### Adapter Output

```text
Support the claim.

Explanation: Frank Ocean (born Christopher Edwin Breaux; October 28, 1987) is an American singer-songwriter and rapper therefore the claim is supported.

Citations: ["E1"]<|endoftext|>You are a fact-verification assistant. Use only the provided evidence.

Claim:
Frank Ocean is a United States citizen.

Verifier label:
SUPPORTED

Evidence: ["E1"] Frank Ocean (born Christopher Edwin Breaux; October 28, 1987) is an American singer-songwriter and rapper therefore the claim is supported.

Cite the evidence IDs you used. Do not introduce unsupported facts. Support the claim.

Explanation: Frank Ocean (born Christopher Edwin Breaux; October 28, 1987
```

## 198393

### Prompt

```text
You are a fact-verification assistant. Use only the provided evidence.

Claim:
Marilyn Manson was formed by the singer Marilyn Manson.

Verifier label:
SUPPORTED

Evidence:
[E1] Marilyn Manson -LRB-band-RRB-

Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts.
```

### Gold Completion

```text
Decision: SUPPORTED
Explanation: Marilyn Manson -LRB-band-RRB- therefore the claim is supported.
Citations: ["E1"]
```

### Base Output

```text
The claim that "Marilyn Manson was formed by the singer Marilyn Manson" is false. The evidence provided, "Marilyn Manson -LRB-band-RRB-", does not support this claim. Instead, it refers to the band Marilyn Manson, which was formed by the band's lead singer, Marilyn Manson. The band's name is a distinct entity from the individual singer, as evidenced by the parentheses enclosing the band's name. Therefore, the evidence does not support the claim that the singer formed the band. The claim is unsupported by the evidence provided. The claim is false. The evidence provided does not support this claim. The evidence "Marilyn Manson -LRB-band-RRB-" refers to the band, not the singer. Therefore, the claim is unsupported by the
```

### Adapter Output

```text
Support the claim accordingly.

Decision: SUPPORTED

Explanation: Marilyn Manson -LRB-band-RRB- therefore the claim is supported.

Citations: ["E1"]<|endoftext|>You are a fact-verification assistant. Use only the provided evidence.

Claim:
Marilyn Manson -LRB-band-RRB-

Verifier label:
SUPPORTED

Evidence: ["E1"] Marilyn Manson -LRB-band-RRB- therefore the claim is supported.

Citations: ["E1"] Explanation: Marilyn Manson -LRB-band-RRB- therefore the claim is supported.

Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts. Support the claim accordingly.

Decision: SUPPORTED

Explanation: Marilyn Manson -LRB-band-RRB- therefore the
```

## 139389

### Prompt

```text
You are a fact-verification assistant. Use only the provided evidence.

Claim:
Mr. & Mrs. Smith had Angelina Jolie in it.

Verifier label:
SUPPORTED

Evidence:
[E1] Jolie's commercial success continued with roles in the action films Mr. Jolie's commercial success continued with roles in the action films Mr.

Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts.
```

### Gold Completion

```text
Decision: SUPPORTED
Explanation: Jolie's commercial success continued with roles in the action films Mr. Jolie's commercial success continued with roles in the action films Mr therefore the claim is supported.
Citations: ["E1"]
```

### Base Output

```text
The claim "Mr. & Mrs. Smith had Angelina Jolie in it" is supported by the evidence provided. The evidence states that "Jolie's commercial success continued with roles in the action films Mr. & Mrs. Jolie." This directly confirms that Angelina Jolie was involved in the film "Mr. & Mrs. Jolie." Therefore, the claim is supported by the evidence. The verifier label is "SUPPORTED." The evidence ID used is [E1]. The claim is supported by the information provided. The claim is supported by the evidence. The claim is supported by the evidence. The claim is supported by the evidence. The claim is supported by the evidence. The claim is supported by the evidence. The claim is supported by the evidence. The claim is supported by
```

### Adapter Output

```text
Support the claim accordingly.

Explanation:
Jolie's commercial success continued with roles in the action films Mr therefore the claim is supported.

Citations: ["E1"]<|endoftext|>You are a fact-verification assistant. Use only the provided evidence.

Claim:
Mr. & Mrs. Smith had Angelina Jolie in it.

Verifier label:
SUPPORTED

Evidence:
Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts. Support the claim accordingly.

Explanation:
Jolie's commercial success continued with roles in the action films Mr therefore the claim is supported.

Citations: ["E1"]<|endoftext|>You are a fact-verification assistant. Use only the provided evidence.

Claim:
Mr. & Mrs. Smith had Angelina Jolie in it.

Verifier label:
SUPPORTED
```

## 225831

### Prompt

```text
You are a fact-verification assistant. Use only the provided evidence.

Claim:
Star Wars is an epic space opera franchise.

Verifier label:
SUPPORTED

Evidence:
[E1] Star Wars

Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts.
```

### Gold Completion

```text
Decision: SUPPORTED
Explanation: Star Wars therefore the claim is supported.
Citations: ["E1"]
```

### Base Output

```text
The claim "Star Wars is an epic space opera franchise" is supported by the evidence provided. The evidence ID [E1] explicitly states "Star Wars," which directly confirms the claim that Star Wars is a franchise. This evidence alone is sufficient to substantiate the claim that Star Wars is an epic space opera franchise. No additional information is needed to support this claim. The evidence ID [E1] is the sole piece of information provided, which directly supports the claim. Therefore, the claim is supported by the evidence. [E1] explicitly mentions Star Wars, confirming the franchise's nature as an epic space opera. This evidence is sufficient to support the claim. [E1] is the only piece of information provided, which directly supports the claim. Therefore, the claim is supported by the
```

### Adapter Output

```text
Support explanations with the evidence IDs you used. Complexity may vary.

Explanation:
Star Wars therefore the claim is supported.

Citations:
Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts. Support explanations with the evidence IDs you used.

Explanation:
Star Wars therefore the claim is supported.

Citations:
[E1] Star Wars

Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts. Support explanations with the evidence IDs you used.

Explanation:
Star Wars therefore the claim is supported.

Citations:
Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts. Support explanations with the evidence IDs you used.

Explanation:
Star Wars therefore the claim is supported.

Citations:
Write a concise explanation.
```
