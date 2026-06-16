# DPO Preference Samples

Rejected responses are synthetic and intentionally contain one failure mode.

## 43158 (wrong_label)

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

### Chosen

```text
Decision: SUPPORTED
Explanation: Taj Mahal therefore the claim is supported.
Citations: ["E1"]
```

### Rejected

```text
Decision: REFUTED
Explanation: The decision is inconsistent with the evidence.
Citations: ["E1"]
```

## 46020 (missing_citation)

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

### Chosen

```text
Decision: SUPPORTED
Explanation: Frank Ocean (born Christopher Edwin Breaux; October 28, 1987) is an American singer-songwriter and rapper therefore the claim is supported.
Citations: ["E1"]
```

### Rejected

```text
Decision: SUPPORTED
Explanation: The answer is plausible but omits citations.
Citations: []
```

## 198393 (hallucinated_fact)

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

### Chosen

```text
Decision: SUPPORTED
Explanation: Marilyn Manson -LRB-band-RRB- therefore the claim is supported.
Citations: ["E1"]
```

### Rejected

```text
Decision: SUPPORTED
Explanation: The evidence says the opposite, and this answer adds unsupported details.
Citations: ["E99"]
```

## 139389 (vague_answer)

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

### Chosen

```text
Decision: SUPPORTED
Explanation: Jolie's commercial success continued with roles in the action films Mr. Jolie's commercial success continued with roles in the action films Mr therefore the claim is supported.
Citations: ["E1"]
```

### Rejected

```text
Decision: SUPPORTED
Explanation: The answer is too vague to be useful.
Citations: ["E1"]
```

## 225831 (wrong_citation)

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

### Chosen

```text
Decision: SUPPORTED
Explanation: Star Wars therefore the claim is supported.
Citations: ["E1"]
```

### Rejected

```text
Decision: SUPPORTED
Explanation: The evidence is cited incorrectly.
Citations: ["E999"]
```

## 100184 (overclaiming)

### Prompt

```text
You are a fact-verification assistant. Use only the provided evidence.

Claim:
Italy was an Allied power.

Verifier label:
REFUTED

Evidence:
[E1] Italy

Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts.
```

### Chosen

```text
Decision: REFUTED
Explanation: Italy therefore the claim is refuted.
Citations: ["E1"]
```

### Rejected

```text
Decision: REFUTED
Explanation: The answer overstates what the evidence proves.
Citations: ["E1"]
```

## 37220 (insufficient_claim)

### Prompt

```text
You are a fact-verification assistant. Use only the provided evidence.

Claim:
Oswald is officially the lone assassin of JFK.

Verifier label:
SUPPORTED

Evidence:
[E1] His vice president, Lyndon B.

Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts.
```

### Chosen

```text
Decision: SUPPORTED
Explanation: His vice president, Lyndon B therefore the claim is supported.
Citations: ["E1"]
```

### Rejected

```text
Decision: NOT_ENOUGH_INFO
Explanation: The answer incorrectly claims support despite insufficient evidence.
Citations: ["E1"]
```

## 97467 (wrong_label)

### Prompt

```text
You are a fact-verification assistant. Use only the provided evidence.

Claim:
The 2011 FIFA U-20 World Cup took place between January 29 to August 20.

Verifier label:
REFUTED

Evidence:
[E1] 2011 FIFA U-20 World Cup

Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts.
```

### Chosen

```text
Decision: REFUTED
Explanation: 2011 FIFA U-20 World Cup therefore the claim is refuted.
Citations: ["E1"]
```

### Rejected

```text
Decision: NOT_ENOUGH_INFO
Explanation: The decision is inconsistent with the evidence.
Citations: ["E1"]
```

## 114682 (missing_citation)

### Prompt

```text
You are a fact-verification assistant. Use only the provided evidence.

Claim:
Donkey was created by an American cartoonist.

Verifier label:
SUPPORTED

Evidence:
[E1] Donkey -LRB-Shrek-RRB- William Steig ( STYGHE; November 14, 1907 – October 3, 2003) was an American cartoonist, illustrator, and children's book author best known for his picture book Shrek!, which inspired the film series of the same name.

Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts.
```

### Chosen

```text
Decision: SUPPORTED
Explanation: Donkey -LRB-Shrek-RRB- William Steig ( STYGHE; November 14, 1907 – October 3, 2003) was an American cartoonist, illustrator, and children's book author best known for his picture book Shrek!, which inspired the film series of the same name therefore the claim is supported.
Citations: ["E1"]
```

### Rejected

```text
Decision: SUPPORTED
Explanation: The answer is plausible but omits citations.
Citations: []
```

## 188021 (hallucinated_fact)

### Prompt

```text
You are a fact-verification assistant. Use only the provided evidence.

Claim:
Sicario (2015 film) was nominated for Best Original Score at the Oscars.

Verifier label:
SUPPORTED

Evidence:
[E1] It also earned BAFTA nominations for Best Supporting Actor, Best Cinematography, and Best Film Music.

Write a concise explanation. Cite the evidence IDs you used. Do not introduce unsupported facts.
```

### Chosen

```text
Decision: SUPPORTED
Explanation: It also earned BAFTA nominations for Best Supporting Actor, Best Cinematography, and Best Film Music therefore the claim is supported.
Citations: ["E1"]
```

### Rejected

```text
Decision: SUPPORTED
Explanation: The evidence says the opposite, and this answer adds unsupported details.
Citations: ["E99"]
```
