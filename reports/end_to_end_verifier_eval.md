# End-to-End Verifier Evaluation (Retrieved Evidence)

- Checkpoint: `checkpoints/transformer_verifier_clean`
- Evidence source: top-1 BM25 retrieval over data/processed/evidence_corpus_large.jsonl (9804 passages)
- Retrieval runtime (s): 38.43
- Example count: 650
- Latency (ms/example): 26.26

- accuracy: 0.440
- macro_f1: 0.414

## Per-class metrics

| label | precision | recall | f1 |
| --- | --- | --- | --- |
| SUPPORTED | 0.420 | 0.420 | 0.420 |
| REFUTED | 0.423 | 0.750 | 0.541 |
| NOT_ENOUGH_INFO | 0.594 | 0.183 | 0.280 |

## Confusion matrix

| true \ pred | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
| --- | --- | --- | --- |
| SUPPORTED | 95 | 115 | 16 |
| REFUTED | 38 | 150 | 12 |
| NOT_ENOUGH_INFO | 93 | 90 | 41 |

## Oracle vs. retrieved gap

- accuracy gap (oracle - end_to_end): 0.2769
- macro_f1 gap (oracle - end_to_end): 0.2962

## Top errors

- claim_id=88346 true=REFUTED pred=SUPPORTED: "Temple Grandin features Claire Danes playing a cartoon penguin." | evidence: "Temple Grandin -LRB-film-RRB-"
- claim_id=188967 true=SUPPORTED pred=REFUTED: "William Cohen is a politician." | evidence: "William Sebastian Cohen (born August 28, 1940) is an American attorney, author, and politician from Maine."
- claim_id=115807 true=REFUTED pred=SUPPORTED: "Susan Collins was the second woman to become the nominee of a major party for Governor of Maine." | evidence: "He became involved in the Labour Party and was elected to the House of Commons in 1983 for Sedgefield."
- claim_id=81480 true=REFUTED pred=SUPPORTED: "Matt Bomer was born in Spain." | evidence: "Matt Bomer"
- claim_id=172302 true=SUPPORTED pred=REFUTED: "Scream 2 is a film that is categorized as a slasher." | evidence: "Scream 2 is a 1997 American slasher film directed by Wes Craven and written by Kevin Williamson."
- claim_id=148171 true=SUPPORTED pred=REFUTED: "SummerSlam had no pre-show, but contested ten matches." | evidence: "Ten matches were contested at the event, with no match on the pre-show."
- claim_id=51435 true=SUPPORTED pred=REFUTED: "There was an attempt to change Cyprus." | evidence: "Glass Houses (1980) was an attempt to further establish him as a rock artist; it featured "It's Still Rock and Roll to Me" (Joel's first single to top the Billboard Hot 100), "You May Be Right", "Don't Ask Me Why", and "Sometimes a Fantasy"."
- claim_id=61922 true=REFUTED pred=SUPPORTED: "The American actor that plays Chumlee was born on December." | evidence: "Jacob Benjamin Gyllenhaal ( JIL-ən-hawl, Swedish: [ˈjʏ̂lːɛnˌhɑːl]; born December 19, 1980) is an American actor whose career on screen and stage has spanned more than three decades."
- claim_id=12118 true=SUPPORTED pred=REFUTED: "Yugoslavia was a country." | evidence: "In 1963, the country was renamed for the final time, as the Socialist Federal Republic of Yugoslavia (SFRY)."
- claim_id=31728 true=SUPPORTED pred=REFUTED: "Medical school around the world vary in admission structure." | evidence: "Assessment for selection in medicine and the health professions should follow the same quality assurance processes as in-course assessment. The literature on selection is limited and is not strongly theoretical or conceptual. For written testing, there is evidence of the predictive validity of Medic"

