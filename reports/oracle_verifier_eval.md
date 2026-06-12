# Oracle Verifier Evaluation

- Checkpoint: `checkpoints/transformer_verifier_clean`
- Evidence source: gold/curated evidence from data/processed/verifier_test.jsonl
- Example count: 650
- Latency (ms/example): 22.09

- accuracy: 0.717
- macro_f1: 0.710

## Per-class metrics

| label | precision | recall | f1 |
| --- | --- | --- | --- |
| SUPPORTED | 0.662 | 0.469 | 0.549 |
| REFUTED | 0.561 | 0.740 | 0.638 |
| NOT_ENOUGH_INFO | 0.938 | 0.946 | 0.942 |

## Confusion matrix

| true \ pred | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
| --- | --- | --- | --- |
| SUPPORTED | 106 | 115 | 5 |
| REFUTED | 43 | 148 | 9 |
| NOT_ENOUGH_INFO | 11 | 1 | 212 |

## Top errors

- claim_id=88346 true=REFUTED pred=SUPPORTED: "Temple Grandin features Claire Danes playing a cartoon penguin." | evidence: "Temple Grandin -LRB-film-RRB-"
- claim_id=188967 true=SUPPORTED pred=REFUTED: "William Cohen is a politician." | evidence: "William Sebastian Cohen (born August 28, 1940) is an American attorney, author, and politician from Maine."
- claim_id=166925 true=SUPPORTED pred=NOT_ENOUGH_INFO: "Baadshah had at least three languages dubbed over it." | evidence: "Brahmanandam won the SIIMA Award for Best Comedian (Telugu) for his comic performance in the film. T. Brahmanandam won the SIIMA Award for Best Comedian (Telugu) for his comic performance in the film. Brahmanandam won the SIIMA Award for Best Comedian (Telugu) for his comic performance in the film."
- claim_id=115807 true=REFUTED pred=SUPPORTED: "Susan Collins was the second woman to become the nominee of a major party for Governor of Maine." | evidence: "After her bid, she became the founding director of the Center for Family Business at Husson University in Bangor, Maine."
- claim_id=81480 true=REFUTED pred=SUPPORTED: "Matt Bomer was born in Spain." | evidence: "Matt Bomer"
- claim_id=172302 true=SUPPORTED pred=REFUTED: "Scream 2 is a film that is categorized as a slasher." | evidence: "Scream 2 is a 1997 American slasher film directed by Wes Craven and written by Kevin Williamson."
- claim_id=148171 true=SUPPORTED pred=REFUTED: "SummerSlam had no pre-show, but contested ten matches." | evidence: "Ten matches were contested at the event, with no match on the pre-show."
- claim_id=51435 true=SUPPORTED pred=REFUTED: "There was an attempt to change Cyprus." | evidence: "This action precipitated the Turkish invasion of Cyprus on 20 July, which captured the present-day territory of Northern Cyprus and displaced over 150,000 Greek Cypriots and 50,000 Turkish Cypriots."
- claim_id=61922 true=REFUTED pred=SUPPORTED: "The American actor that plays Chumlee was born on December." | evidence: "Austin Lee Russell (born September 8, 1982), better known by his stage name Chumlee, is an American businessman and reality television personality, best known for his appearances on the History Channel television show Pawn Stars, which depicts day-to-day business at the Gold and Silver Pawn Shop in "
- claim_id=12118 true=SUPPORTED pred=REFUTED: "Yugoslavia was a country." | evidence: "Yugoslavia (; lit. Peter I was the country's first sovereign. The Constituent Assembly proclaimed Yugoslavia a federal republic on 29 November 1945, thus abolishing monarchical rule. This marked the onset of a four-decade long uncontested communist party rule of the country. In 1963, the country was"

