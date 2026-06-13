# Top-k Verifier Evaluation

- checkpoint: checkpoints/transformer_verifier_clean
- evidence_corpus_source: data/processed/evidence_corpus_large.jsonl
- sample_size: 650
- retrieval_runtime_seconds: 37.397
- best_top_k: 5

| setting | accuracy | macro_f1 | latency_ms_per_example |
| --- | ---: | ---: | ---: |
| oracle | 0.7154 | 0.7081 | 25.3836 |
| top_1 | 0.4431 | 0.4327 | 38.6077 |
| top_3 | 0.4569 | 0.4452 | 43.6859 |
| top_5 | 0.46 | 0.4538 | 43.6635 |

## Oracle vs. retrieved gap

- top_1: accuracy_gap=0.2723 macro_f1_gap=0.2754
- top_3: accuracy_gap=0.2585 macro_f1_gap=0.2629
- top_5: accuracy_gap=0.2554 macro_f1_gap=0.2543

## Top errors

- claim_id=35882 true=REFUTED pred=SUPPORTED: "Pearl (Steven Universe) exists as a magical dog." | evidence: "[E1] Pearl_-LRB-Steven_Universe-RRB-: Based on the gemstone pearl, she is a Gem, an alien being that exists as a magical gemstone projecting a holographic body.
[E2] The Bitterest Pills: The Troubling Story of Antipsychotic Drugs: © Joanna Moncrieff 2013. All rights reserved. A challenging reapprais"
- claim_id=88346 true=REFUTED pred=SUPPORTED: "Temple Grandin features Claire Danes playing a cartoon penguin." | evidence: "[E1] Temple_Grandin_-LRB-film-RRB-: Temple Grandin -LRB-film-RRB-
[E2] Temple_Grandin_-LRB-film-RRB-: Temple Grandin -LRB-film-RRB-
[E3] Temple_Grandin_-LRB-film-RRB-: Temple Grandin -LRB-film-RRB-
[E4] Temple_Grandin_-LRB-film-RRB-: Temple Grandin -LRB-film-RRB-
[E5] Temple_Grandin_-LRB-film-RRB-: "
- claim_id=145707 true=REFUTED pred=SUPPORTED: "The great white shark does not prefer to prey on humans because it's a herbivore." | evidence: "[E1] Great_white_shark: Great white shark
[E2] Lamniformes: It includes some of the most familiar species of sharks, such as the great white  and mako sharks as well as less familiar ones, such as the goblin shark and megamouth shark.
[E3] Humans and great apes share a large frontal cortex: Some of "
- claim_id=166925 true=SUPPORTED pred=NOT ENOUGH INFO: "Baadshah had at least three languages dubbed over it." | evidence: "[E1] Western_Iranian_languages: The Western Iranian languages or Western Iranic languages are a branch of the Iranian languages, attested from the time of Old Persian (6th century BC) and Median.
[E2] Contraceptive practices of women requesting termination of pregnancy : A study from China: Abstract"
- claim_id=115807 true=REFUTED pred=NOT ENOUGH INFO: "Susan Collins was the second woman to become the nominee of a major party for Governor of Maine." | evidence: "[E1] Tony_Blair: He became involved in the Labour Party and was elected to the House of Commons in 1983 for Sedgefield.
[E2] Tony_Blair: He became involved in the Labour Party and was elected to the House of Commons in 1983 for Sedgefield.
[E3] Tony_Blair: He became involved in the Labour Party and "
- claim_id=178230 true=SUPPORTED pred=REFUTED: "Nawaz Sharif is the 20th Prime Minister of Pakistan." | evidence: "[E1] Nawaz_Sharif: Mian Muhammad Nawaz Sharif (born 25 December 1949) is a Pakistani politician and businessman who served as the prime minister of Pakistan for three non-consecutive terms, first serving from 1990 to 1993, then from 1997 to 1999 and later from 2013 to 2017.
[E2] Narendra_Modi: Modi "
- claim_id=81480 true=REFUTED pred=SUPPORTED: "Matt Bomer was born in Spain." | evidence: "[E1] Matt_Bomer: Matt Bomer
[E2] Casas_de_Pedro_Barba: Casas de Pedro Barba, or simply Pedro Barba, is a small community of summer residences on the island of La Graciosa, Canary Islands, Spain.
[E3] Transformers-COLON-_The_Last_Knight: The film was directed by Michael Bay and written by Art Marcum,"
- claim_id=172302 true=SUPPORTED pred=REFUTED: "Scream 2 is a film that is categorized as a slasher." | evidence: "[E1] Scream_2: Scream 2 is a 1997 American slasher film directed by Wes Craven and written by Kevin Williamson.
[E2] Scream_-LRB-franchise-RRB-: Scream 3 (2000) received a more mixed response, as did Scream 4 (2011), Craven's final film; both were later reappraised and the fourth film was generally "
- claim_id=148171 true=SUPPORTED pred=REFUTED: "SummerSlam had no pre-show, but contested ten matches." | evidence: "[E1] SummerSlam_-LRB-2015-RRB-: Ten matches were contested at the event, with no match on the pre-show.
[E2] Royal_Rumble_-LRB-2002-RRB-: Six matches were contested at the event.
[E3] 2014_WTA_Finals: The tournament was held at the Singapore Indoor Stadium, and contested by eight singles players and"
- claim_id=51435 true=SUPPORTED pred=REFUTED: "There was an attempt to change Cyprus." | evidence: "[E1] Billy_Joel: Glass Houses (1980) was an attempt to further establish him as a rock artist; it featured "It's Still Rock and Roll to Me" (Joel's first single to top the Billboard Hot 100), "You May Be Right", "Don't Ask Me Why", and "Sometimes a Fantasy".
[E2] Validity and reliability of observat"
