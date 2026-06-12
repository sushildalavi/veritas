# Verifier Clean Baseline

- Checkpoint: `checkpoints/verifier_clean/model.joblib`
- Model: TF-IDF (1-2 grams) + LogisticRegression
- Classes: SUPPORTED, REFUTED, NOT_ENOUGH_INFO
- Sklearn version: 1.9.0
- Python version: 3.13.5
- Git commit: cb758630f1a1251291acf0f391784eb6f368dab9
- Training command: `python3 scripts/train_verifier_clean.py`

| split | examples | accuracy | macro_f1 |
| --- | --- | --- | --- |
| train | 2809 | 0.827 | 0.821 |
| validation | 650 | 0.454 | 0.444 |
| test | 650 | 0.486 | 0.484 |

## Test set per-class metrics

| label | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| SUPPORTED | 0.433 | 0.562 | 0.489 | 226 |
| REFUTED | 0.424 | 0.375 | 0.398 | 200 |
| NOT_ENOUGH_INFO | 0.633 | 0.509 | 0.564 | 224 |

## Test set confusion matrix

| true \ pred | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
| --- | --- | --- | --- |
| SUPPORTED | 127 | 66 | 33 |
| REFUTED | 92 | 75 | 33 |
| NOT_ENOUGH_INFO | 74 | 36 | 114 |

## Threshold check

- macro_f1 >= 0.55: FAIL (0.484)
- accuracy >= 0.60: FAIL (0.486)

## Top test errors

| claim_id | claim | evidence | true | predicted |
| --- | --- | --- | --- | --- |
| 35882 | Pearl (Steven Universe) exists as a magical dog. | Based on the gemstone pearl, she is a Gem, an alien being that exists as a magical gemstone projecting a holographic body. | REFUTED | SUPPORTED |
| 12084 | Acetylcholine prevents neuromodulation. | Some of the effects of neuromodulators include altering intrinsic firing activity, increasing or decreasing voltage-dependent currents, altering synaptic efficacy, increasing bursting activity and rec | REFUTED | SUPPORTED |
| 127777 | Exotic Birds achieved mostly local failure. | Exotic Birds | REFUTED | NOT_ENOUGH_INFO |
| 88346 | Temple Grandin features Claire Danes playing a cartoon penguin. | Temple Grandin -LRB-film-RRB- | REFUTED | SUPPORTED |
| 193837 | Ed Gagliardi died on October 13, 2005. | Edward John Gagliardi (February 13, 1952 – May 11, 2014) was an American bass guitarist, best known as the original bass player for the 1970s rock band Foreigner. | REFUTED | SUPPORTED |
| 148376 | Google Search is not the product of an American multinational technology company. | Google Search (also known simply as Google or google.com) is a search engine operated by Google. Google LLC ( , GOO-gəl) is an American multinational technology corporation focused on information tech | REFUTED | SUPPORTED |
| 115807 | Susan Collins was the second woman to become the nominee of a major party for Governor of Maine. | After her bid, she became the founding director of the Center for Family Business at Husson University in Bangor, Maine. | REFUTED | SUPPORTED |
| 81480 | Matt Bomer was born in Spain. | Matt Bomer | REFUTED | NOT_ENOUGH_INFO |
| 148171 | SummerSlam had no pre-show, but contested ten matches. | Ten matches were contested at the event, with no match on the pre-show. | SUPPORTED | REFUTED |
| 97474 | The Burj Khalifa contains zero escalators. | Burj Khalifa | REFUTED | NOT_ENOUGH_INFO |
| 61922 | The American actor that plays Chumlee was born on December. | Austin Lee Russell (born September 8, 1982), better known by his stage name Chumlee, is an American businessman and reality television personality, best known for his appearances on the History Channe | REFUTED | SUPPORTED |
| 106683 | The English Wikipedia is an edition of an expensive online encyclopedia. | English Wikipedia | REFUTED | NOT_ENOUGH_INFO |
| 148966 | The ability of individuals to connect to the internet is called Internet Access. | Internet access | SUPPORTED | NOT_ENOUGH_INFO |
| 53319 | The Big Country is a novel. | The Big Country is a 1958 American epic Western film directed by William Wyler, and starring Gregory Peck, Jean Simmons, Carroll Baker, Charlton Heston, and Charles Bickford. The supporting cast featu | REFUTED | SUPPORTED |
| 80625 | Daredevil follows Matt Murdock. | Marvel's Daredevil is an American television series created by Drew Goddard for the streaming service Netflix, based on the Marvel Comics character Daredevil. This puts him in conflict with many super | SUPPORTED | REFUTED |
| 183456 | Henry III of France was succeeded by Henry IV, founder of the House of Bourbon. | Henry III of France (French: Henri III, né Alexandre Édouard; 19 September 1551 – 2 August 1589) was King of France from 1574 until his assassination in 1589 and, as Henry of Valois (Polish: Henryk Wa | REFUTED | NOT_ENOUGH_INFO |
| 168075 | Jean-Jacques Dessalines was partially in charge of the Haitian Revolution. | Jean-Jacques Dessalines | SUPPORTED | NOT_ENOUGH_INFO |
| 129805 | Internet access avoids the use of computers. | Internet access | REFUTED | NOT_ENOUGH_INFO |
| 11269 | Legendary Entertainment and Wanda Cinemas are owned by the same entity. | Legendary Entertainment Wanda Group | SUPPORTED | NOT_ENOUGH_INFO |
| 228062 | Caleb McLaughlin was born on the day 14th. | Caleb Reginald McLaughlin (born October 13, 2001) is an American actor. Caleb Reginald McLaughlin (born October 13, 2001) is an American actor. | REFUTED | SUPPORTED |
