# Error Analysis

- Backend: transformers
- Top K: 3
- Error rate: 0.750

## Category Counts

| category | count |
| --- | --- |
| entity | 4 |
| negation | 1 |
| numerical | 1 |

## Examples

| claim | truth | prediction | category |
| --- | --- | --- | --- |
| There is a guest house in the Taj Mahal. | SUPPORTED | NOT ENOUGH INFO | entity |
| The Saw franchise is a collection of bugs. | SUPPORTED | NOT ENOUGH INFO | entity |
| The Book of Mormon won a Tony Award for Best Musical. | SUPPORTED | NOT ENOUGH INFO | entity |
| TNFAIP3 is a glioblastoma tumor suppressor. | REFUTED | NOT ENOUGH INFO | numerical |
| Broadly HIV-1 Neutralizing Antibodies (bnAb) 10EB have no affinity for phospholipids. | SUPPORTED | NOT ENOUGH INFO | negation |
| All hematopoietic stem cells segregate their chromosomes randomly. | SUPPORTED | NOT ENOUGH INFO | entity |
