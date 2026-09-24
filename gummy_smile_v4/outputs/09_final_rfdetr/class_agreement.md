# Threshold-class agreement (Table 1 labels; a measurement check, not a clinical validation)

Rows = clinical reference label, columns = pipeline label. Overlapping bands give combined labels (E1-E2, E2-E3).

## (a) OOF masks, all reference images (n = 145) — primary
observed agreement 75.9 %, linear-weighted κ 0.718 [0.631, 0.797]

| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 75 | 13 | 1 | 0 |
| E1-E2 | 2 | 12 | 10 | 0 |
| E2-E3 | 1 | 2 | 21 | 2 |
| E3 | 0 | 1 | 3 | 2 |


## (b) final model, test high images (n = 29) — secondary set
observed agreement 69.0 %, linear-weighted κ 0.621 [0.421, 0.788]

| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 15 | 3 | 0 | 0 |
| E1-E2 | 0 | 2 | 3 | 0 |
| E2-E3 | 0 | 1 | 3 | 1 |
| E3 | 0 | 1 | 0 | 0 |


## GT masks, all reference images (Stage 3 geometry only)
observed agreement 78.6 %, linear-weighted κ 0.742 [0.660, 0.820]

| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 77 | 10 | 2 | 0 |
| E1-E2 | 3 | 15 | 6 | 0 |
| E2-E3 | 0 | 4 | 20 | 2 |
| E3 | 0 | 1 | 3 | 2 |
