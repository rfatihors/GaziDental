# Threshold-class agreement (Table 1 labels; a measurement check, not a clinical validation)

Rows = clinical reference label, columns = pipeline label. Overlapping bands give combined labels (E1-E2, E2-E3).

## (a) OOF masks, all reference images (n = 144) — primary
observed agreement 62.5 %, linear-weighted κ 0.594 [0.503, 0.677]

| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 60 | 24 | 5 | 0 |
| E1-E2 | 0 | 8 | 16 | 0 |
| E2-E3 | 0 | 1 | 19 | 5 |
| E3 | 0 | 1 | 2 | 3 |

## (b) final model, test high images (n = 29) — secondary
observed agreement 48.3 %, linear-weighted κ 0.401 [0.205, 0.588]

| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 11 | 5 | 2 | 0 |
| E1-E2 | 0 | 1 | 4 | 0 |
| E2-E3 | 0 | 1 | 2 | 2 |
| E3 | 0 | 1 | 0 | 0 |

## GT masks, all reference images (Stage 3 geometry only)
observed agreement 78.6 %, linear-weighted κ 0.742 [0.660, 0.820]

| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 77 | 10 | 2 | 0 |
| E1-E2 | 3 | 15 | 6 | 0 |
| E2-E3 | 0 | 4 | 19 | 2 |
| E3 | 0 | 1 | 3 | 2 |
