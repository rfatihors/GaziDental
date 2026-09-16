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

### (a) corrected (lower gingiva edge -13 px at mask level (-0.77 mm at 16.84 px/mm), secondary)
observed agreement 80.6 %, linear-weighted κ 0.752 [0.661, 0.829]; images with zeroed columns: 78

| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 79 | 10 | 0 | 0 |
| E1-E2 | 3 | 18 | 3 | 0 |
| E2-E3 | 1 | 5 | 17 | 2 |
| E3 | 1 | 0 | 3 | 2 |

## (b) final model, test high images (n = 29) — secondary set
observed agreement 48.3 %, linear-weighted κ 0.401 [0.205, 0.588]

| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 11 | 5 | 2 | 0 |
| E1-E2 | 0 | 1 | 4 | 0 |
| E2-E3 | 0 | 1 | 2 | 2 |
| E3 | 0 | 1 | 0 | 0 |

### (b) corrected (lower gingiva edge -13 px at mask level (-0.77 mm at 16.84 px/mm), secondary)
observed agreement 75.9 %, linear-weighted κ 0.640 [0.373, 0.857]; images with zeroed columns: 17

| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 15 | 3 | 0 | 0 |
| E1-E2 | 0 | 4 | 1 | 0 |
| E2-E3 | 0 | 1 | 3 | 1 |
| E3 | 1 | 0 | 0 | 0 |

## GT masks, all reference images (Stage 3 geometry only)
observed agreement 78.6 %, linear-weighted κ 0.742 [0.660, 0.820]

| reference | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| E1 | 77 | 10 | 2 | 0 |
| E1-E2 | 3 | 15 | 6 | 0 |
| E2-E3 | 0 | 4 | 19 | 2 |
| E3 | 0 | 1 | 3 | 2 |
