# Stage 1 — split summary

Seed 42; patient level (= image level after cleaning); stratified by smile-line group and, within `high`, by the reference class label; fixed test set.

| group | train | valid | test | total |
|---|---|---|---|---|
| high | 87 | 29 | 29 | 145 |
| low | 210 | 45 | 45 | 300 |
| normal | 549 | 118 | 118 | 785 |

Train-only images: 0; images forced to train by a train-only twin: 0.

## High group by reference label
| split | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| test | 18 | 5 | 5 | 1 |
| train | 53 | 14 | 16 | 4 |
| valid | 18 | 5 | 5 | 1 |

## 5-fold CV over the 145 measured high images
Fold sizes: {0: 29, 1: 29, 2: 29, 3: 29, 4: 29}

| cv_fold | E1 | E1-E2 | E2-E3 | E3 |
|---|---|---|---|---|
| 0 | 18 | 4 | 6 | 1 |
| 1 | 18 | 5 | 5 | 1 |
| 2 | 18 | 5 | 5 | 1 |
| 3 | 18 | 5 | 5 | 1 |
| 4 | 17 | 5 | 5 | 2 |

Original Roboflow split of the kept high test images (for the record): {'train': 22, 'valid': 5, 'test': 2}.
