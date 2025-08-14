# GaziDental


## Intra-Observer Analysis

Scripts in `intra_observer` calculate intra-observer consistency using two calibration Excel files.

### Example usage

```bash
python intra_observer/intra_observer.py intra_observer/calibration-first.xlsx intra_observer/calibration-last.xlsx
python intra_observer/intra-observer-eng.py intra_observer/calibration-first.xlsx intra_observer/calibration-last.xlsx
```

=======

Bu depo, diş hekimliği alanında segmentasyon, regresyon ve istatistik analizleri için geliştirilen çalışmaları içermektedir.

## Ortam Kurulumu

Projeyi çalıştırmak için aşağıdaki adımları izleyin:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows için: .venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanılan Paketler

### Segmentasyon
- torch
- torchvision
- segmentation-models-pytorch
- albumentations
- opencv-python
- numpy
- matplotlib
- tqdm

### Regresyon
- xgboost
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib

### İstatistik
- pingouin
- pandas
- numpy
- scipy
=======
This repository contains experiments and tools for gum and tooth analysis.

## Segmentation Model

The gum segmentation model is implemented only in [`gummy_smile_guncel/gum_segmentation_guncel.py`](gummy_smile_guncel/gum_segmentation_guncel.py).


