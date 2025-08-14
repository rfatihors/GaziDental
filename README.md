# GaziDental


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


