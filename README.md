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

### Derin Öğrenme
- tensorflow
- keras
- tensorflow-datasets
=======
This repository contains experiments and tools for gum and tooth analysis.

## Segmentation Model

The gum segmentation model is implemented only in [`gummy_smile_guncel/gum_segmentation_guncel.py`](gummy_smile_guncel/gum_segmentation_guncel.py).

### Usage

Gum segmentation expects a dataset directory containing `Train` and `Test` folders. Specify the dataset location either via a command line argument or by editing `config.py`:

```bash
# Option 1: provide the path explicitly
python gummy_smile_guncel/gum_segmentation_guncel.py --root path/to/gummy_smile_guncel_dt

# Option 2: edit config.py and run without arguments
python gummy_smile_guncel/gum_segmentation_guncel.py
```

If a relative path is supplied, it is resolved with respect to the project root.

## Path-based utilities

`mask_bmp.py`, `json_to_yoloformat.py` ve `gummy_smile_project/sam2_segmentation.py` varsayılan olarak yolları `config.py` içindeki `DATASET_ROOT` değerinden türetir. İstenirse komut satırından farklı yollar verilebilir:

```bash
python mask_bmp.py --json-path path/to/formatted_file.json --output-dir path/to/masks
python json_to_yoloformat.py convert --json-path path/to/formatted_file.json --output-folder path/to/txt_files
python gummy_smile_project/sam2_segmentation.py --image path/to/image.jpg
```

Argümanlar sağlanmadığında örneklerdeki yollar `DATASET_ROOT` ile ilişkilendirilmiş varsayılan değerlerdir.


