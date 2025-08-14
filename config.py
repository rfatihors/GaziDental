from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent

# Base data directory
LOCAL_DIR = ROOT_DIR / "local"

# Commonly used subdirectories
ANNOTATION_DIR = LOCAL_DIR / "annotation"
MASKS_DIR = LOCAL_DIR / "masks"
TXT_FILES_DIR = LOCAL_DIR / "txt_files"
IMAGES_DIR = LOCAL_DIR / "images"
GUMMY_SMILE_GUNCEL_DT_DIR = LOCAL_DIR / "gummy_smile_guncel_dt"
