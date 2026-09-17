#!/usr/bin/env bash
# RF-DETR-Seg: isolated install and compatibility check (outputs/08_architecture/PROTOCOL.md §9).
#
#   bash scripts/rfdetr_setup.sh            # create .venv-rfdetr, install, report
#   bash scripts/rfdetr_setup.sh --check    # report only, no install
#
# A SEPARATE virtualenv is mandatory. rfdetr pulls transformers >= 5.1, pytorch_lightning >= 2.6
# and pins torch-hungarian == 0.1.0rc0; installing it into the training environment risks the
# resolver changing torch or torchvision under the model of record. Nothing here touches
# .venv-train. Every step prints what it found, and a failure is a result, not a crash: it is
# written into outputs/08_architecture/rfdetr_environment.json and reported.
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv-rfdetr}
OUT=outputs/08_architecture
mkdir -p "$OUT"
PY=${PY:-python3.11}

report() { echo "[rfdetr-setup] $*"; }

if [ "${1:-}" != "--check" ]; then
  report "creating $VENV with $PY"
  "$PY" -m venv "$VENV" || { report "FAILED to create the virtualenv with $PY"; exit 1; }
  "$VENV/bin/pip" install -q --upgrade pip
  report "installing rfdetr[train] (this resolves torch itself; the training venv is untouched)"
  "$VENV/bin/pip" install "rfdetr[train]" opencv-python-headless pandas 2>&1 | tail -20
  rc=${PIPESTATUS[0]}
  report "pip exit code: $rc"
fi

"$VENV/bin/python" - <<'PYEOF'
import json, os, platform, sys
from pathlib import Path

info = {"python": sys.version.split()[0], "platform": platform.platform()}
def probe(name, fn):
    try:
        info[name] = fn()
    except Exception as exc:
        info[name] = f"FAILED: {type(exc).__name__}: {exc}"

def torch_info():
    import torch
    d = {"version": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        d["gpu"] = torch.cuda.get_device_name(0)
        d["capability"] = list(torch.cuda.get_device_capability(0))
    return d

probe("torch", torch_info)
probe("torchvision", lambda: __import__("torchvision").__version__)
probe("transformers", lambda: __import__("transformers").__version__)
probe("numpy", lambda: __import__("numpy").__version__)
probe("pytorch_lightning", lambda: __import__("pytorch_lightning").__version__)
probe("rfdetr", lambda: __import__("rfdetr").__version__ if hasattr(__import__("rfdetr"), "__version__") else "imported")
probe("segmentation_variants", lambda: [n for n in dir(__import__("rfdetr")) if n.startswith("RFDETRSeg")])
def build():
    from rfdetr import RFDETRSegLarge
    m = RFDETRSegLarge()
    return {"resolution": getattr(m, "resolution", None), "built": True}
probe("build_RFDETRSegLarge", build)

# torch-hungarian is declared by rfdetr[train] but is never imported by rfdetr itself (a grep over
# the package finds no reference), and its import name differs from the distribution name. It is
# reported for the record and never gates `usable`, which a hard check on it did incorrectly.
soft = {"torch_hungarian_distribution"}
def _dist(name):
    from importlib.metadata import version
    return version(name)
probe("torch_hungarian_distribution", lambda: _dist("torch-hungarian"))

ok = all(not (isinstance(v, str) and v.startswith("FAILED")) for k, v in info.items() if k not in soft)
info["usable"] = ok
info["soft_checks"] = sorted(soft)
out = Path("outputs/08_architecture/rfdetr_environment.json")
out.write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
print(json.dumps(info, indent=1, default=str))
print(("[rfdetr-setup] OK: the environment builds an RF-DETR-Seg model." if ok else
       "[rfdetr-setup] NOT USABLE: see the FAILED entries above. Record this in PROTOCOL.md as an "
       "amendment and report the architecture as not evaluable under controlled conditions."))
PYEOF
