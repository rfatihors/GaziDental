"""Shared training plumbing: environment log, argument resolution from config.yaml,
restartable runs (``runs/<name>/DONE``), artefacts copied to ``outputs/05_predictions``.

Ultralytics is imported lazily inside the functions that need it, so every script
works with ``--dry-run`` on a machine without torch/ultralytics.
"""
from __future__ import annotations

import datetime as dt
import gc
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from gsv4.config import resolve

ARTEFACTS = ("results.csv", "args.yaml")


def git_commit_hash(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def environment_info(root: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"time": dt.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit_hash(root)}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            info["cuda"] = torch.version.cuda
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"not available ({exc.__class__.__name__})"
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception as exc:  # noqa: BLE001
        info["ultralytics"] = f"not available ({exc.__class__.__name__})"
    return info


def resolve_train_args(cfg: Dict[str, Any], data_yaml: Path, name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ultralytics ``model.train`` keyword arguments from ``yolo.train`` in the config."""
    t = dict(cfg["yolo"]["train"])
    args: Dict[str, Any] = {
        "data": str(data_yaml), "epochs": int(t["epochs"]), "batch": int(t["batch"]), "imgsz": int(t["imgsz"]),
        "lr0": float(t["lr0"]), "lrf": float(t["lrf"]), "optimizer": str(t["optimizer"]), "cos_lr": bool(t["cos_lr"]),
        "close_mosaic": int(t["close_mosaic"]), "patience": int(t["patience"]), "cache": t.get("cache", False),
        "workers": int(t.get("workers", 8)), "seed": int(t.get("seed", cfg["seed"])), "deterministic": bool(t.get("deterministic", True)),
        "project": str(resolve(cfg, cfg["paths"]["runs"])), "name": name, "exist_ok": True, "plots": True, "verbose": True,
    }
    args.update(overrides or {})
    return args


def run_dir(cfg: Dict[str, Any], name: str) -> Path:
    return resolve(cfg, cfg["paths"]["runs"]) / name


def artefact_dir(cfg: Dict[str, Any], name: str) -> Path:
    return resolve(cfg, cfg["paths"]["predictions"]) / name


def is_done(cfg: Dict[str, Any], name: str) -> bool:
    return (run_dir(cfg, name) / "DONE").exists()


def mark_done(cfg: Dict[str, Any], name: str, payload: Optional[Dict[str, Any]] = None) -> None:
    d = run_dir(cfg, name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "DONE").write_text(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"), **(payload or {})}, indent=1), encoding="utf-8")


def write_run_header(cfg: Dict[str, Any], name: str, args: Dict[str, Any]) -> Path:
    """Environment + resolved arguments + a copy of config.yaml, in the run directory and
    in the artefact directory (commit_hash.txt next to args.yaml)."""
    root = cfg["_root"]
    info = environment_info(root)
    for d in (run_dir(cfg, name), artefact_dir(cfg, name)):
        d.mkdir(parents=True, exist_ok=True)
        (d / "environment.json").write_text(json.dumps({"environment": info, "train_args": args}, indent=1, default=str), encoding="utf-8")
        (d / "commit_hash.txt").write_text(info["git_commit"] + "\n", encoding="utf-8")
        shutil.copy(root / "configs" / "config.yaml", d / "config_used.yaml")
    print(f"[{name}] environment: " + ", ".join(f"{k}={v}" for k, v in info.items()))
    print(f"[{name}] train args: " + json.dumps(args, default=str))
    return run_dir(cfg, name)


def copy_artefacts(cfg: Dict[str, Any], name: str) -> List[Path]:
    src, dst = run_dir(cfg, name), artefact_dir(cfg, name)
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in ARTEFACTS:
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
            copied.append(dst / f)
    return copied


def check_paths(cfg: Dict[str, Any], data_yaml: Path) -> List[str]:
    """Dry-run validation: dataset yaml, list files and a sample of symlinks must exist."""
    problems: List[str] = []
    if not data_yaml.exists():
        return [f"missing data yaml: {data_yaml} (run gsv4/train/prepare_yolo_dataset.py first)"]
    d = yaml.safe_load(data_yaml.read_text())
    base = Path(d["path"])
    for key in ("train", "val", "test"):
        if key in d:
            lst = base / d[key]
            if not lst.exists():
                problems.append(f"missing list: {lst}")
                continue
            names = lst.read_text().splitlines()
            if not names:
                problems.append(f"empty list: {lst}")
            for n in names[:3]:
                if not (base / n).exists():
                    problems.append(f"broken image link: {base / n}")
                lbl = base / n.replace("images/", "labels/", 1)
                if not lbl.with_suffix(".txt").exists():
                    problems.append(f"missing label: {lbl.with_suffix('.txt')}")
    model = resolve(cfg, cfg["yolo"]["model"]) if "/" in str(cfg["yolo"]["model"]) else None
    if model is not None and not model.exists():
        problems.append(f"missing weights: {model}")
    return problems


def train_model(cfg: Dict[str, Any], data_yaml: Path, name: str, dry_run: bool = False, overrides: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """Train one model (restartable). Returns the path to best.pt (None in dry-run)."""
    args = resolve_train_args(cfg, data_yaml, name, overrides)
    problems = check_paths(cfg, data_yaml)
    if dry_run:
        print(f"[{name}] DRY RUN — model {cfg['yolo']['model']}; args: {json.dumps(args, default=str)}")
        print(f"[{name}] path check: {'OK' if not problems else problems}")
        return None
    if problems:
        raise SystemExit(f"[{name}] aborting, path problems: {problems}")
    if is_done(cfg, name):
        print(f"[{name}] already DONE, skipping")
        return run_dir(cfg, name) / "weights" / "best.pt"
    write_run_header(cfg, name, args)
    from ultralytics import YOLO  # lazy: requirements-train.txt

    last = run_dir(cfg, name) / "weights" / "last.pt"
    if last.exists():
        # interrupted training: continue from the last checkpoint with the saved arguments
        print(f"[{name}] resuming from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        model = YOLO(cfg["yolo"]["model"])  # yolo11x-seg.pt is downloaded by Ultralytics on first use (internet needed once)
        model.train(**args)
    copy_artefacts(cfg, name)
    best = run_dir(cfg, name) / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"[{name}] training ended without weights/best.pt — DONE not written")
    mark_done(cfg, name, {"best": str(best)})  # written by Python only, after a successful training
    return best


def release_cuda() -> None:
    """Drop unreachable objects and return cached CUDA blocks to the driver.

    Called between the validation and the prediction model in evaluate_test.py and after
    every prediction chunk in cv_predict.py. A no-op without torch / without a GPU.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def per_class_metrics(metrics: Any, names: Dict[int, str]) -> Dict[str, Dict[str, float]]:
    """Per-class box/mask P, R, mAP50, mAP50-95 and F1 from an Ultralytics validation result."""
    out: Dict[str, Dict[str, float]] = {}
    idx = list(getattr(metrics, "ap_class_index", []))
    for kind in ("box", "seg"):
        m = getattr(metrics, kind, None)
        if m is None:
            continue
        for pos, ci in enumerate(idx):
            cname = names.get(int(ci), str(ci))
            p, r = float(m.p[pos]), float(m.r[pos])
            out.setdefault(cname, {})[f"{kind}_precision"] = p
            out[cname][f"{kind}_recall"] = r
            out[cname][f"{kind}_f1"] = 2 * p * r / (p + r) if (p + r) else 0.0
            out[cname][f"{kind}_map50"] = float(m.ap50[pos])
            out[cname][f"{kind}_map50_95"] = float(m.ap[pos])
        out.setdefault("all", {})[f"{kind}_map50"] = float(m.map50)
        out["all"][f"{kind}_map50_95"] = float(m.map)
        out["all"][f"{kind}_precision"] = float(m.mp)
        out["all"][f"{kind}_recall"] = float(m.mr)
    return out
