"""Chunked, streaming prediction (cv_predict.predict_list) on a fake model: every chunk goes
through model.predict(stream=True, batch=n), masks land on disk per result, the model is
released, and a CUDA OOM turns into an actionable message (predict_batch)."""
import numpy as np
import pandas as pd
import pytest

from gsv4.train import cv_predict
from gsv4.train.cv_predict import chunked, is_cuda_oom, oom_message, predict_batch_size, predict_list
from tests.test_masks import CLASS_NAMES, _Arr, _fake


def test_chunked_preserves_order_and_covers_everything():
    assert list(chunked(list(range(10)), 4)) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert list(chunked([], 4)) == []
    assert list(chunked([1, 2], 8)) == [[1, 2]]
    with pytest.raises(ValueError):
        list(chunked([1], 0))


def test_predict_batch_size_cli_overrides_config_and_defaults_to_8():
    assert predict_batch_size({"yolo": {"predict_batch": 4}}) == 4
    assert predict_batch_size({"yolo": {"predict_batch": 4}}, 2) == 2
    assert predict_batch_size({"yolo": {}}) == 8
    with pytest.raises(ValueError):
        predict_batch_size({"yolo": {"predict_batch": 0}})
    with pytest.raises(ValueError):
        predict_batch_size({"yolo": {"predict_batch": "many"}})


def test_is_cuda_oom_by_class_name_or_message():
    class OutOfMemoryError(RuntimeError):  # torch.cuda.OutOfMemoryError stand-in
        pass

    assert is_cuda_oom(OutOfMemoryError("CUDA out of memory. Tried to allocate 7.20 GiB"))
    assert is_cuda_oom(RuntimeError("CUDA error: out of memory"))
    assert not is_cuda_oom(RuntimeError("shape mismatch"))
    assert not is_cuda_oom(ValueError("out of memory"))
    msg = oom_message("test", 3, 24, 8, 640)
    assert "predict_batch değerini düşürün" in msg and "3/24" in msg and "--batch" in msg


class _Boxes:
    """Ultralytics Boxes stand-in: cls + conf tensors and a length."""

    def __init__(self, cls, conf):
        self.cls, self.conf = _Arr(cls), _Arr(conf)

    def __len__(self):
        return len(self.cls)


class _Model:
    """Fake Ultralytics model: records every predict() call, yields one FakeResult per path."""

    def __init__(self, fail_on_call=None):
        self.calls = []
        self.fail_on_call = fail_on_call

    def predict(self, source, **kwargs):
        self.calls.append((list(source), kwargs))
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("CUDA out of memory. Tried to allocate 7.20 GiB")
        assert kwargs["stream"] is True

        def gen():
            for _ in source:
                r = _fake()
                r.boxes = _Boxes([1, 1, 2], [0.9, 0.8, 0.7])
                yield r

        return gen()


def _setup(tmp_path, n_images=10):
    ds = tmp_path / "yolo"; (ds / "images").mkdir(parents=True)
    names = [f"images/high__IMG_{i}.jpg" for i in range(n_images)]
    pd.DataFrame({"yolo_name": [f"high__IMG_{i}.jpg" for i in range(n_images)], "image": [f"IMG_{i}" for i in range(n_images)]}).to_csv(ds / "yolo_index.csv", index=False)
    lst = ds / "list.txt"; lst.write_text("\n".join(names) + "\n")
    cfg = {"_root": tmp_path, "paths": {"yolo_dataset": str(ds)}, "class_names": CLASS_NAMES,
           "yolo": {"imgsz": 640, "conf": 0.25, "iou": 0.5, "max_det": 20, "retina_masks": True, "predict_batch": 4}}
    return cfg, lst, tmp_path / "out"


def test_predict_list_streams_in_chunks_and_writes_masks(tmp_path, monkeypatch):
    cfg, lst, out = _setup(tmp_path, 10)
    model = _Model()
    released = []
    monkeypatch.setattr(cv_predict, "release_cuda", lambda: released.append(1))
    df = predict_list(cfg, tmp_path / "best.pt", lst, out, "fold0", {"fold": 0}, model=model)
    assert len(df) == 10 and list(df["image"]) == [f"IMG_{i}" for i in range(10)]
    assert [len(c[0]) for c in model.calls] == [4, 4, 2]                         # config predict_batch = 4
    assert all(k["batch"] == 4 and k["stream"] is True and k["retina_masks"] is True for _, k in model.calls)
    assert all(k["imgsz"] == 640 and k["max_det"] == 20 for _, k in model.calls)
    assert (out / "IMG_9_gingiva.png").exists() and (out / "IMG_0_lip.png").exists()
    assert set(df["mask_source"]) == {"yolo:masks.data"} and df["max_conf"].iloc[0] == pytest.approx(0.9)
    assert len(released) >= 3 + 1                                                # after every chunk + final model release
    assert not (out / "fold0_predictions.partial.csv").exists()                  # partial file removed on success
    assert "fold" in df.columns and set(df["fold"]) == {0}


def test_predict_list_batch_argument_overrides_config(tmp_path, monkeypatch):
    cfg, lst, out = _setup(tmp_path, 5)
    model = _Model()
    monkeypatch.setattr(cv_predict, "release_cuda", lambda: None)
    predict_list(cfg, tmp_path / "best.pt", lst, out, "t", {}, batch=2, model=model)
    assert [len(c[0]) for c in model.calls] == [2, 2, 1] and model.calls[0][1]["batch"] == 2


def test_predict_list_oom_gives_actionable_message_and_keeps_partial_results(tmp_path, monkeypatch):
    cfg, lst, out = _setup(tmp_path, 10)
    model = _Model(fail_on_call=2)
    monkeypatch.setattr(cv_predict, "release_cuda", lambda: None)
    with pytest.raises(SystemExit) as ei:
        predict_list(cfg, tmp_path / "best.pt", lst, out, "test", {}, model=model)
    msg = str(ei.value)
    assert "predict_batch değerini düşürün" in msg and "chunk 2/3" in msg and "predict_batch=4" in msg
    assert (out / "IMG_3_gingiva.png").exists() and not (out / "IMG_4_gingiva.png").exists()
    partial = pd.read_csv(out / "test_predictions.partial.csv")
    assert len(partial) == 4


def test_predict_list_non_oom_errors_propagate(tmp_path, monkeypatch):
    cfg, lst, out = _setup(tmp_path, 3)

    class Bad(_Model):
        def predict(self, source, **kwargs):
            raise ValueError("something else")

    monkeypatch.setattr(cv_predict, "release_cuda", lambda: None)
    with pytest.raises(ValueError, match="something else"):
        predict_list(cfg, tmp_path / "best.pt", lst, out, "t", {}, model=Bad())


def test_release_cuda_is_a_noop_without_torch():
    from gsv4.train.common import release_cuda

    release_cuda()  # must not raise on a machine without torch / GPU
