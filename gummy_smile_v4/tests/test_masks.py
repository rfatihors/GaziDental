"""Class-aware mask extraction: COCO polygons, PNG files and a fake Ultralytics result."""
import numpy as np
import pytest

from gsv4.masks.extract import ClassMasks, from_coco, from_png, from_yolo, rasterize_polygons, save_masks

CLASS_NAMES = {"gingiva": "diseti", "lip": "dudak"}
CAT_IDS = {"gingiva": 1, "lip": 2}


def _square(x0, y0, x1, y1):
    return [float(v) for v in (x0, y0, x1, y0, x1, y1, x0, y1)]


def test_rasterize_polygons_union_and_shape():
    m = rasterize_polygons([_square(10, 10, 20, 20), _square(30, 10, 40, 20)], (50, 60))
    assert m.shape == (50, 60) and m.dtype == bool
    assert m[15, 15] and m[15, 35] and not m[15, 25]


def test_from_coco_unions_all_instances_per_class():
    anns = [
        {"image_id": 7, "category_id": 1, "segmentation": [_square(10, 30, 30, 40)]},
        {"image_id": 7, "category_id": 1, "segmentation": [_square(50, 30, 70, 40)]},
        {"image_id": 7, "category_id": 2, "segmentation": [_square(0, 0, 100, 25)]},
        {"image_id": 8, "category_id": 1, "segmentation": [_square(0, 0, 5, 5)]},  # other image
    ]
    cm = from_coco(anns, image_id=7, shape=(60, 100), category_ids=CAT_IDS)
    assert cm.gingiva.shape == (60, 100) and cm.lip.shape == (60, 100)
    assert cm.n_gingiva_instances == 2 and cm.n_lip_instances == 1
    assert cm.gingiva[35, 20] and cm.gingiva[35, 60] and not cm.gingiva[35, 40]
    assert cm.lip[10, 50] and not cm.lip[35, 20]
    assert not cm.gingiva[2, 2]  # instance of image 8 ignored


def test_from_coco_without_lip_returns_none_lip():
    anns = [{"image_id": 1, "category_id": 1, "segmentation": [_square(0, 0, 10, 10)]}]
    cm = from_coco(anns, image_id=1, shape=(20, 20), category_ids=CAT_IDS)
    assert cm.lip is None and cm.n_lip_instances == 0


def test_save_and_reload_png(tmp_path):
    cm = ClassMasks(gingiva=np.eye(8, dtype=bool), lip=np.zeros((8, 8), dtype=bool), n_gingiva_instances=1, n_lip_instances=1)
    paths = save_masks(cm, tmp_path, "IMG_1")
    assert paths["gingiva"].name == "IMG_1_gingiva.png" and paths["lip"].name == "IMG_1_lip.png"
    back = from_png(paths["gingiva"], paths["lip"], expected_shape=(8, 8))
    assert np.array_equal(back.gingiva, cm.gingiva) and np.array_equal(back.lip, cm.lip)
    with pytest.raises(ValueError):
        from_png(paths["gingiva"], paths["lip"], expected_shape=(9, 8))
    back2 = from_png(paths["gingiva"], None, expected_shape=(8, 8))
    assert back2.lip is None


class _Arr:
    """Minimal tensor stand-in with .cpu().numpy() like torch."""

    def __init__(self, a):
        self._a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    @property
    def shape(self):
        return self._a.shape

    def __len__(self):
        return len(self._a)


class _Boxes:
    def __init__(self, cls):
        self.cls = _Arr(cls)


class _Masks:
    def __init__(self, data, xy):
        self.data = _Arr(data) if data is not None else None
        self.xy = xy


class FakeResult:
    def __init__(self, names, cls, data, xy, orig_shape):
        self.names = names
        self.boxes = _Boxes(cls)
        self.masks = _Masks(data, xy) if (data is not None or xy is not None) else None
        self.orig_shape = orig_shape


def _fake(orig=(40, 60), retina=True, names=None):
    names = names or {0: "dudak-diseti", 1: "diseti", 2: "dudak"}
    g1 = np.zeros(orig, dtype=np.uint8); g1[20:30, 5:25] = 1
    g2 = np.zeros(orig, dtype=np.uint8); g2[20:30, 35:55] = 1
    lip = np.zeros(orig, dtype=np.uint8); lip[5:18, 0:60] = 1
    xy = [
        np.array([[5, 20], [24, 20], [24, 29], [5, 29]], dtype=np.float32),
        np.array([[35, 20], [54, 20], [54, 29], [35, 29]], dtype=np.float32),
        np.array([[0, 5], [59, 5], [59, 17], [0, 17]], dtype=np.float32),
    ]
    data = np.stack([g1, g2, lip]) if retina else np.stack([m[::2, ::2] for m in (g1, g2, lip)])
    return FakeResult(names, [1, 1, 2], data, xy, orig)


def test_from_yolo_retina_masks_union_by_class():
    cm = from_yolo(_fake(), class_names=CLASS_NAMES, image_shape=(40, 60))
    assert cm.gingiva.shape == (40, 60)
    assert cm.n_gingiva_instances == 2 and cm.n_lip_instances == 1
    assert cm.gingiva[25, 10] and cm.gingiva[25, 40] and not cm.gingiva[25, 30]
    assert cm.lip[10, 30] and not cm.lip[25, 10]
    assert cm.source == "yolo:masks.data"


def test_from_yolo_falls_back_to_polygons_when_data_is_letterboxed():
    cm = from_yolo(_fake(retina=False), class_names=CLASS_NAMES, image_shape=(40, 60))
    assert cm.gingiva.shape == (40, 60) and cm.source == "yolo:masks.xy"
    assert cm.gingiva[25, 10] and cm.gingiva[25, 40] and not cm.gingiva[25, 30]
    assert cm.lip[10, 30]


def test_from_yolo_rejects_unexpected_class_names():
    with pytest.raises(ValueError):
        from_yolo(_fake(names={0: "x", 1: "gum", 2: "lip"}), class_names=CLASS_NAMES, image_shape=(40, 60))


def test_from_yolo_no_masks_gives_empty_masks():
    r = FakeResult({1: "diseti", 2: "dudak"}, [], None, None, (40, 60))
    cm = from_yolo(r, class_names=CLASS_NAMES, image_shape=(40, 60))
    assert cm.gingiva.shape == (40, 60) and not cm.gingiva.any() and cm.lip is None
    assert cm.n_gingiva_instances == 0


def test_from_yolo_shape_mismatch_is_an_error():
    with pytest.raises(ValueError):
        from_yolo(_fake(), class_names=CLASS_NAMES, image_shape=(41, 60))
