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


# ---------------------------------------------------------------- shared instance core (architecture comparison)
class _Detections:
    """supervision.Detections-shaped stand-in: (K, H, W) boolean masks + one class id each."""

    def __init__(self, mask, class_id):
        self.mask = mask
        self.class_id = np.asarray(class_id)


def test_from_instances_unions_by_class_and_ignores_other_classes():
    from gsv4.masks.extract import from_instances

    g1 = np.zeros((40, 60), dtype=bool); g1[20:30, 5:25] = True
    g2 = np.zeros((40, 60), dtype=bool); g2[20:30, 35:55] = True
    lip = np.zeros((40, 60), dtype=bool); lip[5:18, :] = True
    other = np.ones((40, 60), dtype=bool)
    names = {0: "dudak-diseti", 1: "diseti", 2: "dudak"}
    cm = from_instances(np.stack([g1, g2, lip, other]), [1, 1, 2, 0], names, CLASS_NAMES, (40, 60), source="rfdetr")
    assert cm.n_gingiva_instances == 2 and cm.n_lip_instances == 1 and cm.source == "rfdetr"
    assert cm.gingiva.sum() == g1.sum() + g2.sum() and np.array_equal(cm.lip, lip)
    assert not cm.gingiva[0, 0] and cm.shape == (40, 60)


def test_from_instances_rejects_a_wrong_shape_and_unknown_class_names():
    from gsv4.masks.extract import from_instances

    names = {0: "dudak-diseti", 1: "diseti", 2: "dudak"}
    with pytest.raises(ValueError, match="mask shape"):
        from_instances(np.zeros((1, 20, 30), dtype=bool), [1], names, CLASS_NAMES, (40, 60))
    with pytest.raises(ValueError, match="do not contain"):
        from_instances(np.zeros((1, 40, 60), dtype=bool), [1], {0: "gum", 1: "lip"}, CLASS_NAMES, (40, 60))


def test_from_instances_accepts_float_masks_and_a_list():
    from gsv4.masks.extract import from_instances

    soft = np.zeros((40, 60), dtype=float); soft[10:20, 10:20] = 0.9; soft[25, 25] = 0.4
    cm = from_instances([soft], [1], {1: "diseti", 2: "dudak"}, CLASS_NAMES, (40, 60))
    assert cm.gingiva.sum() == 100 and cm.lip is None and cm.n_lip_instances == 0


def test_from_detections_matches_from_yolo_on_the_same_instances():
    from gsv4.masks.extract import from_detections

    r = _fake()                                   # retina masks, classes [1, 1, 2]
    names = r.names
    ref = from_yolo(r, class_names=CLASS_NAMES, image_shape=(40, 60))
    det = _Detections(np.asarray(r.masks.data.numpy()) > 0.5, [1, 1, 2])
    cm = from_detections(det, names, CLASS_NAMES, (40, 60), source="rfdetr")
    assert np.array_equal(cm.gingiva, ref.gingiva) and np.array_equal(cm.lip, ref.lip)
    assert cm.n_gingiva_instances == ref.n_gingiva_instances and cm.source == "rfdetr"


def test_from_detections_without_any_instance_gives_empty_masks():
    from gsv4.masks.extract import from_detections

    cm = from_detections(_Detections(None, []), {1: "diseti", 2: "dudak"}, CLASS_NAMES, (40, 60))
    assert cm.gingiva.sum() == 0 and cm.lip is None and cm.n_gingiva_instances == 0 and cm.source.endswith(":none")
