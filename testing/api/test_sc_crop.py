from types import SimpleNamespace

import numpy as np
from ultralytics.models.yolo.detect.predict import DetectionPredictor

from spinalcordtoolbox.sc_crop.crop import infer_slices

# These values describe the shipped detector and are kept local so this regression
# test can be committed and run before the corresponding sc_crop fix.
DETECTOR_IMGSZ = 320
DETECTOR_MIN_SIDE = 64
DETECTOR_TOPK = 300
DETECTOR_STRIDES = (8, 16, 32)
DETECTOR_STRIDE = max(DETECTOR_STRIDES)


class FakeDetector:
    """Exercise detector preprocessing and simulate the exported model's fixed TopK."""

    def __init__(self):
        self.preprocessed = []

    def predict(self, source, predictor=None, **kwargs):
        predictor = predictor or DetectionPredictor
        predictor = predictor.__new__(predictor)
        predictor.args = SimpleNamespace(rect=True)
        predictor.model = SimpleNamespace(format="onnx", dynamic=True, stride=DETECTOR_STRIDE)
        predictor.imgsz = (kwargs["imgsz"], kwargs["imgsz"])
        self.preprocessed = predictor.pre_transform(source)

        for image in self.preprocessed:
            anchors = sum((image.shape[0] // stride) * (image.shape[1] // stride)
                          for stride in DETECTOR_STRIDES)
            if anchors < DETECTOR_TOPK:
                raise RuntimeError(
                    f"k argument [{DETECTOR_TOPK}] should not be greater than specified "
                    f"axis dim value [{anchors}]"
                )
        return [SimpleNamespace(boxes=None) for _ in source]


def test_infer_slices_prevents_topk_error_for_anisotropic_input():
    """Reproduce infer_slices' ONNX TopK crash (#5303) from a thin input yielding too few detector anchors."""
    model = FakeDetector()
    image = np.zeros((60, 5, 3), dtype=np.uint8)

    assert infer_slices(
        model, [image], [0], conf_thresh=0.1, imgsz=DETECTOR_IMGSZ
    ) == {}

    assert model.preprocessed[0].shape == (DETECTOR_IMGSZ, DETECTOR_MIN_SIDE, 3)
    anchors = sum((model.preprocessed[0].shape[0] // stride) * (model.preprocessed[0].shape[1] // stride)
                  for stride in DETECTOR_STRIDES)
    assert anchors >= DETECTOR_TOPK
