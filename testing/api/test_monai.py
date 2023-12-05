import os

from spinalcordtoolbox.image import Image, compute_dice
from spinalcordtoolbox.math import binarize
from spinalcordtoolbox.monai.run_inference_single_image import main
from spinalcordtoolbox.utils.sys import sct_test_path, __sct_dir__


def test_monai_inference(tmp_path):
    main(argv=[
        "--path-img", sct_test_path("t2", "t2.nii.gz"),
        "--chkp-path", os.path.join(__sct_dir__, "data", "softseg_models"),
        "--path-out", str(tmp_path)
    ])

    # Binarize the softseg prediction
    im_seg = Image(str(tmp_path / "t2_pred.nii.gz"))
    im_seg.data = binarize(im_seg.data, 0.5)

    # Compare the prediction with the manually segmented ground truth
    im_seg_manual = Image(sct_test_path("t2", "t2_seg-manual.nii.gz"))
    dice_segmentation = compute_dice(im_seg, im_seg_manual, mode='3d', zboundaries=False)
    assert dice_segmentation > 0.95
