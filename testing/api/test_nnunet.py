import os

import pytest

from spinalcordtoolbox.nnunet.run_inference_single_subject import main
from spinalcordtoolbox.utils.sys import sct_test_path, __sct_dir__


@pytest.mark.parametrize("pred_type", ["sc", "lesion"])
@pytest.mark.parametrize("model_path", ["sci_models", "rootlet_models"])
def test_nnunet_inference(pred_type, model_path, tmp_path):
    main(argv=[
        "-i", sct_test_path("t2", "t2.nii.gz"),
        "-pred-type", pred_type,
        "-path-model", os.path.join(__sct_dir__, "data", model_path),
        "-o", str(tmp_path)
    ])

    assert (tmp_path / "t2_pred.nii.gz").exists()
