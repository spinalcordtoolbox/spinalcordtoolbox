from spinalcordtoolbox.utils.sys import sct_test_path
from spinalcordtoolbox.image import Image

from spinalcordtoolbox.scripts import sct_compute_flow


def test_sct_compute_flow(tmp_path):
    venc = 20
    fname_in = sct_test_path("qmri", "venc.nii.gz")
    fname_out = str(tmp_path / "venc_velocity.nii.gz")
    sct_compute_flow.main(argv=['-i', fname_in, "-venc", str(venc), '-o', fname_out])

    im_out = Image(fname_out)
    assert -venc <= im_out.data.min() < im_out.data.max() <= venc
