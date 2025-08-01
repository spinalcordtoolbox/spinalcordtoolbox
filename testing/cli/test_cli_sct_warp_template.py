# pytest unit tests for sct_warp_template

import logging

from spinalcordtoolbox.image import Image
from spinalcordtoolbox.scripts import sct_warp_template, sct_register_to_template
from spinalcordtoolbox.utils.sys import sct_test_path

logger = logging.getLogger(__name__)


def test_sct_warp_template_warp_small_PAM50(tmp_path_qc):
    """Warp the cropped, resampled version of the template from `sct_testing_data/template`."""
    sct_warp_template.main(argv=['-d', sct_test_path('mt', 'mt1.nii.gz'),
                                 '-w', sct_test_path('mt', 'warp_template2mt.nii.gz'),
                                 '-a', '0',  # -a is '1' by default, but atlas isn't present in 'template'
                                 '-t', sct_test_path('template'),
                                 '-qc', tmp_path_qc])


def test_sct_warp_template_warp_full_PAM50(tmp_path_qc):
    """Warp the full PAM50 template (i.e. the one that is downloaded to `data/PAM50` during installation)."""
    sct_warp_template.main(argv=['-d', sct_test_path('mt', 'mt1.nii.gz'),
                                 '-w', sct_test_path('mt', 'warp_template2mt.nii.gz'),
                                 '-a', '1', '-histo', '1',
                                 '-qc', tmp_path_qc])


def test_sct_warp_template_point_labels(tmp_path, tmp_path_qc):
    """Warp the full PAM50 template, then test whether point labels are preserved."""
    # Register the cropped T2 image to the full PAM50 template
    sct_register_to_template.main(argv=['-i', sct_test_path('t2', 't2.nii.gz'),
                                        '-s', sct_test_path('t2', 't2_seg-manual.nii.gz'),
                                        '-ldisc', sct_test_path('t2', 'labels.nii.gz'),
                                        '-ref', 'subject', '-ofolder', str(tmp_path),
                                        '-qc', tmp_path_qc])

    # Warp the PAM50 template to the cropped T2 image space
    path_out = tmp_path/"PAM50_warped"
    sct_warp_template.main(argv=['-d', sct_test_path('t2', 't2.nii.gz'),
                                 '-w', str(tmp_path/'warp_template2anat.nii.gz'),
                                 '-a', '0',  # -a is '1' by default, but save time since not needed for point labels
                                 '-qc', tmp_path_qc, '-ofolder', str(path_out)])

    # Ensure that point labels were preserved
    point_label_files = ['PAM50_label_discPosterior.nii.gz', 'PAM50_spinal_midpoint.nii.gz',
                         'PAM50_label_body.nii.gz', 'PAM50_label_disc.nii.gz']
    for label in point_label_files:
        coords = Image(str(path_out/'template'/label)).getNonZeroCoordinates()
        assert coords  # Ensure no labels were lost during warping
        assert all(c.value == int(c.value) for c in coords)  # Ensure point labels are equivalent to integers
