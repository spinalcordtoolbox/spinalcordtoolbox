from spinalcordtoolbox.scripts import sct_dmri_denoise_patch2self


def test_sct_dmri_denoise_patch2self():
    sct_dmri_denoise_patch2self.main(argv=['-i', 'sct_testing_data/dmri/dmri.nii.gz',
                                           '-b', 'sct_testing_data/dmri/bvals.txt', '-v', '2'])