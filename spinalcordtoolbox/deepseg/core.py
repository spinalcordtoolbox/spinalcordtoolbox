# coding: utf-8
# Interface API for the deepseg module, which performs segmentation using deep learning with the ivadomed package.


import logging
import nibabel as nib

from ivadomed.utils import segment_volume

from sct_utils import add_suffix

logger = logging.getLogger(__name__)


class ParamDeepseg:
    """
    Parameters for deepseg module.
    """
    def __init__(self):
        self.is_diffusion = is_diffusion
        self.debug = 0
        self.fname_data = ''
        self.fname_bvecs = ''
        self.fname_bvals = ''
        self.fname_target = ''
        self.fname_mask = ''
        self.path_out = ''
        self.mat_final = ''
        self.todo = ''
        self.group_size = group_size
        self.spline_fitting = 0
        self.remove_temp_files = 1
        self.verbose = 1
        self.plot_graph = 0
        self.suffix = '_moco'
        self.poly = '2'  # degree of polynomial function for moco
        self.smooth = smooth
        self.gradStep = '1'  # gradientStep for searching algorithm
        self.iter = '10'  # number of iterations
        self.metric = metric
        self.sampling = '0.2'  # sampling rate used for registration metric
        self.interp = 'spline'  # nn, linear, spline
        self.run_eddy = 0
        self.mat_eddy = ''
        self.min_norm = 0.001
        self.swapXY = 0
        self.num_target = '0'
        self.suffix_mat = None  # '0GenericAffine.mat' or 'Warp.nii.gz' depending which transfo algo is used
        self.bval_min = 100  # in case user does not have min bvalues at 0, set threshold (where csf disapeared).
        self.iterAvg = 1  # iteratively average target image for more robust moco
        self.is_sagittal = False  # if True, then split along Z (right-left) and register each 2D slice (vs. 3D volume)
        self.output_motion_param = True  # if True, the motion parameters are outputted


def segment_nifti(fname_image, folder_model):
    """
    Segment a nifti file.

    :param fname_image: str: Filename of the image to segment.
    :param param_deepseg: class ParamDeepseg: Segmentation parameters.
    :return: fname_out: str: Output filename.
    """
    nii_seg = segment_volume(folder_model, fname_image)

    # TODO: use args to get output name
    fname_out = add_suffix(fname_image, '_seg')
    nib.save(nii_seg, fname_out)
    return fname_out
