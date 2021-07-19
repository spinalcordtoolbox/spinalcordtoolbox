#!/usr/bin/env python

"""Detect Ponto-Medullary Junction.

Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
Author: Charley
Created: 2017-07-21
Modified: 2017-09-12

About the license: see the file LICENSE.TXT
"""

import os
import sys
import logging

from scipy.ndimage.measurements import center_of_mass
import nibabel as nib
import numpy as np

from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.math import correlation
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, run_proc, printv, __data_dir__, set_loglevel
from spinalcordtoolbox.utils.fs import tmp_create, extract_fname, copy, rmtree

logger = logging.getLogger(__name__)


def get_parser():
    parser = SCTArgumentParser(
        description='Detection of the Ponto-Medullary Junction (PMJ). '
                    ' This method is machine-learning based and adapted for T1w-like or '
                    ' T2w-like images. '
                    ' If the PMJ is detected from the input image, a nifti mask is output '
                    ' ("*_pmj.nii.gz") with one voxel (value=50) located at the predicted PMJ '
                    ' position. If the PMJ is not detected, nothing is output.'
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        required=True,
        metavar=Metavar.file,
        help='Input image. Example: t2.nii.gz',
    )
    mandatory.add_argument(
        "-c",
        choices=("t1", "t2"),
        required=True,
        help="Type of image contrast, if your contrast is not in the available options (t1, t2), "
             "use t1 (cord bright/ CSF dark) or t2 (cord dark / CSF bright)",
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-s",
        metavar=Metavar.file,
        help='SC segmentation or centerline mask. '
             'Provide this mask helps the detection of the PMJ by indicating the position of the SC '
             'in the Right-to-Left direction. Example: t2_seg.nii.gz',
        required=False)
    optional.add_argument(
        "-ofolder",
        metavar=Metavar.folder,
        help='Output folder. Example: My_Output_Folder/',
        action=ActionCreateFolder,
        required=False)
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help='Output filename. Example: pmj.nii.gz '),
    optional.add_argument(
        '-qc',
        metavar=Metavar.str,
        help='The path where the quality control generated content will be saved.',
        default=None)
    optional.add_argument(
        "-igt",
        metavar=Metavar.str,
        help="File name of ground-truth PMJ (single voxel).",
        required=False)
    optional.add_argument(
        "-r",
        type=int,
        help="Remove temporary files.",
        required=False,
        default=1,
        choices=(0, 1))
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")

    return parser


class DetectPMJ:
    def __init__(self, fname_im, contrast, fname_seg, path_out, verbose, fname_out):

        self.fname_im = fname_im
        self.contrast = contrast

        self.fname_seg = fname_seg

        self.path_out = path_out

        self.verbose = verbose

        self.tmp_dir = tmp_create()  # path to tmp directory

        self.orientation_im = Image(self.fname_im).orientation  # to re-orient the data at the end

        self.slice2D_im = extract_fname(self.fname_im)[1] + '_midSag.nii'  # file used to do the detection, with only one slice
        self.dection_map_pmj = extract_fname(self.fname_im)[1] + '_map_pmj'  # file resulting from the detection

        # path to the pmj detector
        self.pmj_model = os.path.join(__data_dir__, 'pmj_models', '{}_model'.format(self.contrast))

        self.threshold = -0.75 if self.contrast == 't1' else 0.8  # detection map threshold, depends on the contrast

        self.fname_out = fname_out

        self.fname_qc = 'qc_pmj.png'

    def apply(self):

        self.ifolder2tmp()  # move data to the temp dir

        self.orient2pir()  # orient data to PIR orientation

        self.extract_pmjsymmetrical_sagittal_slice()  # extracts slice based on PMJ symmetry point, contains a detection, but only used to select the ROI for correlation

        self.detect()  # run the detection

        self.get_max_position()  # get the max of the detection map

        self.generate_mask_pmj()  # generate the mask with one voxel (value = 50) at the predicted PMJ position

        fname_out2return = self.tmp2ofolder()  # save results to ofolder

        return fname_out2return, self.tmp_dir

    def tmp2ofolder(self):
        """Copy output files to the ofolder."""
        os.chdir(self.curdir)  # go back to original directory

        if self.pa_coord != -1:  # If PMJ has been detected
            printv('\nSave resulting file...', self.verbose, 'normal')
            copy(os.path.abspath(os.path.join(self.tmp_dir, self.fname_out)),
                 os.path.abspath(os.path.join(self.path_out, self.fname_out)))

            return os.path.join(self.path_out, self.fname_out)
        else:
            return None

    def generate_mask_pmj(self):
        """Output the PMJ mask."""
        if self.pa_coord != -1:  # If PMJ has been detected
            im = Image(''.join(extract_fname(self.fname_im)[1:]))  # image in PIR orientation
            im_mask = zeros_like(im)

            im_mask.data[self.pa_coord, self.is_coord, self.rl_coord] = 50  # voxel with value = 50

            im_mask.change_orientation(self.orientation_im).save(self.fname_out)

            x_pmj, y_pmj, z_pmj = np.where(im_mask.data == 50)
            printv('\tx_pmj = ' + str(x_pmj[0]), self.verbose, 'info')
            printv('\ty_pmj = ' + str(y_pmj[0]), self.verbose, 'info')
            printv('\tz_pmj = ' + str(z_pmj[0]), self.verbose, 'info')

    def get_max_position(self):
        """Find the position of the PMJ by thresholding the probabilistic map."""
        img_pred = Image(self.dection_map_pmj)

        if True in np.unique(img_pred.data > self.threshold):  # threshold the detection map
            img_pred_maxValue = np.max(img_pred.data)  # get the max value of the detection map
            self.pa_coord = np.where(img_pred.data == img_pred_maxValue)[0][0]
            self.is_coord = np.where(img_pred.data == img_pred_maxValue)[1][0]
            printv('\nPonto-Medullary Junction detected', self.verbose, 'normal')

        else:
            self.pa_coord, self.is_coord = -1, -1

            printv('\nPonto-Medullary Junction not detected', self.verbose, 'normal')

        del img_pred

    def detect(self):
        """Run the classifier on self.slice2D_im."""
        printv('\nRun PMJ detector', self.verbose, 'normal')
        os.environ["FSLOUTPUTTYPE"] = "NIFTI_PAIR"
        cmd_pmj = ['isct_spine_detect', self.pmj_model, self.slice2D_im.split('.nii')[0], self.dection_map_pmj]
        print(cmd_pmj)
        run_proc(cmd_pmj, verbose=0, is_sct_binary=True)

        img = nib.load(self.dection_map_pmj + '_svm.hdr')  # convert .img and .hdr files to .nii
        nib.save(img, self.dection_map_pmj + '.nii')

        self.dection_map_pmj += '.nii'  # fname of the resulting detection map

    def extract_sagittal_slice(self):
        """Extract the sagittal slice where the detection is done.

        If the segmentation is provided,
            the 2D sagittal slice is choosen accoding to the segmentation.

        If the segmentation is not provided,
            the 2D sagittal slice is choosen as the mid-sagittal slice of the input image.
        """
        # TODO: get the mean across multiple sagittal slices to reduce noise

        if self.fname_seg is not None:
            img_seg = Image(self.fname_seg)

            z_mid_slice = img_seg.data[:, int(img_seg.dim[1] / 2), :]
            if 1 in z_mid_slice:  # if SC segmentation available at this slice
                self.rl_coord = int(center_of_mass(z_mid_slice)[1])  # Right_left coordinate
            else:
                self.rl_coord = int(img_seg.dim[2] / 2)
            del img_seg

        else:
            img = Image(self.fname_im)
            self.rl_coord = int(img.dim[2] / 2)  # Right_left coordinate
            del img

        run_proc(['sct_crop_image', '-i', self.fname_im, '-zmin', str(self.rl_coord), '-zmax', str(self.rl_coord + 1), '-o', self.slice2D_im])

    def extract_pmjsymmetrical_sagittal_slice(self):
        """Extract a slice that is symmetrical about the estimated PMJ location."""
        # Here, detection is used just as a way to determine the ROI for the sliding window approach
        self.extract_sagittal_slice()
        self.detect()
        self.get_max_position()

        self.compute_cross_corr_3d()  # Updates self.rl_coord

        # Replace the mid-sagittal slice, to be used for the "main" PMJ detection
        run_proc(['sct_crop_image', '-i', self.fname_im, '-zmin', str(self.rl_coord), '-zmax', str(self.rl_coord + 1), '-o', self.slice2D_im])

    def orient2pir(self):
        """Orient input data to PIR orientation."""
        if self.orientation_im != 'PIR':  # open image and re-orient it to PIR if needed
            Image(self.fname_im).change_orientation("PIR").save(''.join(extract_fname(self.fname_im)[1:]))

            if self.fname_seg is not None:
                Image(self.fname_seg).change_orientation('PIR').save(''.join(extract_fname(self.fname_seg)[1:]))

    def ifolder2tmp(self):
        """Copy data to tmp folder."""
        if self.fname_im is not None:  # copy input image
            copy(self.fname_im, self.tmp_dir)
            self.fname_im = ''.join(extract_fname(self.fname_im)[1:])
        else:
            printv('ERROR: No input image', self.verbose, 'error')

        if self.fname_seg is not None:  # copy segmentation image
            copy(self.fname_seg, self.tmp_dir)
            self.fname_seg = ''.join(extract_fname(self.fname_seg)[1:])

        self.curdir = os.getcwd()
        os.chdir(self.tmp_dir)  # go to tmp directory

    def compute_cross_corr_3d(self, xrange=list(range(-10, 10)), xshift=10, yshift=10, zshift=10):
        """
        Compute cross-correlation between image and its mirror using a sliding window in R-L direction to find the image symmetry and adjust R-L coordinate.
        Use a sliding window of 20x20x20 mm by default.
        :param xrange:
        :param xshift:
        :param yshift:
        :param zshift:
        """
        img = Image(self.fname_im)  # img in PIR orientation
        ny, nz, nx, _, py, pz, px, _ = img.dim
        # initializations
        I_corr = np.zeros(len(xrange))
        allzeros = 0
        # current_z = 0
        ind_I = 0
        # Adjust parameters with physical dimensions
        xrange = [int(item//px) for item in xrange]
        xshift = int(xshift//px)
        yshift = int(yshift//py)
        zshift = int(zshift//pz)
        for ix in xrange:
            # if pattern extends towards left part of the image, then crop and pad with zeros
            if self.rl_coord + ix + 1 + xshift > nx:
                padding_size = self.rl_coord + ix + xshift + 1 - nx
                src = img.data[self.pa_coord - yshift:self.pa_coord + yshift + 1,
                               self.is_coord - zshift: self.is_coord + zshift + 1,
                               self.rl_coord + ix - xshift: self.rl_coord + ix + xshift + 1 - padding_size]
                src = np.pad(src, ((0, 0), (0, 0), (0, padding_size)), 'constant',
                             constant_values=0)
            # if pattern extends towards right part of the image, then crop and pad with zeros
            elif self.rl_coord + ix - xshift < 0:
                padding_size = abs(ix - xshift)
                src = img.data[self.pa_coord - yshift:self.pa_coord + yshift + 1,
                               self.is_coord - zshift: self.is_coord + zshift + 1,
                               self.rl_coord + ix - xshift + padding_size: self.rl_coord + ix + xshift + 1]
                src = np.pad(src, ((0, 0), (0, 0), (padding_size, 0)), 'constant',
                             constant_values=0)
            else:
                src = img.data[self.pa_coord - yshift:self.pa_coord + yshift + 1,
                               self.is_coord - zshift: self.is_coord + zshift + 1,
                               self.rl_coord + ix - xshift: self.rl_coord + xshift + ix + 1]
            target = src[:, :, ::-1]  # Mirror of src (in R-L direction)
            # convert to 1d
            src_1d = src.ravel()
            target_1d = target.ravel()
            # check if src_1d contains at least one non-zero value
            if (src_1d.size == target_1d.size) and np.any(src_1d):
                I_corr[ind_I] = correlation(src_1d, target_1d)
            else:
                allzeros = 1
            ind_I = ind_I + 1
        if allzeros:
            logger.warning('Data contained zero. We probably hit the edge of the image.')
        if np.any(I_corr):
            # if I_corr contains at least a non-zero value
            ind_peak = [i for i in range(len(I_corr)) if I_corr[i] == max(I_corr)][0]  # index of max along x
            logger.info('.. Peak found: x=%s (correlation = %s)', xrange[ind_peak], I_corr[ind_peak])
            # TODO (maybe) check if correlation is high enough compared to previous R-L coord
        else:
            # if I_corr contains only zeros
            logger.warning('Correlation vector only contains zeros.')
        # Change adjust rl_coord
        logger.info('R-L coordinate adjusted from %s to  %s)', self.rl_coord, self.rl_coord + xrange[ind_peak])
        self.rl_coord = self.rl_coord + xrange[ind_peak]


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    # Set param arguments ad inputted by user
    fname_in = arguments.i
    contrast = arguments.c

    # Segmentation or Centerline line
    if arguments.s is not None:
        fname_seg = arguments.s
        if not os.path.isfile(fname_seg):
            fname_seg = None
            printv('WARNING: -s input file: "' + arguments.s + '" does not exist.\nDetecting PMJ without using segmentation information', 1, 'warning')
    else:
        fname_seg = None

    # Output Folder
    if arguments.ofolder is not None:
        path_results = arguments.ofolder
        if not os.path.isdir(path_results) and os.path.exists(path_results):
            printv("ERROR output directory %s is not a valid directory" % path_results, 1, 'error')
        if not os.path.exists(path_results):
            os.makedirs(path_results)
    else:
        path_results = '.'
    if arguments.o is not None:
        fname_o = arguments.o
    else:
        fname_o = extract_fname(fname_in)[1] + '_pmj.nii.gz'

    path_qc = arguments.qc

    # Remove temp folder
    rm_tmp = bool(arguments.r)

    # Initialize DetectPMJ
    detector = DetectPMJ(fname_im=fname_in,
                         contrast=contrast,
                         fname_seg=fname_seg,
                         path_out=path_results,
                         verbose=verbose,
                         fname_out=fname_o)

    # run the extraction
    fname_out, tmp_dir = detector.apply()

    # Remove tmp_dir
    if rm_tmp:
        rmtree(tmp_dir)

    # View results
    if fname_out is not None:
        if path_qc is not None:
            from spinalcordtoolbox.reports.qc import generate_qc
            generate_qc(fname_in, fname_seg=fname_out, args=sys.argv[1:], path_qc=os.path.abspath(path_qc), process='sct_detect_pmj')

        display_viewer_syntax([fname_in, fname_out], colormaps=['gray', 'red'])


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
