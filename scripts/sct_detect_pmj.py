#!/usr/bin/env python

"""Detect Ponto-Medullary Junction.

Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
Author: Charley
Created: 2017-07-21
Modified: 2017-09-12

About the license: see the file LICENSE.TXT
"""

import os
import shutil
import sys
import numpy as np
from scipy.ndimage.measurements import center_of_mass
import nibabel as nib


from msct_image import Image
from msct_parser import Parser
import sct_utils as sct
from sct_image import get_orientation, set_orientation


def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Detection of the Ponto-Medullary Junction (PMJ).\n'
                                 ' This method is machine-learning based and adapted for T1w-like or T2w-like images.\n'
                                 ' If the PMJ is detected from the input image, a nifti mask is output ("*_pmj.nii.gz")\n'
                                 ' with one voxel (value=50) located at the predicted PMJ position.\n'
                                 ' If the PMJ is not detected, nothing is output.')
    parser.add_option(name="-i",
                        type_value="file",
                        description="input image.",
                        mandatory=True,
                        example="t2.nii.gz")
    parser.add_option(name="-c",
                        type_value="multiple_choice",
                        description="type of image contrast, if your contrast is not in the available options (t1, t2), use t1 (cord bright / CSF dark) or t2 (cord dark / CSF bright)",
                        mandatory=True,
                        example=["t1", "t2"])
    parser.add_option(name="-s",
                        type_value="file",
                        description="SC segmentation or centerline mask. Provide this mask helps the detection of the PMJ by indicating the position of the SC in the Right-to-Left direction.",
                        mandatory=False,
                        example="t2_seg.nii.gz")
    parser.add_option(name="-ofolder",
                        type_value="folder_creation",
                        description="Output folder",
                        mandatory=False,
                        example="My_Output_Folder/")
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=None)
    parser.add_option(name="-igt",
                      type_value="image_nifti",
                      description="File name of ground-truth PMJ (single voxel).",
                      mandatory=False)
    parser.add_option(name="-r",
                        type_value="multiple_choice",
                        description="Remove temporary files.",
                        mandatory=False,
                        default_value="1",
                        example=["0", "1"])
    parser.add_option(name="-v",
                        type_value='multiple_choice',
                        description="Verbose: 0 = nothing, 1 = classic, 2 = expended",
                        mandatory=False,
                        example=["0", "1", "2"],
                        default_value="1")

    return parser


class DetectPMJ:
    def __init__(self, fname_im, contrast, fname_seg, path_out, verbose):

        self.fname_im = fname_im
        self.contrast = contrast

        self.fname_seg = fname_seg

        self.path_out = path_out

        self.verbose = verbose

        self.tmp_dir = sct.tmp_create(verbose=self.verbose)  # path to tmp directory

        self.orientation_im = get_orientation(Image(self.fname_im))  # to re-orient the data at the end

        self.slice2D_im = sct.extract_fname(self.fname_im)[1] + '_midSag.nii'  # file used to do the detection, with only one slice
        self.dection_map_pmj = sct.extract_fname(self.fname_im)[1] + '_map_pmj'  # file resulting from the detection

        # path to the pmj detector
        path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
        self.pmj_model = os.path.join(path_sct, 'data', 'pmj_models', '{}_model'.format(self.contrast))

        self.threshold = -0.75 if self.contrast == 't1' else 0.8  # detection map threshold, depends on the contrast

        self.fname_out = sct.extract_fname(self.fname_im)[1] + '_pmj.nii.gz'

        self.fname_qc = 'qc_pmj.png'

    def apply(self):

        self.ifolder2tmp()  # move data to the temp dir

        self.orient2pir()  # orient data to PIR orientation

        self.extract_sagital_slice()  # extract a sagital slice, used to do the detection

        self.detect()  # run the detection

        self.get_max_position()  # get the max of the detection map

        self.generate_mask_pmj()  # generate the mask with one voxel (value = 50) at the predicted PMJ position

        fname_out2return = self.tmp2ofolder()  # save results to ofolder

        return fname_out2return, self.tmp_dir

    def tmp2ofolder(self):
        """Copy output files to the ofolder."""
        os.chdir(self.curdir)  # go back to original directory

        if self.pa_coord != -1:  # If PMJ has been detected
            sct.printv('\nSave resulting file...', self.verbose, 'normal')
            sct.copy(os.path.abspath(os.path.join(self.tmp_dir, self.fname_out)),
                        os.path.abspath(os.path.join(self.path_out, self.fname_out)))

            return os.path.join(self.path_out, self.fname_out)
        else:
            return None

    def generate_mask_pmj(self):
        """Output the PMJ mask."""
        if self.pa_coord != -1:  # If PMJ has been detected
            im = Image(''.join(sct.extract_fname(self.fname_im)[1:]))  # image in PIR orientation
            im_mask = im.copy()
            im_mask.data *= 0  # empty mask

            im_mask.data[self.pa_coord, self.is_coord, self.rl_coord] = 50  # voxel with value = 50

            im_mask.setFileName(self.fname_out)

            im_mask = set_orientation(im_mask, self.orientation_im, fname_out=self.fname_out)  # reorient data

            x_pmj, y_pmj, z_pmj = np.where(im_mask.data == 50)
            sct.printv('\tx_pmj = ' + str(x_pmj[0]), self.verbose, 'info')
            sct.printv('\ty_pmj = ' + str(y_pmj[0]), self.verbose, 'info')
            sct.printv('\tz_pmj = ' + str(z_pmj[0]), self.verbose, 'info')

            im_mask.save()

    def get_max_position(self):
        """Find the position of the PMJ by thresholding the probabilistic map."""
        img_pred = Image(self.dection_map_pmj)

        if True in np.unique(img_pred.data > self.threshold):  # threshold the detection map
            img_pred_maxValue = np.max(img_pred.data)  # get the max value of the detection map
            self.pa_coord = np.where(img_pred.data == img_pred_maxValue)[0][0]
            self.is_coord = np.where(img_pred.data == img_pred_maxValue)[1][0]

            sct.printv('\nPonto-Medullary Junction detected', self.verbose, 'normal')

        else:
            self.pa_coord, self.is_coord = -1, -1

            sct.printv('\nPonto-Medullary Junction not detected', self.verbose, 'normal')

        del img_pred

    def detect(self):
        """Run the classifier on self.slice2D_im."""
        sct.printv('\nRun PMJ detector', self.verbose, 'normal')
        os.environ["FSLOUTPUTTYPE"] = "NIFTI_PAIR"
        cmd_pmj = ['isct_spine_detect', self.pmj_model, self.slice2D_im.split('.nii')[0], self.dection_map_pmj]
        print(cmd_pmj)
        sct.run(cmd_pmj, verbose=0)

        img = nib.load(self.dection_map_pmj + '_svm.hdr')  # convert .img and .hdr files to .nii
        nib.save(img, self.dection_map_pmj + '.nii')

        self.dection_map_pmj += '.nii'  # fname of the resulting detection map

    def extract_sagital_slice(self):
        """Extract the sagital slice where the detection is done.

        If the segmentation is provided,
            the 2D sagital slice is choosen accoding to the segmentation.

        If the segmentation is not provided,
            the 2D sagital slice is choosen as the mid-sagital slice of the input image.
        """

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

        sct.run(['sct_crop_image', '-i', self.fname_im, '-start', str(self.rl_coord), '-end', str(self.rl_coord), '-dim', '2', '-o', self.slice2D_im])

    def orient2pir(self):
        """Orient input data to PIR orientation."""
        if self.orientation_im != 'PIR':  # open image and re-orient it to PIR if needed
            im_tmp = Image(self.fname_im)
            set_orientation(im_tmp, 'PIR', fname_out=''.join(sct.extract_fname(self.fname_im)[1:]))

            if self.fname_seg is not None:
                set_orientation(Image(self.fname_seg), 'PIR', fname_out=''.join(sct.extract_fname(self.fname_seg)[1:]))

    def ifolder2tmp(self):
        """Copy data to tmp folder."""
        if self.fname_im is not None:  # copy input image
            sct.copy(self.fname_im, self.tmp_dir)
            self.fname_im = ''.join(sct.extract_fname(self.fname_im)[1:])
        else:
            sct.printv('ERROR: No input image', self.verbose, 'error')

        if self.fname_seg is not None:  # copy segmentation image
            sct.copy(self.fname_seg, self.tmp_dir)
            self.fname_seg = ''.join(sct.extract_fname(self.fname_seg)[1:])

        self.curdir = os.getcwd()
        os.chdir(self.tmp_dir)  # go to tmp directory


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # Get parser
    parser = get_parser()
    arguments = parser.parse(args)

    # Set param arguments ad inputted by user
    fname_in = arguments["-i"]
    contrast = arguments["-c"]

    # Segmentation or Centerline line
    if '-s' in arguments:
        fname_seg = arguments['-s']
        if not os.path.isfile(fname_seg):
            fname_seg = None
            sct.printv('WARNING: -s input file: "' + arguments['-s'] + '" does not exist.\nDetecting PMJ without using segmentation information', 1, 'warning')
    else:
        fname_seg = None

    # Output Folder
    if '-ofolder' in arguments:
        path_results = arguments["-ofolder"]
        if not os.path.isdir(path_results) and os.path.exists(path_results):
            sct.printv("ERROR output directory %s is not a valid directory" % path_results, 1, 'error')
        if not os.path.exists(path_results):
            os.makedirs(path_results)
    else:
        path_results = '.'

    path_qc = arguments.get("-qc", None)

    # Remove temp folder
    rm_tmp = bool(int(arguments.get("-r", 1)))

    # Verbosity
    verbose = int(arguments.get("-v", 1))

    # Initialize DetectPMJ
    detector = DetectPMJ(fname_im=fname_in,
                            contrast=contrast,
                            fname_seg=fname_seg,
                            path_out=path_results,
                            verbose=verbose)

    # run the extraction
    fname_out, tmp_dir = detector.apply()

    # Remove tmp_dir
    if rm_tmp:
        sct.rmtree(tmp_dir)

    # View results
    if fname_out is not None:
        if path_qc is not None:
            generate_qc(fname_in, fname_out, args, os.path.abspath(path_qc))

        sct.display_viewer_syntax([fname_in, fname_out], colormaps=['gray', 'red'])


def generate_qc(fname_in, fname_out, args, path_qc):
    """
    Generate a QC entry allowing to quickly review the PMJ position
    """

    import spinalcordtoolbox.reports.qc as qc
    import spinalcordtoolbox.reports.slice as qcslice

    def highlight_pmj(self, mask):
        """
        Hook to show a rectangle where PMJ is on the slice
        """

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        y, x = np.where(mask == 50)

        ax = plt.gca()
        img = np.full_like(mask, np.nan)
        ax.imshow(img, cmap='gray', alpha=0)

        rect = patches.Rectangle((x - 10, y - 10),
                                    20, 20,
                                    linewidth=2,
                                    edgecolor='lime',
                                    facecolor='none')

        ax.add_patch(rect)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    qc.add_entry(
     src=fname_in,
     process="sct_detect_pmj",
     args=args,
     path_qc=path_qc,
     plane="Sagittal",
     qcslice=qcslice.Sagittal([Image(fname_in), Image(fname_out)]),
     qcslice_operations=[highlight_pmj],
     qcslice_layout=lambda x: x.single(),
    )


if __name__ == "__main__":
    sct.init_sct()
    main()
