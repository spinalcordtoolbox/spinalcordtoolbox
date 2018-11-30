#!/usr/bin/env python
#########################################################################################
#
# Detect vertebral levels from centerline.
# Tips to run the function with init txt file as input:
# sct_label_vertebrae -i t2.nii.gz -s t2_seg_manual.nii.gz  "$(< init_label_vertebrae.txt)" -v 2
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Eugenie Ullmann, Karun Raju, Tanguy Duval, Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: write label C2-C3 when user uses viewer
# TODO: find automatically if -c =t1 or t2 (using dilated seg)
# TODO: address the case when there is more than one max correlation

from __future__ import division, absolute_import

import sys, io, os

import numpy as np
import scipy.ndimage.measurements

from sct_maths import mutual_information
from msct_parser import Parser
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image
import sct_utils as sct
from spinalcordtoolbox.metadata import get_file_label
from spinalcordtoolbox.vertebrae.detect_c2c3 import detect_c2c3
from spinalcordtoolbox.image import Image

# get path of SCT
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))


def center_of_mass(x):
    """
    :return: array center of mass
    """
    if (x == 0).all():
        raise ValueError("Array has no mass")
    return scipy.ndimage.measurements.center_of_mass(x)


# PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        # self.path_template = os.path.join(path_sct, 'data', 'template')
        self.shift_AP_initc2 = 35
        self.size_AP_initc2 = 9  # 15
        self.shift_IS_initc2 = 15  # 15
        self.size_IS_initc2 = 30  # 30
        self.size_RL_initc2 = 1
        self.shift_AP = 32  # 0#32  # shift the centerline towards the spine (in voxel).
        self.size_AP = 11  # 41#11  # window size in AP direction (=y) (in voxel)
        self.size_RL = 1  # 1 # window size in RL direction (=x) (in voxel)
        self.size_IS = 19  # window size in IS direction (=z) (in voxel)
        self.shift_AP_visu = 15  # 0#15  # shift AP for displaying disc values
        self.smooth_factor = [3, 1, 1]  # [3, 1, 1]
        self.gaussian_std = 1.0  # STD of the Gaussian function, centered at the most rostral point of the image, and used to weight C2-C3 disk location finding towards the rostral portion of the FOV. Values to set between 0.1 (strong weighting) and 999 (no weighting).
        self.path_qc = None

    # update constructor with user's parameters
    def update(self, param_user):
        list_objects = param_user.split(',')
        for object in list_objects:
            if len(object) < 2:
                sct.printv('ERROR: Wrong usage.', 1, type='error')
            obj = object.split('=')
            if obj[0] == 'gaussian_std':
                setattr(self, obj[0], float(obj[1]))
            else:
                setattr(self, obj[0], int(obj[1]))


# PARSER
# ==========================================================================================
def get_parser():
    # initialize default param
    param_default = Param()
    # parser initialisation
    parser = Parser(__file__)
    parser.usage.set_description('''This function takes an anatomical image and its cord segmentation (binary file), and outputs the cord segmentation labeled with vertebral level. The algorithm requires an initialization (first disc) and then performs a disc search in the superior, then inferior direction, using template disc matching based on mutual information score.
Tips: To run the function with init txt file that includes flags -initz/-initcenter:
sct_label_vertebrae -i t2.nii.gz -s t2_seg_manual.nii.gz  "$(< init_label_vertebrae.txt)"
''')
    parser.add_option(name="-i",
                      type_value="file",
                      description="input image.",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name="-s",
                      type_value="file",
                      description="Segmentation or centerline of the spinal cord.",
                      mandatory=True,
                      example="t2_seg.nii.gz")
    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark",
                      mandatory=True,
                      example=['t1', 't2'])
    parser.add_option(name="-t",
                      type_value="folder",
                      description="Path to template.",
                      mandatory=False,
                      default_value=os.path.join(path_sct, "data", "PAM50"))
    parser.add_option(name="-initz",
                      type_value=[[','], 'int'],
                      description='Initialize using slice number and disc value. Example: 68,3 (slice 68 corresponds to disc C3/C4). WARNING: Slice number should correspond to superior-inferior direction (e.g. Z in RPI orientation, but Y in LIP orientation).',
                      mandatory=False,
                      example=['125,3'])
    parser.add_option(name="-initcenter",
                      type_value='int',
                      description='Initialize using disc value centered in the rostro-caudal direction. If the spine is curved, then consider the disc that projects onto the cord at the center of the z-FOV',
                      mandatory=False)
    parser.add_option(name="-initc2",
                      type_value=None,
                      description='Initialize by clicking on C2-C3 disc using interactive window.',
                      mandatory=False)
    parser.add_option(name="-initfile",
                      type_value='file',
                      description='Initialize labeling by providing a text file which includes either -initz or -initcenter flag.',
                      mandatory=False)
    parser.add_option(name="-initlabel",
                      type_value='file',
                      description='Initialize vertebral labeling by providing a nifti file that has a single disc label. An example of such file is a single voxel with value "3", which would be located at the posterior tip of C2-C3 disc. Such label file can be created using: sct_label_utils -i IMAGE_REF -create-viewer 3',
                      mandatory=False)
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder.",
                      mandatory=False,
                      default_value='')
    parser.add_option(name="-denoise",
                      type_value="multiple_choice",
                      description="Apply denoising filter to the data. Sometimes denoising is too aggressive, so use with care.",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1'])
    parser.add_option(name="-laplacian",
                      type_value="multiple_choice",
                      description="Apply Laplacian filtering. More accuracy but could mistake disc depending on anatomy.",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1'])
    parser.add_option(name="-param",
                      type_value=[[','], 'str'],
                      description="Advanced parameters. Assign value with \"=\"; Separate arguments with \",\"\n"
                                  "shift_AP_initc2 [mm]: AP shift for finding C2 disc. Default=" + str(param_default.shift_AP_initc2) + ".\n"
                                  "size_AP_initc2 [mm]: AP window size finding C2 disc. Default=" + str(param_default.size_AP_initc2) + ".\n"
                                  "shift_IS_initc2 [mm]: IS shift for finding C2 disc. Default=" + str(param_default.shift_IS_initc2) + ".\n"
                                  "size_IS_initc2 [mm]: IS window size finding C2 disc. Default=" + str(param_default.size_IS_initc2) + ".\n"
                                  "size_RL_initc2 [mm]: RL shift for size finding C2 disc. Default=" + str(param_default.size_RL_initc2) + ".\n"
                                  "shift_AP [mm]: AP shift of centerline for disc search. Default=" + str(param_default.shift_AP) + ".\n"
                                  "size_AP [mm]: AP window size for disc search. Default=" + str(param_default.size_AP) + ".\n"
                                  "size_RL [mm]: RL window size for disc search. Default=" + str(param_default.size_RL) + ".\n"
                                  "size_IS [mm]: IS window size for disc search. Default=" + str(param_default.size_IS) + ".\n"
                                  "gaussian_std [mm]: STD of the Gaussian function, centered at the most rostral point of the image, "
                                  "and used to weight C2-C3 disk location finding towards the rostral portion of the FOV. Values to set "
                                  "between 0.1 (strong weighting) and 999 (no weighting). Default=" + str(param_default.gaussian_std) + ".\n",
                      mandatory=False)
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="Remove temporary files.",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose. 0: nothing. 1: basic. 2: extended.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1', '2'])
    parser.add_option(name="-h",
                      type_value=None,
                      description="display this help",
                      mandatory=False)
    parser.add_option(name='-qc',
                      type_value='folder_creation',
                      description='The path where the quality control generated content will be saved',
                      default_value=param_default.path_qc)
    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    # initializations
    initz = ''
    initcenter = ''
    initc2 = 'auto'
    fname_initlabel = ''
    file_labelz = 'labelz.nii.gz'
    # initauto = False
    param = Param()

    # check user arguments
    if not args:
        args = sys.argv[1:]

    # Get parser info
    parser = get_parser()
    arguments = parser.parse(args)
    fname_in = os.path.abspath(arguments["-i"])
    fname_seg = os.path.abspath(arguments['-s'])
    contrast = arguments['-c']
    path_template = arguments['-t']
    if '-ofolder' in arguments:
        path_output = arguments['-ofolder']
    else:
        path_output = os.curdir
    param.path_qc = arguments.get("-qc", None)

    if '-initz' in arguments:
        initz = arguments['-initz']
    if '-initcenter' in arguments:
        initcenter = arguments['-initcenter']
    # if user provided text file, parse and overwrite arguments
    if '-initfile' in arguments:
        # open file
        file = open(arguments['-initfile'], 'r')
        initfile = ' ' + file.read().replace('\n', '')
        arg_initfile = initfile.split(' ')
        for idx_arg, arg in enumerate(arg_initfile):
            if arg == '-initz':
                initz = [int(x) for x in arg_initfile[idx_arg + 1].split(',')]
            if arg == '-initcenter':
                initcenter = int(arg_initfile[idx_arg + 1])
    if '-initlabel' in arguments:
        # get absolute path of label
        fname_initlabel = os.path.abspath(arguments['-initlabel'])
    if '-initc2' in arguments:
        initc2 = 'manual'
    if '-param' in arguments:
        param.update(arguments['-param'][0])
    verbose = int(arguments['-v'])
    remove_temp_files = int(arguments['-r'])
    denoise = int(arguments['-denoise'])
    laplacian = int(arguments['-laplacian'])

    path_tmp = sct.tmp_create(basename="label_vertebrae", verbose=verbose)

    # Copying input data to tmp folder
    sct.printv('\nCopying input data to tmp folder...', verbose)
    Image(fname_in).save(os.path.join(path_tmp, "data.nii"))
    Image(fname_seg).save(os.path.join(path_tmp, "segmentation.nii.gz"))

    # Go go temp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # create label to identify disc
    sct.printv('\nCreate label to identify disc...', verbose)
    fname_labelz = os.path.join(path_tmp, file_labelz)
    if initz:
        create_label_z('segmentation.nii.gz', initz[0], initz[1], fname_labelz=fname_labelz)  # create label located at z_center
    elif initcenter:
        # find z centered in FOV
        nii = Image('segmentation.nii.gz').change_orientation("RPI")
        nx, ny, nz, nt, px, py, pz, pt = nii.dim  # Get dimensions
        z_center = int(np.round(nz / 2))  # get z_center
        create_label_z('segmentation.nii.gz', z_center, initcenter, fname_labelz=fname_labelz)  # create label located at z_center
    elif fname_initlabel:
        import sct_label_utils
        # subtract "1" to label value because due to legacy, in this code the disc C2-C3 has value "2", whereas in the
        # recent version of SCT it is defined as "3". Therefore, when asking the user to define a label, we point to the
        # new definition of labels (i.e., C2-C3 = 3).
        sct_label_utils.main(['-i', fname_initlabel, '-add', '-1', '-o', fname_labelz])
    else:
        # automatically finds C2-C3 disc
        im_data = Image('data.nii')
        im_seg = Image('segmentation.nii.gz')
        im_label_c2c3 = detect_c2c3(im_data, im_seg, contrast)
        ind_label = np.where(im_label_c2c3.data)
        if not np.size(ind_label) == 0:
            # subtract "1" to label value because due to legacy, in this code the disc C2-C3 has value "2", whereas in the
            # recent version of SCT it is defined as "3".
            im_label_c2c3.data[ind_label] = 2
        else:
            sct.printv('Automatic C2-C3 detection failed. Please run the function with flag -initc2', 1, 'error')
        im_label_c2c3.save(fname_labelz)
        # initauto = True

    # dilate label so it is not lost when applying warping
    import sct_maths
    sct_maths.main(['-i', fname_labelz, '-dilate', '3', '-o', fname_labelz])

    # Straighten spinal cord
    sct.printv('\nStraighten spinal cord...', verbose)
    # check if warp_curve2straight and warp_straight2curve already exist (i.e. no need to do it another time)
    cache_sig = sct.cache_signature(
     input_files=[fname_in, fname_seg],
    )
    cachefile = os.path.join(curdir, "straightening.cache")
    if sct.cache_valid(cachefile, cache_sig) and os.path.isfile(os.path.join(curdir, "warp_curve2straight.nii.gz")) and os.path.isfile(os.path.join(curdir, "warp_straight2curve.nii.gz")) and os.path.isfile(os.path.join(curdir, "straight_ref.nii.gz")):
        # if they exist, copy them into current folder
        sct.printv('Reusing existing warping field which seems to be valid', verbose, 'warning')
        sct.copy(os.path.join(curdir, "warp_curve2straight.nii.gz"), 'warp_curve2straight.nii.gz')
        sct.copy(os.path.join(curdir, "warp_straight2curve.nii.gz"), 'warp_straight2curve.nii.gz')
        sct.copy(os.path.join(curdir, "straight_ref.nii.gz"), 'straight_ref.nii.gz')
        # apply straightening
        s, o = sct.run(['sct_apply_transfo', '-i', 'data.nii', '-w', 'warp_curve2straight.nii.gz', '-d', 'straight_ref.nii.gz', '-o', 'data_straight.nii'])
    else:
        cmd = ['sct_straighten_spinalcord', '-i', 'data.nii', '-s', 'segmentation.nii.gz', '-r', str(remove_temp_files)]
        if param.path_qc is not None and os.environ.get("SCT_RECURSIVE_QC", None) == "1":
            cmd += ['-qc', param.path_qc]
        s, o = sct.run(cmd)
        sct.cache_save(cachefile, cache_sig)

    # resample to 0.5mm isotropic to match template resolution
    sct.printv('\nResample to 0.5mm isotropic...', verbose)
    s, o = sct.run(['sct_resample', '-i', 'data_straight.nii', '-mm', '0.5x0.5x0.5', '-x', 'linear', '-o', 'data_straightr.nii'], verbose=verbose)

    # Apply straightening to segmentation
    # N.B. Output is RPI
    sct.printv('\nApply straightening to segmentation...', verbose)
    s, o = sct.run(['sct_apply_transfo', '-i', 'segmentation.nii.gz', '-d', 'data_straightr.nii', '-w', 'warp_curve2straight.nii.gz', '-o', 'segmentation_straight.nii.gz', '-x', 'linear'], verbose)
    # Threshold segmentation at 0.5
    sct.run(['sct_maths', '-i', 'segmentation_straight.nii.gz', '-thr', '0.5', '-o', 'segmentation_straight.nii.gz'], verbose)

    # if initauto:
    #     init_disc = []
    # else:
    # Apply straightening to z-label
    sct.printv('\nAnd apply straightening to label...', verbose)
    sct.run(['sct_apply_transfo', '-i', file_labelz, '-d', 'data_straightr.nii', '-w', 'warp_curve2straight.nii.gz', '-o', 'labelz_straight.nii.gz', '-x', 'nn'], verbose)
    # get z value and disk value to initialize labeling
    sct.printv('\nGet z and disc values from straight label...', verbose)
    init_disc = get_z_and_disc_values_from_label('labelz_straight.nii.gz')
    sct.printv('.. ' + str(init_disc), verbose)

    # denoise data
    if denoise:
        sct.printv('\nDenoise data...', verbose)
        sct.run(['sct_maths', '-i', 'data_straightr.nii', '-denoise', 'h=0.05', '-o', 'data_straightr.nii'], verbose)

    # apply laplacian filtering
    if laplacian:
        sct.printv('\nApply Laplacian filter...', verbose)
        sct.run(['sct_maths', '-i', 'data_straightr.nii', '-laplacian', '1', '-o', 'data_straightr.nii'], verbose)

    # detect vertebral levels on straight spinal cord
    vertebral_detection('data_straightr.nii', 'segmentation_straight.nii.gz', contrast, param, init_disc=init_disc, verbose=verbose, path_template=path_template, initc2=initc2, path_output=path_output)

    # un-straighten labeled spinal cord
    sct.printv('\nUn-straighten labeling...', verbose)
    sct.run(['sct_apply_transfo', '-i', 'segmentation_straight_labeled.nii.gz', '-d', 'segmentation.nii.gz', '-w', 'warp_straight2curve.nii.gz', '-o', 'segmentation_labeled.nii.gz', '-x', 'nn'], verbose)

    # Clean labeled segmentation
    sct.printv('\nClean labeled segmentation (correct interpolation errors)...', verbose)
    clean_labeled_segmentation('segmentation_labeled.nii.gz', 'segmentation.nii.gz', 'segmentation_labeled.nii.gz')

    # label discs
    sct.printv('\nLabel discs...', verbose)
    label_discs('segmentation_labeled.nii.gz', verbose=verbose)

    # come back
    os.chdir(curdir)

    # Generate output files
    path_seg, file_seg, ext_seg = sct.extract_fname(fname_seg)
    fname_seg_labeled = os.path.join(path_output, file_seg + '_labeled' + ext_seg)
    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(os.path.join(path_tmp, "segmentation_labeled.nii.gz"), fname_seg_labeled)
    sct.generate_output_file(os.path.join(path_tmp, "segmentation_labeled_disc.nii.gz"), os.path.join(path_output, file_seg + '_labeled_discs' + ext_seg))
    # copy straightening files in case subsequent SCT functions need them
    sct.generate_output_file(os.path.join(path_tmp, "warp_curve2straight.nii.gz"), os.path.join(path_output, "warp_curve2straight.nii.gz"), verbose)
    sct.generate_output_file(os.path.join(path_tmp, "warp_straight2curve.nii.gz"), os.path.join(path_output, "warp_straight2curve.nii.gz"), verbose)
    sct.generate_output_file(os.path.join(path_tmp, "straight_ref.nii.gz"), os.path.join(path_output, "straight_ref.nii.gz"), verbose)

    # Remove temporary files
    if remove_temp_files == 1:
        sct.printv('\nRemove temporary files...', verbose)
        sct.rmtree(path_tmp)

    # Generate QC report
    if param.path_qc is not None:
        path_qc = os.path.abspath(param.path_qc)
        labeled_seg_file = os.path.join(path_output, file_seg + '_labeled' + ext_seg)
        generate_qc(fname_in, labeled_seg_file, args, path_qc)

    sct.display_viewer_syntax([fname_in, fname_seg_labeled], colormaps=['', 'subcortical'], opacities=['1', '0.5'])


def generate_qc(fn_in, fn_labeled, args, path_qc):
    """
    Generate a quick visualization of vertebral labeling
    """
    import spinalcordtoolbox.reports.qc as qc
    import spinalcordtoolbox.reports.slice as qcslice

    def label_vertebrae(self, mask):
        """
        Draw vertebrae areas, then add text showing the vertebrae names.
        """

        import matplotlib.pyplot as plt
        import scipy.ndimage

        self.listed_seg(mask)

        ax = plt.gca()
        a = [0.0]
        data = mask
        for index, val in np.ndenumerate(data):
            if val not in a:
                a.append(val)
                index = int(val)
                if index in self._labels_regions.values():
                    color = self._labels_color[index]
                    y, x = scipy.ndimage.measurements.center_of_mass(np.where(data == val, data, 0))

                    # Draw text with a shadow

                    x += 10

                    label = list(self._labels_regions.keys())[list(self._labels_regions.values()).index(index)]
                    ax.text(x, y, label, color='black', clip_on=True)

                    x -= 0.5
                    y -= 0.5

                    ax.text(x, y, label, color=color, clip_on=True)

    qc.add_entry(
        src=fn_in,
        process='sct_label_vertebrae',
        args=args,
        path_qc=path_qc,
        plane='Sagittal',
        dpi=100,
        qcslice=qcslice.Sagittal([Image(fn_in), Image(fn_labeled)]),
        qcslice_operations=[label_vertebrae],
        qcslice_layout=lambda x: x.single(),
    )


# Detect vertebral levels
# ==========================================================================================
def vertebral_detection(fname, fname_seg, contrast, param, init_disc, verbose=1, path_template='', initc2='auto', path_output='../'):
    """
    Find intervertebral discs in straightened image using template matching
    :param fname:
    :param fname_seg:
    :param contrast:
    :param param:  advanced parameters
    :param init_disc:
    :param verbose:
    :param path_template:
    :param path_output: output path for verbose=2 pictures
    :return:
    """
    sct.printv('\nLook for template...', verbose)
    sct.printv('Path template: ' + path_template, verbose)

    # adjust file names if MNI-Poly-AMU template is used
    fname_level = get_file_label(os.path.join(path_template, 'template'), 'vertebral labeling', output='filewithpath')
    fname_template = get_file_label(os.path.join(path_template, 'template'), contrast.upper() + '-weighted template', output='filewithpath')

    # Open template and vertebral levels
    sct.printv('\nOpen template and vertebral levels...', verbose)
    data_template = Image(fname_template).data
    data_disc_template = Image(fname_level).data

    # open anatomical volume
    im_input = Image(fname)
    data = im_input.data

    # smooth data
    from scipy.ndimage.filters import gaussian_filter
    data = gaussian_filter(data, param.smooth_factor, output=None, mode="reflect")

    # get dimension of src
    nx, ny, nz = data.shape
    # define xc and yc (centered in the field of view)
    xc = int(np.round(nx / 2))  # direction RL
    yc = int(np.round(ny / 2))  # direction AP
    # get dimension of template
    nxt, nyt, nzt = data_template.shape
    # define xc and yc (centered in the field of view)
    xct = int(np.round(nxt / 2))  # direction RL
    yct = int(np.round(nyt / 2))  # direction AP

    # define mean distance (in voxel) between adjacent discs: [C1/C2 -> C2/C3], [C2/C3 -> C4/C5], ..., [L1/L2 -> L2/L3]
    centerline_level = data_disc_template[xct, yct, :]
    # attribute value to each disc. Starts from max level, then decrease.
    # NB: value 2 means disc C2/C3 (and so on and so forth).
    min_level = centerline_level[centerline_level.nonzero()].min()
    max_level = centerline_level[centerline_level.nonzero()].max()
    list_disc_value_template = list(range(min_level, max_level))
    # add disc above top one
    list_disc_value_template.insert(int(0), min_level - 1)
    sct.printv('\nDisc values from template: ' + str(list_disc_value_template), verbose)
    # get diff to find transitions (i.e., discs)
    diff_centerline_level = np.diff(centerline_level)
    # get disc z-values
    list_disc_z_template = diff_centerline_level.nonzero()[0].tolist()
    list_disc_z_template.reverse()
    sct.printv('Z-values for each disc: ' + str(list_disc_z_template), verbose)
    list_distance_template = (
        np.diff(list_disc_z_template) * (-1)).tolist()  # multiplies by -1 to get positive distances
    sct.printv('Distances between discs (in voxel): ' + str(list_distance_template), verbose)

    # if automatic mode, find C2/C3 disc
    if init_disc == [] and initc2 == 'auto':
        sct.printv('\nDetect C2/C3 disk...', verbose)
        zrange = list(range(0, nz))
        ind_c2 = list_disc_value_template.index(2)
        z_peak = compute_corr_3d(data, data_template, x=xc, xshift=0, xsize=param.size_RL_initc2,
                                 y=yc, yshift=param.shift_AP_initc2, ysize=param.size_AP_initc2,
                                 z=0, zshift=param.shift_IS_initc2, zsize=param.size_IS_initc2,
                                 xtarget=xct, ytarget=yct, ztarget=list_disc_z_template[ind_c2], zrange=zrange, verbose=verbose, save_suffix='_initC2', gaussian_std=param.gaussian_std, path_output=path_output)
        init_disc = [z_peak, 2]

    # if manual mode, open viewer for user to click on C2/C3 disc
    if init_disc == [] and initc2 == 'manual':
        from spinalcordtoolbox.gui.base import AnatomicalParams
        from spinalcordtoolbox.gui.sagittal import launch_sagittal_dialog

        params = AnatomicalParams()
        params.num_points = 1
        params.vertebraes = [3, ]
        params.subtitle = 'Click at the posterior tip of C2-C3 disc\n'
        input_file = Image(fname)
        output_file = msct_image.zeros_like(input_file)
        output_file.absolutepath = os.path.join(path_output, 'labels.nii.gz')
        controller = launch_sagittal_dialog(input_file, output_file, params)
        mask_points = controller.as_string()
        # assign new init_disc_z value
        # Note: there is a discrepancy between the label value (3) and the disc value (2). As of mid-2017, the SCT convention for disc C2-C3 is value=3. Before that it was value=2.
        init_disc = [int(mask_points.split(',')[2]), 2]

    # display init disc
    if verbose == 2:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # get percentile for automatic contrast adjustment
        data_display = np.mean(data[xc - param.size_RL:xc + param.size_RL, :, :], axis=0).transpose()
        percmin = np.percentile(data_display, 10)
        percmax = np.percentile(data_display, 90)
        # display image
        plt.matshow(data_display, fignum=50, cmap=plt.cm.gray, clim=[percmin, percmax], origin='lower')
        plt.title('Anatomical image')
        plt.autoscale(enable=False)  # to prevent autoscale of axis when displaying plot
        plt.figure(50), plt.scatter(yc + param.shift_AP_visu, init_disc[0], c='yellow', s=50)
        plt.text(yc + param.shift_AP_visu + 4, init_disc[0], str(init_disc[1]) + '/' + str(init_disc[1] + 1),
                 verticalalignment='center', horizontalalignment='left', color='pink', fontsize=15), plt.draw()
        # plt.ion()  # enables interactive mode

    # FIND DISCS
    # ===========================================================================
    sct.printv('\nDetect intervertebral discs...', verbose)
    # assign initial z and disc
    current_z = init_disc[0]
    current_disc = init_disc[1]
    # mean_distance = mean_distance * pz
    # mean_distance_real = np.zeros(len(mean_distance))
    # create list for z and disc
    list_disc_z = []
    list_disc_value = []
    zrange = list(range(-10, 10))
    direction = 'superior'
    search_next_disc = True
    while search_next_disc:
        sct.printv('Current disc: ' + str(current_disc) + ' (z=' + str(current_z) + '). Direction: ' + direction, verbose)
        try:
            # get z corresponding to current disc on template
            current_z_template = list_disc_z_template[current_disc]
        except:
            # in case reached the bottom (see issue #849)
            sct.printv('WARNING: Reached the bottom of the template. Stop searching.', verbose, 'warning')
            break
        # find next disc
        # N.B. Do not search for C1/C2 disc (because poorly visible), use template distance instead
        if current_disc != 1:
            current_z = compute_corr_3d(data, data_template, x=xc, xshift=0, xsize=param.size_RL,
                                        y=yc, yshift=param.shift_AP, ysize=param.size_AP,
                                        z=current_z, zshift=0, zsize=param.size_IS,
                                        xtarget=xct, ytarget=yct, ztarget=current_z_template,
                                        zrange=zrange, verbose=verbose, save_suffix='_disc' + str(current_disc), gaussian_std=999, path_output=path_output)

        # display new disc
        if verbose == 2:
            plt.figure(50), plt.scatter(yc + param.shift_AP_visu, current_z, c='yellow', s=50)
            plt.text(yc + param.shift_AP_visu + 4, current_z, str(current_disc) + '/' + str(current_disc + 1), verticalalignment='center', horizontalalignment='left', color='yellow', fontsize=15), plt.draw()

        # append to main list
        if direction == 'superior':
            # append at the beginning
            list_disc_z.insert(0, current_z)
            list_disc_value.insert(0, current_disc)
        elif direction == 'inferior':
            # append at the end
            list_disc_z.append(current_z)
            list_disc_value.append(current_disc)

        # adjust correcting factor based on already-identified discs
        if len(list_disc_z) > 1:
            # compute distance between already-identified discs
            list_distance_current = (np.diff(list_disc_z) * (-1)).tolist()
            # retrieve the template distance corresponding to the already-identified discs
            index_disc_identified = [i for i, j in enumerate(list_disc_value_template) if j in list_disc_value[:-1]]
            list_distance_template_identified = [list_distance_template[i] for i in index_disc_identified]
            # divide subject and template distances for the identified discs
            list_subject_to_template_distance = [float(list_distance_current[i]) / list_distance_template_identified[i] for i in range(len(list_distance_current))]
            # average across identified discs to obtain an average correcting factor
            correcting_factor = np.mean(list_subject_to_template_distance)
            sct.printv('.. correcting factor: ' + str(correcting_factor), verbose)
        else:
            correcting_factor = 1
        # update list_distance specific for the subject
        list_distance = [int(np.round(list_distance_template[i] * correcting_factor)) for i in range(len(list_distance_template))]
        # updated average_disc_distance (in case it is needed)
        # average_disc_distance = int(np.round(np.mean(list_distance)))

        # assign new current_z and disc value
        if direction == 'superior':
            try:
                approx_distance_to_next_disc = list_distance[list_disc_value_template.index(current_disc - 1)]
            except ValueError:
                sct.printv('WARNING: Disc value not included in template. Using previously-calculated distance: ' + str(approx_distance_to_next_disc))
            # assign new current_z and disc value
            current_z = current_z + approx_distance_to_next_disc
            current_disc = current_disc - 1
        elif direction == 'inferior':
            try:
                approx_distance_to_next_disc = list_distance[list_disc_value_template.index(current_disc)]
            except:
                sct.printv('WARNING: Disc value not included in template. Using previously-calculated distance: ' + str(approx_distance_to_next_disc))
            # assign new current_z and disc value
            current_z = current_z - approx_distance_to_next_disc
            current_disc = current_disc + 1

        # if current_z is larger than searching zone, switch direction (and start from initial z minus approximate distance from updated template distance)
        if current_z >= nz or current_disc == 0:
            sct.printv('.. Switching to inferior direction.', verbose)
            direction = 'inferior'
            current_disc = init_disc[1] + 1
            current_z = init_disc[0] - list_distance[list_disc_value_template.index(current_disc)]
        # if current_z is lower than searching zone, stop searching
        if current_z <= 0:
            search_next_disc = False

        # if verbose == 2:
        #     # close figures
        #     plt.figure(fig_corr), plt.close()
        #     plt.figure(fig_pattern), plt.close()

    # if upper disc is not 1, add disc above top disc based on mean_distance_adjusted
    upper_disc = min(list_disc_value)
    # if not upper_disc == 1:
    sct.printv('Adding top disc based on adjusted template distance: #' + str(upper_disc - 1), verbose)
    approx_distance_to_next_disc = list_distance[list_disc_value_template.index(upper_disc - 1)]
    next_z = max(list_disc_z) + approx_distance_to_next_disc
    sct.printv('.. approximate distance: ' + str(approx_distance_to_next_disc), verbose)
    # make sure next disc does not go beyond FOV in superior direction
    if next_z > nz:
        list_disc_z.insert(0, nz)
    else:
        list_disc_z.insert(0, next_z)
    # assign disc value
    list_disc_value.insert(0, upper_disc - 1)

    # Label segmentation
    label_segmentation(fname_seg, list_disc_z, list_disc_value, verbose=verbose)

    # save figure
    if verbose == 2:
        plt.figure(50), plt.savefig(os.path.join(path_output, "fig_anat_straight_with_labels.png"))
        # plt.close()


# Create label
# ==========================================================================================
def create_label_z(fname_seg, z, value, fname_labelz='labelz.nii.gz'):
    """
    Create a label at coordinates x_center, y_center, z
    :param fname_seg: segmentation
    :param z: int
    :param fname_labelz: string file name of output label
    :return: fname_labelz
    """
    nii = Image(fname_seg)
    orientation_origin = nii.orientation
    nii = nii.change_orientation("RPI")
    nx, ny, nz, nt, px, py, pz, pt = nii.dim  # Get dimensions
    # find x and y coordinates of the centerline at z using center of mass
    x, y = center_of_mass(nii.data[:, :, z])
    x, y = int(np.round(x)), int(np.round(y))
    nii.data[:, :, :] = 0
    nii.data[x, y, z] = value
    # dilate label to prevent it from disappearing due to nearestneighbor interpolation
    from sct_maths import dilate
    nii.data = dilate(nii.data, [3])
    nii.change_orientation(orientation_origin)  # put back in original orientation
    nii.save(fname_labelz)
    return fname_labelz


# Get z and label value
# ==========================================================================================
def get_z_and_disc_values_from_label(fname_label):
    """
    Find z-value and label-value based on labeled image in RPI orientation
    :param fname_label: image in RPI orientation that contains label
    :return: [z_label, value_label] int list
    """
    nii = Image(fname_label)
    # get center of mass of label
    x_label, y_label, z_label = center_of_mass(nii.data)
    x_label, y_label, z_label = int(np.round(x_label)), int(np.round(y_label)), int(np.round(z_label))
    # get label value
    value_label = int(nii.data[x_label, y_label, z_label])
    return [z_label, value_label]


# Clean labeled segmentation
# ==========================================================================================
def clean_labeled_segmentation(fname_labeled_seg, fname_seg, fname_labeled_seg_new):
    """
    Clean labeled segmentation by:
      (i)  removing voxels in segmentation_labeled that are not in segmentation and
      (ii) adding voxels in segmentation that are not in segmentation_labeled
    :param fname_labeled_seg:
    :param fname_seg:
    :param fname_labeled_seg_new: output
    :return: none
    """
    # remove voxels in segmentation_labeled that are not in segmentation
    sct.run(['sct_maths', '-i', fname_labeled_seg, '-mul', fname_seg, '-o', 'segmentation_labeled_mul.nii.gz'])
    # add voxels in segmentation that are not in segmentation_labeled
    sct.run(['sct_maths', '-i', fname_labeled_seg, '-dilate', '2', '-o', 'segmentation_labeled_dilate.nii.gz'])  # dilate labeled segmentation
    data_label_dilate = Image('segmentation_labeled_dilate.nii.gz').data
    sct.run(['sct_maths', '-i', 'segmentation_labeled_mul.nii.gz', '-bin', '0', '-o', 'segmentation_labeled_mul_bin.nii.gz'])
    data_label_bin = Image('segmentation_labeled_mul_bin.nii.gz').data
    data_seg = Image(fname_seg).data
    data_diff = data_seg - data_label_bin
    ind_nonzero = np.where(data_diff)
    im_label = Image('segmentation_labeled_mul.nii.gz')
    for i_vox in range(len(ind_nonzero[0])):
        # assign closest label value for this voxel
        ix, iy, iz = ind_nonzero[0][i_vox], ind_nonzero[1][i_vox], ind_nonzero[2][i_vox]
        im_label.data[ix, iy, iz] = data_label_dilate[ix, iy, iz]
    # save new label file (overwrite)
    im_label.absolutepath = fname_labeled_seg_new
    im_label.save()


def compute_corr_3d(src, target, x, xshift, xsize, y, yshift, ysize, z, zshift, zsize, xtarget, ytarget, ztarget, zrange, verbose, save_suffix, gaussian_std, path_output):
    """
    Find z that maximizes correlation between src and target 3d data.
    :param src: 3d source data
    :param target: 3d target data
    :param x:
    :param xshift:
    :param xsize:
    :param y:
    :param yshift:
    :param ysize:
    :param z:
    :param zshift:
    :param zsize:
    :param xtarget:
    :param ytarget:
    :param ztarget:
    :param zrange:
    :param verbose:
    :param save_suffix:
    :param gaussian_std:
    :return:
    """
    # parameters
    thr_corr = 0.2  # disc correlation threshold. Below this value, use template distance.
    # get dimensions from src
    nx, ny, nz = src.shape
    # Get pattern from template
    pattern = target[xtarget - xsize: xtarget + xsize + 1,
                     ytarget + yshift - ysize: ytarget + yshift + ysize + 1,
                     ztarget + zshift - zsize: ztarget + zshift + zsize + 1]
    pattern1d = pattern.ravel()
    # initializations
    I_corr = np.zeros(len(zrange))
    allzeros = 0
    # current_z = 0
    ind_I = 0
    # loop across range of z defined by src
    for iz in zrange:
        # if pattern extends towards the top part of the image, then crop and pad with zeros
        if z + iz + zsize + 1 > nz:
            # sct.printv('iz='+str(iz)+': padding on top')
            padding_size = z + iz + zsize + 1 - nz
            data_chunk3d = src[x - xsize: x + xsize + 1,
                               y + yshift - ysize: y + yshift + ysize + 1,
                               z + iz - zsize: z + iz + zsize + 1 - padding_size]
            data_chunk3d = np.pad(data_chunk3d, ((0, 0), (0, 0), (0, padding_size)), 'constant',
                                  constant_values=0)
        # if pattern extends towards bottom part of the image, then crop and pad with zeros
        elif z + iz - zsize < 0:
            # sct.printv('iz='+str(iz)+': padding at bottom')
            padding_size = abs(iz - zsize)
            data_chunk3d = src[x - xsize: x + xsize + 1,
                               y + yshift - ysize: y + yshift + ysize + 1,
                               z + iz - zsize + padding_size: z + iz + zsize + 1]
            data_chunk3d = np.pad(data_chunk3d, ((0, 0), (0, 0), (padding_size, 0)), 'constant',
                                  constant_values=0)
        else:
            data_chunk3d = src[x - xsize: x + xsize + 1,
                               y + yshift - ysize: y + yshift + ysize + 1,
                               z + iz - zsize: z + iz + zsize + 1]

        # convert subject pattern to 1d
        data_chunk1d = data_chunk3d.ravel()
        # check if data_chunk1d contains at least one non-zero value
        if (data_chunk1d.size == pattern1d.size) and np.any(data_chunk1d):
            I_corr[ind_I] = mutual_information(data_chunk1d, pattern1d, nbins=16, normalized=False)
        else:
            allzeros = 1
        ind_I = ind_I + 1
    # ind_y = ind_y + 1
    if allzeros:
        sct.printv('.. WARNING: Data contained zero. We probably hit the edge of the image.', verbose)

    # adjust correlation with Gaussian function centered at the right edge of the curve (most rostral point of FOV)
    from scipy.signal import gaussian
    gaussian_window = gaussian(len(I_corr) * 2, std=len(I_corr) * gaussian_std)
    I_corr_gauss = np.multiply(I_corr, gaussian_window[0:len(I_corr)])

    # Find global maximum
    if np.any(I_corr_gauss):
        # if I_corr contains at least a non-zero value
        ind_peak = [i for i in range(len(I_corr_gauss)) if I_corr_gauss[i] == max(I_corr_gauss)][0]  # index of max along z
        sct.printv('.. Peak found: z=' + str(zrange[ind_peak]) + ' (correlation = ' + str(I_corr_gauss[ind_peak]) + ')', verbose)
        # check if correlation is high enough
        if I_corr_gauss[ind_peak] < thr_corr:
            sct.printv('.. WARNING: Correlation is too low. Using adjusted template distance.', verbose)
            ind_peak = zrange.index(0)  # approx_distance_to_next_disc
    else:
        # if I_corr contains only zeros
        sct.printv('.. WARNING: Correlation vector only contains zeros. Using adjusted template distance.', verbose)
        ind_peak = zrange.index(0)  # approx_distance_to_next_disc

    # display patterns and correlation
    if verbose == 2:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # display template pattern
        plt.figure(11, figsize=(15, 7))
        plt.subplot(131)
        plt.imshow(np.flipud(np.mean(pattern[:, :, :], axis=0).transpose()), origin='upper', cmap=plt.cm.gray,
                   interpolation='none')
        plt.title('Template pattern')
        # display subject pattern at best z
        plt.subplot(132)
        iz = zrange[ind_peak]
        data_chunk3d = src[x - xsize: x + xsize + 1,
                           y + yshift - ysize: y + yshift + ysize + 1,
                           z + iz - zsize: z + iz + zsize + 1]
        plt.imshow(np.flipud(np.mean(data_chunk3d[:, :, :], axis=0).transpose()), origin='upper', cmap=plt.cm.gray,
                   clim=[0, 800], interpolation='none')
        plt.title('Subject at iz=' + str(iz))
        # display correlation curve
        plt.subplot(133)
        plt.plot(zrange, I_corr)
        plt.plot(zrange, I_corr_gauss, 'black', linestyle='dashed')
        plt.legend(['I_corr', 'I_corr_gauss'])
        plt.title('Mutual Info, gaussian_std=' + str(gaussian_std))
        plt.plot(zrange[ind_peak], I_corr_gauss[ind_peak], 'ro'), plt.draw()
        plt.axvline(x=zrange.index(0), linewidth=1, color='black', linestyle='dashed')
        plt.axhline(y=thr_corr, linewidth=1, color='r', linestyle='dashed')
        plt.grid()
        # save figure
        plt.figure(11), plt.savefig(os.path.join(path_output, "fig_pattern" + save_suffix + '.png')), plt.close()

    # return z-origin (z) + z-displacement minus zshift (to account for non-centered disc)
    return z + zrange[ind_peak] - zshift


def label_segmentation(fname_seg, list_disc_z, list_disc_value, verbose=1):
    """
    Label segmentation image
    :param fname_seg: fname of the segmentation, no orientation expected
    :param list_disc_z: list of z that correspond to a disc
    :param list_disc_value: list of associated disc values
    :param verbose:
    :return:
    """

    # open segmentation
    seg = Image(fname_seg)
    init_orientation = seg.orientation
    seg.change_orientation("RPI")

    dim = seg.dim
    ny = dim[1]
    nz = dim[2]
    # loop across z
    for iz in range(nz):
        # get index of the disc right above iz
        try:
            ind_above_iz = max([i for i in range(len(list_disc_z)) if list_disc_z[i] > iz])
        except ValueError:
            # if ind_above_iz is empty, attribute value 0
            vertebral_level = 0
        else:
            # assign vertebral level (add one because iz is BELOW the disk)
            vertebral_level = list_disc_value[ind_above_iz] + 1
            # sct.printv(vertebral_level)
        # get voxels in mask
        ind_nonzero = np.nonzero(seg.data[:, :, iz])
        seg.data[ind_nonzero[0], ind_nonzero[1], iz] = vertebral_level
        if verbose == 2:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(50)
            plt.scatter(int(np.round(ny / 2)), iz, c=vertebral_level, vmin=min(list_disc_value), vmax=max(list_disc_value), cmap='prism', marker='_', s=200)
    # write file
    seg.change_orientation(init_orientation).save(sct.add_suffix(fname_seg, '_labeled'))


def label_discs(fname_seg_labeled, verbose=1):
    """
    Label discs from labeled_segmentation. The convention is C2/C3-->3, C3/C4-->4, etc.
    :param fname_seg_labeld: fname of the labeled segmentation
    :param verbose:
    :return:
    """
    # open labeled segmentation
    im_seg_labeled = Image(fname_seg_labeled)
    orientation_native = im_seg_labeled.orientation
    im_seg_labeled.change_orientation("RPI")
    nx, ny, nz = im_seg_labeled.dim[0], im_seg_labeled.dim[1], im_seg_labeled.dim[2]
    data_disc = np.zeros([nx, ny, nz])
    vertebral_level_previous = np.max(im_seg_labeled.data)  # initialize with the max label value
    # loop across z in the superior direction (i.e. starts with the bottom slice), and each time the i/i+1 interface
    # between two levels is found, create a label at the center of the cord with the value corresponding to the
    # vertebral level below the point. E.g., at interface C3/C4, the value would be 4.
    for iz in range(nz):
        # get 2d slice
        slice = im_seg_labeled.data[:, :, iz]
        # check if at least one voxel is non-zero
        if np.any(slice):
            slice_one = np.copy(slice)
            # set all non-zero values to 1 (to compute center of mass)
            # Note: the reason we do this is because if one slice includes part of vertebral level i and i+1, the
            # center of mass will be shifted towards the i+1 level.We don't want that here (i.e. the goal is to be at
            # the center of the cord)
            slice_one[slice.nonzero()] = 1
            # compute center of mass
            cx, cy = [int(x) for x in np.round(center_of_mass(slice_one)).tolist()]
            # retrieve vertebral level
            vertebral_level = slice[cx, cy]
            # if smaller than previous level, then labeled as a disc
            if vertebral_level < vertebral_level_previous:
                # label disc
                data_disc[cx, cy, iz] = vertebral_level + 1
            # update variable
            vertebral_level_previous = vertebral_level
    # save disc labeled file
    im_seg_labeled.data = data_disc
    im_seg_labeled.change_orientation(orientation_native).save(sct.add_suffix(fname_seg_labeled, '_disc'))


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
