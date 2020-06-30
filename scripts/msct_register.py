#!/usr/bin/env python
#########################################################################################
# Various modules for registration.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Tanguy Magnan, Julien Cohen-Adad, Nicolas Pinon
#
# License: see the LICENSE.TXT
#########################################################################################

# TODO: before running the PCA, correct for the "stretch" effect caused by curvature
# TODO: columnwise: check inverse field
# TODO: columnwise: add regularization: should not binarize at 0.5, especially problematic for edge (because division by zero to compute Sx, Sy).
# TODO: clean code for generate_warping_field (unify with centermass_rot)

from __future__ import division, absolute_import

import sys, os, logging
from math import asin, cos, sin, acos
import numpy as np

from scipy import ndimage
from scipy.signal import argrelmax, medfilt
from scipy.io import loadmat
from nibabel import load, Nifti1Image, save

from spinalcordtoolbox.image import Image, find_zmin_zmax, spatial_crop
from spinalcordtoolbox.utils import sct_progress_bar

import sct_utils as sct
import sct_apply_transfo
import sct_concat_transfo
from sct_convert import convert
from sct_image import split_data, concat_warp2d
from msct_register_landmarks import register_landmarks

logger = logging.getLogger(__name__)


# Parameters for registration
class Paramreg(object):
    def __init__(self, step=None, type=None, algo='syn', metric='MeanSquares', iter='10', shrink='1', smooth='0',
                 gradStep='0.5', deformation='1x1x0', init='', filter_size=5, poly='5', slicewise='0', laplacian='0',
                 dof='Tx_Ty_Tz_Rx_Ry_Rz', smoothWarpXY='2', pca_eigenratio_th='1.6', rot_method='pca'):
        """
        Class to define registration method.

        :param step: int: Step number (starts at 1, except for type=label which corresponds to step=0).
        :param type: {im, seg, imseg, label} Type of data used for registration. Use type=label only at step=0.
        :param algo:
        :param metric:
        :param iter:
        :param shrink:
        :param smooth:
        :param gradStep:
        :param deformation:
        :param init:
        :param filter_size: int: Size of the Gaussian kernel when filtering the cord rotation estimate across z.
        :param poly:
        :param slicewise: {'0', '1'}: Slice-by-slice 2d transformation.
        :param laplacian:
        :param dof:
        :param smoothWarpXY:
        :param pca_eigenratio_th:
        :param rot_method: {'pca', 'hog', 'pcahog'}: Rotation method to be used with algo=centermassrot.
            pca: approximate cord segmentation by an ellipse and finds it orientation using PCA's
            eigenvectors; hog: finds the orientation using the symmetry of the image; pcahog: tries method pca and if it
            fails, uses method hog. If using hog or pcahog, type should be set to 'imseg'."
        """
        self.step = step
        self.type = type
        self.algo = algo
        self.metric = metric
        self.iter = iter
        self.shrink = shrink
        self.smooth = smooth
        self.laplacian = laplacian
        self.gradStep = gradStep
        self.deformation = deformation
        self.slicewise = slicewise
        self.init = init
        self.poly = poly  # only for algo=slicereg
        self.filter_size = filter_size  # only for algo=centermassrot
        self.dof = dof  # only for type=label
        self.smoothWarpXY = smoothWarpXY  # only for algo=columnwise
        self.pca_eigenratio_th = pca_eigenratio_th  # only for algo=centermassrot
        self.rot_method = rot_method  # only for algo=centermassrot
        self.rot_src = None  # this variable is used to set the angle of the cord on the src image if it is known
        self.rot_dest = None  # same as above for the destination image (e.g., if template, should be set to 0)

        # list of possible values for self.type
        self.type_list = ['im', 'seg', 'imseg', 'label']

    # update constructor with user's parameters
    def update(self, paramreg_user):
        list_objects = paramreg_user.split(',')
        for object in list_objects:
            if len(object) < 2:
                sct.printv('Please check parameter -param (usage changed from previous version)', 1, type='error')
            obj = object.split('=')
            setattr(self, obj[0], obj[1])


class ParamregMultiStep:
    """
    Class to aggregate multiple Paramreg() classes into a dictionary. The method addStep() is used to build this class.
    """
    def __init__(self, listParam=[]):
        self.steps = dict()
        for stepParam in listParam:
            if isinstance(stepParam, Paramreg):
                self.steps[stepParam.step] = stepParam
            else:
                self.addStep(stepParam)

    def addStep(self, stepParam):
        # this function checks if the step is already present. If it is present, it must update it. If it is not, it
        # must add it.
        param_reg = Paramreg()
        param_reg.update(stepParam)
        # parameters must contain 'step'
        if param_reg.step is None:
            sct.printv("ERROR: parameters must contain 'step'", 1, 'error')
        else:
            if param_reg.step in self.steps:
                self.steps[param_reg.step].update(stepParam)
            else:
                self.steps[param_reg.step] = param_reg
        # parameters must contain 'type'
        if int(param_reg.step) != 0 and param_reg.type not in param_reg.type_list:
            sct.printv("ERROR: parameters must contain a type, either 'im' or 'seg'", 1, 'error')


def register_wrapper(fname_src, fname_dest, param, paramregmulti, fname_src_seg='', fname_dest_seg='', fname_src_label='',
                     fname_dest_label='', fname_mask='', fname_initwarp='', fname_initwarpinv='', identity=False,
                     interp='linear', fname_output='', fname_output_warp='', path_out='', same_space=False):
    """
    Wrapper for image registration.

    :param fname_src:
    :param fname_dest:
    :param param: Class Param(): See definition in sct_register_multimodal
    :param paramregmulti: Class ParamregMultiStep(): See definition in this file
    :param fname_src_seg:
    :param fname_dest_seg:
    :param fname_src_label:
    :param fname_dest_label:
    :param fname_mask:
    :param fname_initwarp: str: File name of initial transformation
    :param fname_initwarpinv: str: File name of initial inverse transformation
    :param identity:
    :param interp:
    :param fname_output:
    :param fname_output_warp:
    :param path_out:
    :param same_space: Bool: Source and destination images are in the same physical space (i.e. same coordinates).
    :return: fname_src2dest, fname_dest2src, fname_output_warp, fname_output_warpinv
    """
    # TODO: move interp inside param.
    # TODO: merge param inside paramregmulti by having a "global" sets of parameters that apply to all steps

    # Extract path, file and extension
    path_src, file_src, ext_src = sct.extract_fname(fname_src)
    path_dest, file_dest, ext_dest = sct.extract_fname(fname_dest)

    # check if source and destination images have the same name (related to issue #373)
    # If so, change names to avoid conflict of result files and warns the user
    suffix_src, suffix_dest = '_reg', '_reg'
    if file_src == file_dest:
        suffix_src, suffix_dest = '_src_reg', '_dest_reg'

    # define output folder and file name
    if fname_output == '':
        path_out = '' if not path_out else path_out  # output in user's current directory
        file_out = file_src + suffix_src
        file_out_inv = file_dest + suffix_dest
        ext_out = ext_src
    else:
        path, file_out, ext_out = sct.extract_fname(fname_output)
        path_out = path if not path_out else path_out
        file_out_inv = file_out + '_inv'

    # create temporary folder
    path_tmp = sct.tmp_create(basename="register")

    sct.printv('\nCopying input data to tmp folder and convert to nii...', param.verbose)
    Image(fname_src).save(os.path.join(path_tmp, "src.nii"))
    Image(fname_dest).save(os.path.join(path_tmp, "dest.nii"))

    if fname_src_seg:
        Image(fname_src_seg).save(os.path.join(path_tmp, "src_seg.nii"))

    if fname_dest_seg:
        Image(fname_dest_seg).save(os.path.join(path_tmp, "dest_seg.nii"))

    if fname_src_label:
        Image(fname_src_label).save(os.path.join(path_tmp, "src_label.nii"))
        Image(fname_dest_label).save(os.path.join(path_tmp, "dest_label.nii"))

    if fname_mask != '':
        Image(fname_mask).save(os.path.join(path_tmp, "mask.nii.gz"))

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # reorient destination to RPI
    Image('dest.nii').change_orientation("RPI").save('dest_RPI.nii')
    if fname_dest_seg:
        Image('dest_seg.nii').change_orientation("RPI").save('dest_seg_RPI.nii')
    if fname_dest_label:
        Image('dest_label.nii').change_orientation("RPI").save('dest_label_RPI.nii')
    if fname_mask:
        # TODO: change output name
        Image('mask.nii.gz').change_orientation("RPI").save('mask.nii.gz')

    if identity:
        # overwrite paramregmulti and only do one identity transformation
        step0 = Paramreg(step='0', type='im', algo='syn', metric='MI', iter='0', shrink='1', smooth='0', gradStep='0.5')
        paramregmulti = ParamregMultiStep([step0])

    # initialize list of warping fields
    warp_forward = []
    warp_forward_winv = []
    warp_inverse = []
    warp_inverse_winv = []
    generate_warpinv = 1

    # initial warping is specified, update list of warping fields and skip step=0
    if fname_initwarp:
        sct.printv('\nSkip step=0 and replace with initial transformations: ', param.verbose)
        sct.printv('  ' + fname_initwarp, param.verbose)
        # sct.copy(fname_initwarp, 'warp_forward_0.nii.gz')
        warp_forward.append(fname_initwarp)
        start_step = 1
        if fname_initwarpinv:
            warp_inverse.append(fname_initwarpinv)
        else:
            sct.printv('\nWARNING: No initial inverse warping field was specified, therefore the inverse warping field '
                       'will NOT be generated.', param.verbose, 'warning')
            generate_warpinv = 0
    else:
        if same_space:
            start_step = 1
        else:
            start_step = 0

    # loop across registration steps
    for i_step in range(start_step, len(paramregmulti.steps)):
        sct.printv('\n--\nESTIMATE TRANSFORMATION FOR STEP #' + str(i_step), param.verbose)
        # identify which is the src and dest
        if paramregmulti.steps[str(i_step)].type == 'im':
            src = ['src.nii']
            dest = ['dest_RPI.nii']
            interp_step = ['spline']
        elif paramregmulti.steps[str(i_step)].type == 'seg':
            src = ['src_seg.nii']
            dest = ['dest_seg_RPI.nii']
            interp_step = ['nn']
        elif paramregmulti.steps[str(i_step)].type == 'imseg':
            src = ['src.nii', 'src_seg.nii']
            dest = ['dest_RPI.nii', 'dest_seg_RPI.nii']
            interp_step = ['spline', 'nn']
        elif paramregmulti.steps[str(i_step)].type == 'label':
            src = ['src_label.nii']
            dest = ['dest_label_RPI.nii']
            interp_step = ['nn']
        else:
            sct.printv('ERROR: Wrong image type: {}'.format(paramregmulti.steps[str(i_step)].type), 1, 'error')
        # if step>0, apply warp_forward_concat to the src image to be used
        if (not same_space and i_step > 0) or (same_space and i_step > 1):
            sct.printv('\nApply transformation from previous step', param.verbose)
            for ifile in range(len(src)):
                sct_apply_transfo.main(args=[
                    '-i', src[ifile],
                    '-d', dest[ifile],
                    '-w', warp_forward,
                    '-o', sct.add_suffix(src[ifile], '_reg'),
                    '-x', interp_step[ifile]])
                src[ifile] = sct.add_suffix(src[ifile], '_reg')
        # register src --> dest
        warp_forward_out, warp_inverse_out = register(src, dest, paramregmulti, param, str(i_step))
        # deal with transformations with "-" as prefix. They should be inverted with calling sct_concat_transfo.
        if warp_forward_out[0] == "-":
            warp_forward_out = warp_forward_out[1:]
            warp_forward_winv.append(warp_forward_out)
        if warp_inverse_out[0] == "-":
            warp_inverse_out = warp_inverse_out[1:]
            warp_inverse_winv.append(warp_inverse_out)
        # update list of forward/inverse transformations
        warp_forward.append(warp_forward_out)
        warp_inverse.insert(0, warp_inverse_out)

    # Concatenate transformations
    sct.printv('\nConcatenate transformations...', param.verbose)
    sct_concat_transfo.main(args=[
        '-w', warp_forward,
        '-winv', warp_forward_winv,
        '-d', 'dest.nii',
        '-o', 'warp_src2dest.nii.gz'])
    sct_concat_transfo.main(args=[
        '-w', warp_inverse,
        '-winv', warp_inverse_winv,
        '-d', 'src.nii',
        '-o', 'warp_dest2src.nii.gz'])

    # TODO: make the following code optional (or move it to sct_register_multimodal)
    # Apply warping field to src data
    sct.printv('\nApply transfo source --> dest...', param.verbose)
    sct_apply_transfo.main(args=[
        '-i', 'src.nii',
        '-d', 'dest.nii',
        '-w', 'warp_src2dest.nii.gz',
        '-o', 'src_reg.nii',
        '-x', interp])
    sct.printv('\nApply transfo dest --> source...', param.verbose)
    sct_apply_transfo.main(args=[
        '-i', 'dest.nii',
        '-d', 'src.nii',
        '-w', 'warp_dest2src.nii.gz',
        '-o', 'dest_reg.nii',
        '-x', interp])

    # come back
    os.chdir(curdir)

    # Generate output files
    # ------------------------------------------------------------------------------------------------------------------

    sct.printv('\nGenerate output files...', param.verbose)
    # generate: src_reg
    fname_src2dest = sct.generate_output_file(
        os.path.join(path_tmp, "src_reg.nii"), os.path.join(path_out, file_out + ext_out), param.verbose)

    # generate: dest_reg
    fname_dest2src = sct.generate_output_file(
        os.path.join(path_tmp, "dest_reg.nii"), os.path.join(path_out, file_out_inv + ext_dest), param.verbose)

    # generate: forward warping field
    if fname_output_warp == '':
        fname_output_warp = os.path.join(path_out, 'warp_' + file_src + '2' + file_dest + '.nii.gz')
    sct.generate_output_file(os.path.join(path_tmp, "warp_src2dest.nii.gz"), fname_output_warp, param.verbose)

    # generate: inverse warping field
    if generate_warpinv:
        fname_output_warpinv = os.path.join(path_out, 'warp_' + file_dest + '2' + file_src + '.nii.gz')
        sct.generate_output_file(os.path.join(path_tmp, "warp_dest2src.nii.gz"), fname_output_warpinv, param.verbose)
    else:
        fname_output_warpinv = None

    # Delete temporary files
    if param.remove_temp_files:
        sct.printv('\nRemove temporary files...', param.verbose)
        sct.rmtree(path_tmp, verbose=param.verbose)

    return fname_src2dest, fname_dest2src, fname_output_warp, fname_output_warpinv


# register images
# ==========================================================================================
def register(src, dest, paramregmulti, param, i_step_str):
    """
    Register src onto dest image. Output affine transformations that need to be inverted will have the prefix "-".
    :param src:
    :param dest:
    :param paramregmulti: Class ParamregMultiStep()
    :param param:
    :param i_step_str:
    :return: list: warp_forward, warp_inverse
    """
    # initiate default parameters of antsRegistration transformation
    ants_registration_params = {'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '',
                                'bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}
    output = ''  # default output if problem

    # If the input type is either im or seg, we can convert the input list into a string for improved code clarity
    if not paramregmulti.steps[i_step_str].type == 'imseg':
        src = src[0]
        dest = dest[0]

    # display arguments
    sct.printv('Registration parameters:', param.verbose)
    sct.printv('  type ........... ' + paramregmulti.steps[i_step_str].type, param.verbose)
    sct.printv('  algo ........... ' + paramregmulti.steps[i_step_str].algo, param.verbose)
    sct.printv('  slicewise ...... ' + paramregmulti.steps[i_step_str].slicewise, param.verbose)
    sct.printv('  metric ......... ' + paramregmulti.steps[i_step_str].metric, param.verbose)
    sct.printv('  iter ........... ' + paramregmulti.steps[i_step_str].iter, param.verbose)
    sct.printv('  smooth ......... ' + paramregmulti.steps[i_step_str].smooth, param.verbose)
    sct.printv('  laplacian ...... ' + paramregmulti.steps[i_step_str].laplacian, param.verbose)
    sct.printv('  shrink ......... ' + paramregmulti.steps[i_step_str].shrink, param.verbose)
    sct.printv('  gradStep ....... ' + paramregmulti.steps[i_step_str].gradStep, param.verbose)
    sct.printv('  deformation .... ' + paramregmulti.steps[i_step_str].deformation, param.verbose)
    sct.printv('  init ........... ' + paramregmulti.steps[i_step_str].init, param.verbose)
    sct.printv('  poly ........... ' + paramregmulti.steps[i_step_str].poly, param.verbose)
    sct.printv('  filter_size .... ' + str(paramregmulti.steps[i_step_str].filter_size), param.verbose)
    sct.printv('  dof ............ ' + paramregmulti.steps[i_step_str].dof, param.verbose)
    sct.printv('  smoothWarpXY ... ' + paramregmulti.steps[i_step_str].smoothWarpXY, param.verbose)
    sct.printv('  rot_method ..... ' + paramregmulti.steps[i_step_str].rot_method, param.verbose)

    # set metricSize
    if paramregmulti.steps[i_step_str].metric == 'MI':
        metricSize = '32'  # corresponds to number of bins
    else:
        metricSize = '4'  # corresponds to radius (for CC, MeanSquares...)

    # set masking
    if param.fname_mask:
        fname_mask = 'mask.nii.gz'
        masking = ['-x', 'mask.nii.gz']
    else:
        fname_mask = ''
        masking = []

    if paramregmulti.steps[i_step_str].algo == 'slicereg':
        # check if user used type=label
        if paramregmulti.steps[i_step_str].type == 'label':
            sct.printv('\nERROR: this algo is not compatible with type=label. Please use type=im or type=seg', 1,
                       'error')
        else:
            # Find the min (and max) z-slice index below which (and above which) slices only have voxels below a given
            # threshold.
            list_fname = [src, dest]
            if not masking == []:
                list_fname.append(fname_mask)
            zmin_global, zmax_global = 0, 99999  # this is assuming that typical image has less slice than 99999
            for fname in list_fname:
                im = Image(fname)
                zmin, zmax = find_zmin_zmax(im, threshold=0.1)
                if zmin > zmin_global:
                    zmin_global = zmin
                if zmax < zmax_global:
                    zmax_global = zmax
            # crop images (see issue #293)
            src_crop = sct.add_suffix(src, '_crop')
            spatial_crop(Image(src), dict(((2, (zmin_global, zmax_global)),))).save(src_crop)
            dest_crop = sct.add_suffix(dest, '_crop')
            spatial_crop(Image(dest), dict(((2, (zmin_global, zmax_global)),))).save(dest_crop)
            # update variables
            src = src_crop
            dest = dest_crop
            scr_regStep = sct.add_suffix(src, '_regStep' + i_step_str)
            # estimate transfo
            # TODO fixup isct_ants* parsers
            cmd = ['isct_antsSliceRegularizedRegistration',
                   '-t', 'Translation[' + paramregmulti.steps[i_step_str].gradStep + ']',
                   '-m',
                   paramregmulti.steps[i_step_str].metric + '[' + dest + ',' + src + ',1,' + metricSize + ',Regular,0.2]',
                   '-p', paramregmulti.steps[i_step_str].poly,
                   '-i', paramregmulti.steps[i_step_str].iter,
                   '-f', paramregmulti.steps[i_step_str].shrink,
                   '-s', paramregmulti.steps[i_step_str].smooth,
                   '-v', '1',  # verbose (verbose=2 does not exist, so we force it to 1)
                   '-o', '[step' + i_step_str + ',' + scr_regStep + ']',  # here the warp name is stage10 because
                   # antsSliceReg add "Warp"
                   ] + masking
            warp_forward_out = 'step' + i_step_str + 'Warp.nii.gz'
            warp_inverse_out = 'step' + i_step_str + 'InverseWarp.nii.gz'
            # run command
            status, output = sct.run(cmd, param.verbose, is_sct_binary=True)

    # ANTS 3d
    elif paramregmulti.steps[i_step_str].algo.lower() in ants_registration_params \
            and paramregmulti.steps[i_step_str].slicewise == '0':
        # make sure type!=label. If type==label, this will be addressed later in the code.
        if not paramregmulti.steps[i_step_str].type == 'label':
            # Pad the destination image (because ants doesn't deform the extremities)
            # N.B. no need to pad if iter = 0
            if not paramregmulti.steps[i_step_str].iter == '0':
                dest_pad = sct.add_suffix(dest, '_pad')
                sct.run(['sct_image', '-i', dest, '-o', dest_pad, '-pad', '0,0,' + str(param.padding)])
                dest = dest_pad
            # apply Laplacian filter
            if not paramregmulti.steps[i_step_str].laplacian == '0':
                sct.printv('\nApply Laplacian filter', param.verbose)
                sct.run(['sct_maths', '-i', src, '-laplacian', paramregmulti.steps[i_step_str].laplacian + ','
                         + paramregmulti.steps[i_step_str].laplacian + ',0', '-o', sct.add_suffix(src, '_laplacian')])
                sct.run(['sct_maths', '-i', dest, '-laplacian', paramregmulti.steps[i_step_str].laplacian + ','
                         + paramregmulti.steps[i_step_str].laplacian + ',0', '-o', sct.add_suffix(dest, '_laplacian')])
                src = sct.add_suffix(src, '_laplacian')
                dest = sct.add_suffix(dest, '_laplacian')
            # Estimate transformation
            sct.printv('\nEstimate transformation', param.verbose)
            scr_regStep = sct.add_suffix(src, '_regStep' + i_step_str)
            # TODO fixup isct_ants* parsers
            cmd = ['isct_antsRegistration',
                   '--dimensionality', '3',
                   '--transform', paramregmulti.steps[i_step_str].algo + '[' + paramregmulti.steps[i_step_str].gradStep
                   + ants_registration_params[paramregmulti.steps[i_step_str].algo.lower()] + ']',
                   '--metric', paramregmulti.steps[i_step_str].metric + '[' + dest + ',' + src + ',1,' + metricSize + ']',
                   '--convergence', paramregmulti.steps[i_step_str].iter,
                   '--shrink-factors', paramregmulti.steps[i_step_str].shrink,
                   '--smoothing-sigmas', paramregmulti.steps[i_step_str].smooth + 'mm',
                   '--restrict-deformation', paramregmulti.steps[i_step_str].deformation,
                   '--output', '[step' + i_step_str + ',' + scr_regStep + ']',
                   '--interpolation', 'BSpline[3]',
                   '--verbose', '1',
                   ] + masking
            # add init translation
            if not paramregmulti.steps[i_step_str].init == '':
                init_dict = {'geometric': '0', 'centermass': '1', 'origin': '2'}
                cmd += ['-r', '[' + dest + ',' + src + ',' + init_dict[paramregmulti.steps[i_step_str].init] + ']']
            # run command
            status, output = sct.run(cmd, param.verbose, is_sct_binary=True)
            # get appropriate file name for transformation
            if paramregmulti.steps[i_step_str].algo in ['rigid', 'affine', 'translation']:
                warp_forward_out = 'step' + i_step_str + '0GenericAffine.mat'
                warp_inverse_out = '-step' + i_step_str + '0GenericAffine.mat'
            else:
                warp_forward_out = 'step' + i_step_str + '0Warp.nii.gz'
                warp_inverse_out = 'step' + i_step_str + '0InverseWarp.nii.gz'

    # ANTS 2d
    elif paramregmulti.steps[i_step_str].algo.lower() in ants_registration_params \
            and paramregmulti.steps[i_step_str].slicewise == '1':
        # make sure type!=label. If type==label, this will be addressed later in the code.
        if not paramregmulti.steps[i_step_str].type == 'label':
            # if shrink!=1, force it to be 1 (otherwise, it generates a wrong 3d warping field). TODO: fix that!
            if not paramregmulti.steps[i_step_str].shrink == '1':
                sct.printv('\nWARNING: when using slicewise with SyN or BSplineSyN, shrink factor needs to be one. '
                           'Forcing shrink=1.', 1, 'warning')
                paramregmulti.steps[i_step_str].shrink = '1'
            warp_forward_out = 'step' + i_step_str + 'Warp.nii.gz'
            warp_inverse_out = 'step' + i_step_str + 'InverseWarp.nii.gz'

            register_slicewise(src,
                               dest,
                               paramreg=paramregmulti.steps[i_step_str],
                               fname_mask=fname_mask,
                               warp_forward_out=warp_forward_out,
                               warp_inverse_out=warp_inverse_out,
                               ants_registration_params=ants_registration_params,
                               remove_temp_files=param.remove_temp_files,
                               verbose=param.verbose)

    # slice-wise transfo
    elif paramregmulti.steps[i_step_str].algo in ['centermass', 'centermassrot', 'columnwise']:
        # if type=label, exit with error
        if paramregmulti.steps[i_step_str].type == 'label':
            sct.printv('\nERROR: this algo is not compatible with type=label. Please use type=im or type=seg', 1,
                       'error')
        # check if user provided a mask-- if so, inform it will be ignored
        if not fname_mask == '':
            sct.printv('\nWARNING: algo ' + paramregmulti.steps[i_step_str].algo + ' will ignore the provided mask.\n', 1,
                       'warning')
        # smooth data
        if not paramregmulti.steps[i_step_str].smooth == '0':
            sct.printv('\nWARNING: algo ' + paramregmulti.steps[i_step_str].algo + ' will ignore the parameter smoothing.\n',
                       1, 'warning')
        warp_forward_out = 'step' + i_step_str + 'Warp.nii.gz'
        warp_inverse_out = 'step' + i_step_str + 'InverseWarp.nii.gz'
        register_slicewise(
            src, dest, paramreg=paramregmulti.steps[i_step_str], fname_mask=fname_mask, warp_forward_out=warp_forward_out,
            warp_inverse_out=warp_inverse_out, ants_registration_params=ants_registration_params,
            remove_temp_files=param.remove_temp_files, verbose=param.verbose)
    else:
        sct.printv('\nERROR: algo ' + paramregmulti.steps[i_step_str].algo + ' does not exist. Exit program\n', 1, 'error')

    # landmark-based registration
    if paramregmulti.steps[i_step_str].type in ['label']:
        # check if user specified ilabel and dlabel
        # TODO
        warp_forward_out = 'step' + i_step_str + '0GenericAffine.txt'
        warp_inverse_out = '-step' + i_step_str + '0GenericAffine.txt'
        register_landmarks(src,
                           dest,
                           paramregmulti.steps[i_step_str].dof,
                           fname_affine=warp_forward_out,
                           verbose=param.verbose)

    if not os.path.isfile(warp_forward_out):
        # no forward warping field for rigid and affine
        sct.printv('\nERROR: file ' + warp_forward_out + ' doesn\'t exist (or is not a file).\n' + output +
                   '\nERROR: ANTs failed. Exit program.\n', 1, 'error')
    elif not os.path.isfile(warp_inverse_out) and \
            paramregmulti.steps[i_step_str].algo not in ['rigid', 'affine', 'translation'] and \
            paramregmulti.steps[i_step_str].type not in ['label']:
        # no inverse warping field for rigid and affine
        sct.printv('\nERROR: file ' + warp_inverse_out + ' doesn\'t exist (or is not a file).\n' + output +
                   '\nERROR: ANTs failed. Exit program.\n', 1, 'error')
    else:
        # rename warping fields
        if (paramregmulti.steps[i_step_str].algo.lower() in ['rigid', 'affine', 'translation'] and
                paramregmulti.steps[i_step_str].slicewise == '0'):
            # if ANTs is used with affine/rigid --> outputs .mat file
            warp_forward = 'warp_forward_' + i_step_str + '.mat'
            os.rename(warp_forward_out, warp_forward)
            warp_inverse = '-warp_forward_' + i_step_str + '.mat'
        elif paramregmulti.steps[i_step_str].type in ['label']:
            # if label-based registration is used --> outputs .txt file
            warp_forward = 'warp_forward_' + i_step_str + '.txt'
            os.rename(warp_forward_out, warp_forward)
            warp_inverse = '-warp_forward_' + i_step_str + '.txt'
        else:
            warp_forward = 'warp_forward_' + i_step_str + '.nii.gz'
            warp_inverse = 'warp_inverse_' + i_step_str + '.nii.gz'
            os.rename(warp_forward_out, warp_forward)
            os.rename(warp_inverse_out, warp_inverse)

    return warp_forward, warp_inverse


def register_slicewise(fname_src, fname_dest, paramreg=None, fname_mask='', warp_forward_out='step0Warp.nii.gz',
                       warp_inverse_out='step0InverseWarp.nii.gz', ants_registration_params=None,
                       path_qc='./', remove_temp_files=0, verbose=0):
    """
    Main function that calls various methods for slicewise registration.

    :param fname_src: Str or List: If List, first element is image, second element is segmentation.
    :param fname_dest: Str or List: If List, first element is image, second element is segmentation.
    :param paramreg: Class Paramreg()
    :param fname_mask:
    :param warp_forward_out:
    :param warp_inverse_out:
    :param ants_registration_params:
    :param path_qc:
    :param remove_temp_files:
    :param verbose:
    :return:
    """

    # create temporary folder
    path_tmp = sct.tmp_create(basename="register", verbose=verbose)

    # copy data to temp folder
    sct.printv('\nCopy input data to temp folder...', verbose)
    if isinstance(fname_src, list):
        # TODO: swap 0 and 1 (to be consistent with the child function below)
        convert(fname_src[0], os.path.join(path_tmp, "src.nii"))
        convert(fname_src[1], os.path.join(path_tmp, "src_seg.nii"))
        convert(fname_dest[0], os.path.join(path_tmp, "dest.nii"))
        convert(fname_dest[1], os.path.join(path_tmp, "dest_seg.nii"))
    else:
        convert(fname_src, os.path.join(path_tmp, "src.nii"))
        convert(fname_dest, os.path.join(path_tmp, "dest.nii"))

    if fname_mask != '':
        convert(fname_mask, os.path.join(path_tmp, "mask.nii.gz"))

    # go to temporary folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # Calculate displacement
    if paramreg.algo in ['centermass', 'centermassrot']:
        # translation of center of mass between source and destination in voxel space
        if paramreg.algo in 'centermass':
            rot_method = 'none'
        else:
            rot_method = paramreg.rot_method
        if rot_method in ['hog', 'pcahog']:
            src_input = ['src_seg.nii', 'src.nii']
            dest_input = ['dest_seg.nii', 'dest.nii']
        else:
            src_input = ['src.nii']
            dest_input = ['dest.nii']
        register2d_centermassrot(
            src_input, dest_input, paramreg=paramreg, fname_warp=warp_forward_out, fname_warp_inv=warp_inverse_out,
            rot_method=rot_method, filter_size=paramreg.filter_size, path_qc=path_qc, verbose=verbose,
            pca_eigenratio_th=float(paramreg.pca_eigenratio_th), )

    elif paramreg.algo == 'columnwise':
        # scaling R-L, then column-wise center of mass alignment and scaling
        register2d_columnwise('src.nii',
                              'dest.nii',
                              fname_warp=warp_forward_out,
                              fname_warp_inv=warp_inverse_out,
                              verbose=verbose,
                              path_qc=path_qc,
                              smoothWarpXY=int(paramreg.smoothWarpXY),
                              )

    # ANTs registration
    else:
        # convert SCT flags into ANTs-compatible flags
        algo_dic = {'translation': 'Translation', 'rigid': 'Rigid', 'affine': 'Affine', 'syn': 'SyN', 'bsplinesyn': 'BSplineSyN', 'centermass': 'centermass'}
        paramreg.algo = algo_dic[paramreg.algo]
        # run slicewise registration
        register2d('src.nii',
                   'dest.nii',
                   fname_mask=fname_mask,
                   fname_warp=warp_forward_out,
                   fname_warp_inv=warp_inverse_out,
                   paramreg=paramreg,
                   ants_registration_params=ants_registration_params,
                   verbose=verbose,
                   )

    sct.printv('\nMove warping fields...', verbose)
    sct.copy(warp_forward_out, curdir)
    sct.copy(warp_inverse_out, curdir)

    # go back
    os.chdir(curdir)

    if remove_temp_files:
        sct.rmtree(path_tmp, verbose=verbose)


def register2d_centermassrot(fname_src, fname_dest, paramreg=None, fname_warp='warp_forward.nii.gz',
                             fname_warp_inv='warp_inverse.nii.gz', rot_method='pca', filter_size=0, path_qc='./',
                             verbose=0, pca_eigenratio_th=1.6, th_max_angle=40):
    """
    Rotate the source image to match the orientation of the destination image, using the first and second eigenvector
    of the PCA. This function should be used on segmentations (not images).
    This works for 2D and 3D images.  If 3D, it splits the image and performs the rotation slice-by-slice.

    :param fname_src: List: Name of moving image. If rot=0 or 1, only the first element is used (should be a
        segmentation). If rot=2 or 3, the first element is a segmentation and the second is an image.
    :param fname_dest: List: Name of fixed image. If rot=0 or 1, only the first element is used (should be a
        segmentation). If rot=2 or 3, the first element is a segmentation and the second is an image.
    :param paramreg: Class Paramreg()
    :param fname_warp: name of output 3d forward warping field
    :param fname_warp_inv: name of output 3d inverse warping field
    :param rot_method: {'none', 'pca', 'hog', 'pcahog'}. Depending on the rotation method, input might be segmentation
        only or segmentation and image.
    :param filter_size: size of the gaussian filter for regularization along z for rotation angle (type: float).
        0: no regularization
    :param path_qc:
    :param verbose:
    :param pca_eigenratio_th: threshold for the ratio between the first and second eigenvector of the estimated ellipse
        for the PCA rotation detection method. If below this threshold, the estimation will be discarded (poorly robust)
    :param th_max_angle: threshold of the absolute value of the estimated rotation using the PCA method, above
        which the estimation will be discarded (unlikely to happen genuinely and hence considered outlier)
    :return:
    """
    # TODO: no need to split the src or dest if it is the template (we know its centerline and orientation already)

    if verbose == 2:
        import matplotlib
        matplotlib.use('Agg')  # prevent display figure
        import matplotlib.pyplot as plt

    # Get image dimensions and retrieve nz
    sct.printv('\nGet image dimensions of destination image...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_dest[0]).dim
    sct.printv('  matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)
    sct.printv('  voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm', verbose)

    # Split source volume along z
    sct.printv('\nSplit input segmentation...', verbose)
    im_src = Image(fname_src[0])
    split_source_list = split_data(im_src, 2)
    for im in split_source_list:
        im.save()

    # Split destination volume along z
    sct.printv('\nSplit destination segmentation...', verbose)
    im_dest = Image(fname_dest[0])
    split_dest_list = split_data(im_dest, 2)
    for im in split_dest_list:
        im.save()

    data_src = im_src.data
    data_dest = im_dest.data

    # if input data is 2D, reshape into pseudo 3D (only one slice)
    if len(data_src.shape) == 2:
        new_shape = list(data_src.shape)
        new_shape.append(1)
        new_shape = tuple(new_shape)
        data_src = data_src.reshape(new_shape)
        data_dest = data_dest.reshape(new_shape)

    # Deal with cases where both an image and segmentation are input
    if len(fname_src) > 1:
        # Split source volume along z
        sct.printv('\nSplit input image...', verbose)
        im_src_im = Image(fname_src[1])
        split_source_list = split_data(im_src_im, 2)
        for im in split_source_list:
            im.save()

        # Split destination volume along z
        sct.printv('\nSplit destination image...', verbose)
        im_dest_im = Image(fname_dest[1])
        split_dest_list = split_data(im_dest_im, 2)
        for im in split_dest_list:
            im.save()

        data_src_im = im_src_im.data
        data_dest_im = im_dest_im.data

    # initialize displacement and rotation
    coord_src = [None] * nz
    pca_src = [None] * nz
    coord_dest = [None] * nz
    pca_dest = [None] * nz
    centermass_src = np.zeros([nz, 2])
    centermass_dest = np.zeros([nz, 2])
    # displacement_forward = np.zeros([nz, 2])
    # displacement_inverse = np.zeros([nz, 2])
    angle_src_dest = np.zeros(nz)
    z_nonzero = []
    th_max_angle *= np.pi / 180

    # Loop across slices
    for iz in sct_progress_bar(range(0, nz), unit='iter', unit_scale=False, desc="Estimate cord angle for each slice",
                               ascii=False, ncols=100):
        try:
            # compute PCA and get center or mass based on segmentation
            coord_src[iz], pca_src[iz], centermass_src[iz, :] = compute_pca(data_src[:, :, iz])
            coord_dest[iz], pca_dest[iz], centermass_dest[iz, :] = compute_pca(data_dest[:, :, iz])

            # detect rotation using the HOG method
            if rot_method in ['hog', 'pcahog']:
                angle_src_hog, conf_score_src = find_angle_hog(data_src_im[:, :, iz], centermass_src[iz, :],
                                                               px, py, angle_range=th_max_angle)
                angle_dest_hog, conf_score_dest = find_angle_hog(data_dest_im[:, :, iz], centermass_dest[ iz, : ],
                                                                 px, py, angle_range=th_max_angle)
                # In case no maxima is found (it should never happen)
                if (angle_src_hog is None) or (angle_dest_hog is None):
                    sct.printv('WARNING: Slice #' + str(iz) + ' no angle found in dest or src. It will be ignored.',
                               verbose, 'warning')
                    continue
                if rot_method == 'hog':
                    angle_src = -angle_src_hog  # flip sign to be consistent with PCA output
                    angle_dest = angle_dest_hog

            # Detect rotation using the PCA or PCA-HOG method
            if rot_method in ['pca', 'pcahog']:
                eigenv_src = pca_src[iz].components_.T[0][0], pca_src[iz].components_.T[1][0]
                eigenv_dest = pca_dest[iz].components_.T[0][0], pca_dest[iz].components_.T[1][0]
                # Make sure first element is always positive (to prevent sign flipping)
                if eigenv_src[0] <= 0:
                    eigenv_src = tuple([i * (-1) for i in eigenv_src])
                if eigenv_dest[0] <= 0:
                    eigenv_dest = tuple([i * (-1) for i in eigenv_dest])
                angle_src = angle_between(eigenv_src, [1, 0])
                angle_dest = angle_between([1, 0], eigenv_dest)
                # compute ratio between axis of PCA
                pca_eigenratio_src = pca_src[iz].explained_variance_ratio_[0] / pca_src[iz].explained_variance_ratio_[1]
                pca_eigenratio_dest = pca_dest[iz].explained_variance_ratio_[0] / pca_dest[iz].explained_variance_ratio_[1]
                # angle is set to 0 if either ratio between axis is too low or outside angle range
                if pca_eigenratio_src < pca_eigenratio_th or angle_src > th_max_angle or angle_src < -th_max_angle:
                    if rot_method == 'pca':
                        angle_src = 0
                    elif rot_method == 'pcahog':
                        logger.info("Switched to method 'hog' for slice: {}".format(iz))
                        angle_src = -angle_src_hog  # flip sign to be consistent with PCA output
                if pca_eigenratio_dest < pca_eigenratio_th or angle_dest > th_max_angle or angle_dest < -th_max_angle:
                    if rot_method == 'pca':
                        angle_dest = 0
                    elif rot_method == 'pcahog':
                        logger.info("Switched to method 'hog' for slice: {}".format(iz))
                        angle_dest = angle_dest_hog

            if not rot_method == 'none':
                # bypass estimation is source or destination angle is known a priori
                if paramreg.rot_src is not None:
                    angle_src = paramreg.rot_src
                if paramreg.rot_dest is not None:
                    angle_dest = paramreg.rot_dest
                # the angle between (src, dest) is the angle between (src, origin) + angle between (origin, dest)
                angle_src_dest[iz] = angle_src + angle_dest

            # append to list of z_nonzero
            z_nonzero.append(iz)

        # if one of the slice is empty, ignore it
        except ValueError:
            sct.printv('WARNING: Slice #' + str(iz) + ' is empty. It will be ignored.', verbose, 'warning')

    # regularize rotation
    if not filter_size == 0 and (rot_method in ['pca', 'hog', 'pcahog']):
        # Filtering the angles by gaussian filter
        angle_src_dest_regularized = ndimage.filters.gaussian_filter1d(angle_src_dest[z_nonzero], filter_size)
        if verbose == 2:
            plt.plot(180 * angle_src_dest[z_nonzero] / np.pi, 'ob')
            plt.plot(180 * angle_src_dest_regularized / np.pi, 'r', linewidth=2)
            plt.grid()
            plt.xlabel('z')
            plt.ylabel('Angle (deg)')
            plt.title("Regularized cord angle estimation (filter_size: {})".format(filter_size))
            plt.savefig(os.path.join(path_qc, 'register2d_centermassrot_regularize_rotation.png'))
            plt.close()
        # update variable
        angle_src_dest[z_nonzero] = angle_src_dest_regularized

    warp_x = np.zeros(data_dest.shape)
    warp_y = np.zeros(data_dest.shape)
    warp_inv_x = np.zeros(data_src.shape)
    warp_inv_y = np.zeros(data_src.shape)

    # construct 3D warping matrix
    for iz in sct_progress_bar(z_nonzero, unit='iter', unit_scale=False, desc="Build 3D deformation field",
                               ascii=False, ncols=100):
        # get indices of x and y coordinates
        row, col = np.indices((nx, ny))
        # build 2xn array of coordinates in pixel space
        coord_init_pix = np.array([row.ravel(), col.ravel(), np.array(np.ones(len(row.ravel())) * iz)]).T
        # convert coordinates to physical space
        coord_init_phy = np.array(im_src.transfo_pix2phys(coord_init_pix))
        # get centermass coordinates in physical space
        centermass_src_phy = im_src.transfo_pix2phys([[centermass_src[iz, :].T[0], centermass_src[iz, :].T[1], iz]])[0]
        centermass_dest_phy = im_src.transfo_pix2phys([[centermass_dest[iz, :].T[0], centermass_dest[iz, :].T[1], iz]])[0]
        # build rotation matrix
        R = np.matrix(((cos(angle_src_dest[iz]), sin(angle_src_dest[iz])), (-sin(angle_src_dest[iz]), cos(angle_src_dest[iz]))))
        # build 3D rotation matrix
        R3d = np.eye(3)
        R3d[0:2, 0:2] = R
        # apply forward transformation (in physical space)
        coord_forward_phy = np.array(np.dot((coord_init_phy - np.transpose(centermass_dest_phy)), R3d) + np.transpose(centermass_src_phy))
        # apply inverse transformation (in physical space)
        coord_inverse_phy = np.array(np.dot((coord_init_phy - np.transpose(centermass_src_phy)), R3d.T) + np.transpose(centermass_dest_phy))
        # display rotations
        if verbose == 2 and not angle_src_dest[iz] == 0 and not rot_method == 'hog':
            # compute new coordinates
            coord_src_rot = coord_src[iz] * R
            coord_dest_rot = coord_dest[iz] * R.T
            # generate figure
            plt.figure(figsize=(9, 9))
            # plt.ion()  # enables interactive mode (allows keyboard interruption)
            for isub in [221, 222, 223, 224]:
                # plt.figure
                plt.subplot(isub)
                # ax = matplotlib.pyplot.axis()
                try:
                    if isub == 221:
                        plt.scatter(coord_src[iz][:, 0], coord_src[iz][:, 1], s=5, marker='o', zorder=10, color='steelblue',
                                    alpha=0.5)
                        pcaaxis = pca_src[iz].components_.T
                        pca_eigenratio = pca_src[iz].explained_variance_ratio_
                        plt.title('src')
                    elif isub == 222:
                        plt.scatter([coord_src_rot[i, 0] for i in range(len(coord_src_rot))], [coord_src_rot[i, 1] for i in range(len(coord_src_rot))], s=5, marker='o', zorder=10, color='steelblue', alpha=0.5)
                        pcaaxis = pca_dest[iz].components_.T
                        pca_eigenratio = pca_dest[iz].explained_variance_ratio_
                        plt.title('src_rot')
                    elif isub == 223:
                        plt.scatter(coord_dest[iz][:, 0], coord_dest[iz][:, 1], s=5, marker='o', zorder=10, color='red',
                                    alpha=0.5)
                        pcaaxis = pca_dest[iz].components_.T
                        pca_eigenratio = pca_dest[iz].explained_variance_ratio_
                        plt.title('dest')
                    elif isub == 224:
                        plt.scatter([coord_dest_rot[i, 0] for i in range(len(coord_dest_rot))], [coord_dest_rot[i, 1] for i in range(len(coord_dest_rot))], s=5, marker='o', zorder=10, color='red', alpha=0.5)
                        pcaaxis = pca_src[iz].components_.T
                        pca_eigenratio = pca_src[iz].explained_variance_ratio_
                        plt.title('dest_rot')
                    plt.text(-2.5, -2, 'eigenvectors:', horizontalalignment='left', verticalalignment='bottom')
                    plt.text(-2.5, -2.8, str(pcaaxis), horizontalalignment='left', verticalalignment='bottom')
                    plt.text(-2.5, 2.5, 'eigenval_ratio:', horizontalalignment='left', verticalalignment='bottom')
                    plt.text(-2.5, 2, str(pca_eigenratio), horizontalalignment='left', verticalalignment='bottom')
                    plt.plot([0, pcaaxis[0, 0]], [0, pcaaxis[1, 0]], linewidth=2, color='red')
                    plt.plot([0, pcaaxis[0, 1]], [0, pcaaxis[1, 1]], linewidth=2, color='orange')
                    plt.axis([-3, 3, -3, 3])
                    plt.gca().set_aspect('equal', adjustable='box')
                except Exception as e:
                    raise Exception

            plt.savefig(os.path.join(path_qc, 'register2d_centermassrot_pca_z' + str(iz) + '.png'))
            plt.close()

        # construct 3D warping matrix
        warp_x[:, :, iz] = np.array([coord_forward_phy[i, 0] - coord_init_phy[i, 0] for i in range(nx * ny)]).reshape((nx, ny))
        warp_y[:, :, iz] = np.array([coord_forward_phy[i, 1] - coord_init_phy[i, 1] for i in range(nx * ny)]).reshape((nx, ny))
        warp_inv_x[:, :, iz] = np.array([coord_inverse_phy[i, 0] - coord_init_phy[i, 0] for i in range(nx * ny)]).reshape((nx, ny))
        warp_inv_y[:, :, iz] = np.array([coord_inverse_phy[i, 1] - coord_init_phy[i, 1] for i in range(nx * ny)]).reshape((nx, ny))

    # Generate forward warping field (defined in destination space)
    generate_warping_field(fname_dest[0], warp_x, warp_y, fname_warp, verbose)
    generate_warping_field(fname_src[0], warp_inv_x, warp_inv_y, fname_warp_inv, verbose)


def register2d_columnwise(fname_src, fname_dest, fname_warp='warp_forward.nii.gz', fname_warp_inv='warp_inverse.nii.gz', verbose=0, path_qc='./', smoothWarpXY=1):
    """
    Column-wise non-linear registration of segmentations. Based on an idea from Allan Martin.
    - Assumes src/dest are segmentations (not necessarily binary), and already registered by center of mass
    - Assumes src/dest are in RPI orientation.
    - Split along Z, then for each slice:
    - scale in R-L direction to match src/dest
    - loop across R-L columns and register by (i) matching center of mass and (ii) scaling.
    :param fname_src:
    :param fname_dest:
    :param fname_warp:
    :param fname_warp_inv:
    :param verbose:
    :return:
    """

    # initialization
    th_nonzero = 0.5  # values below are considered zero

    # for display stuff
    if verbose == 2:
        import matplotlib
        matplotlib.use('Agg')  # prevent display figure
        import matplotlib.pyplot as plt

    # Get image dimensions and retrieve nz
    sct.printv('\nGet image dimensions of destination image...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_dest).dim
    sct.printv('  matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)
    sct.printv('  voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm', verbose)

    # Split source volume along z
    sct.printv('\nSplit input volume...', verbose)
    im_src = Image('src.nii')
    split_source_list = split_data(im_src, 2)
    for im in split_source_list:
        im.save()

    # Split destination volume along z
    sct.printv('\nSplit destination volume...', verbose)
    im_dest = Image('dest.nii')
    split_dest_list = split_data(im_dest, 2)
    for im in split_dest_list:
        im.save()

    # open image
    data_src = im_src.data
    data_dest = im_dest.data

    if len(data_src.shape) == 2:
        # reshape 2D data into pseudo 3D (only one slice)
        new_shape = list(data_src.shape)
        new_shape.append(1)
        new_shape = tuple(new_shape)
        data_src = data_src.reshape(new_shape)
        data_dest = data_dest.reshape(new_shape)

    # initialize forward warping field (defined in destination space)
    warp_x = np.zeros(data_dest.shape)
    warp_y = np.zeros(data_dest.shape)

    # initialize inverse warping field (defined in source space)
    warp_inv_x = np.zeros(data_src.shape)
    warp_inv_y = np.zeros(data_src.shape)

    # Loop across slices
    sct.printv('\nEstimate columnwise transformation...', verbose)
    for iz in range(0, nz):
        sct.printv(str(iz) + '/' + str(nz) + '..',)

        # PREPARE COORDINATES
        # ============================================================
        # get indices of x and y coordinates
        row, col = np.indices((nx, ny))
        # build 2xn array of coordinates in pixel space
        # ordering of indices is as follows:
        # coord_init_pix[:, 0] = 0, 0, 0, ..., 1, 1, 1..., nx, nx, nx
        # coord_init_pix[:, 1] = 0, 1, 2, ..., 0, 1, 2..., 0, 1, 2
        coord_init_pix = np.array([row.ravel(), col.ravel(), np.array(np.ones(len(row.ravel())) * iz)]).T
        # convert coordinates to physical space
        coord_init_phy = np.array(im_src.transfo_pix2phys(coord_init_pix))
        # get 2d data from the selected slice
        src2d = data_src[:, :, iz]
        dest2d = data_dest[:, :, iz]
        # julien 20161105
        #<<<
        # threshold at 0.5
        src2d[src2d < th_nonzero] = 0
        dest2d[dest2d < th_nonzero] = 0
        # get non-zero coordinates, and transpose to obtain nx2 dimensions
        coord_src2d = np.array(np.where(src2d > 0)).T
        coord_dest2d = np.array(np.where(dest2d > 0)).T
        # here we use 0.5 as threshold for non-zero value
        # coord_src2d = np.array(np.where(src2d > th_nonzero)).T
        # coord_dest2d = np.array(np.where(dest2d > th_nonzero)).T
        #>>>

        # SCALING R-L (X dimension)
        # ============================================================
        # sum data across Y to obtain 1D signal: src_y and dest_y
        src1d = np.sum(src2d, 1)
        dest1d = np.sum(dest2d, 1)
        # make sure there are non-zero data in src or dest
        if np.any(src1d > th_nonzero) and np.any(dest1d > th_nonzero):
            # retrieve min/max of non-zeros elements (edge of the segmentation)
            # julien 20161105
            # <<<
            src1d_min, src1d_max = min(np.where(src1d != 0)[0]), max(np.where(src1d != 0)[0])
            dest1d_min, dest1d_max = min(np.where(dest1d != 0)[0]), max(np.where(dest1d != 0)[0])
            # for i in range(len(src1d)):
            #     if src1d[i] > 0.5:
            #         found index above 0.5, exit loop
                    # break
            # get indices (in continuous space) at half-maximum of upward and downward slope
            # src1d_min, src1d_max = find_index_halfmax(src1d)
            # dest1d_min, dest1d_max = find_index_halfmax(dest1d)
            # >>>
            # 1D matching between src_y and dest_y
            mean_dest_x = (dest1d_max + dest1d_min) / 2
            mean_src_x = (src1d_max + src1d_min) / 2
            # compute x-scaling factor
            Sx = (dest1d_max - dest1d_min + 1) / float(src1d_max - src1d_min + 1)
            # apply transformation to coordinates
            coord_src2d_scaleX = np.copy(coord_src2d)  # need to use np.copy to avoid copying pointer
            coord_src2d_scaleX[:, 0] = (coord_src2d[:, 0] - mean_src_x) * Sx + mean_dest_x
            coord_init_pix_scaleX = np.copy(coord_init_pix)
            coord_init_pix_scaleX[:, 0] = (coord_init_pix[:, 0] - mean_src_x) * Sx + mean_dest_x
            coord_init_pix_scaleXinv = np.copy(coord_init_pix)
            coord_init_pix_scaleXinv[:, 0] = (coord_init_pix[:, 0] - mean_dest_x) / float(Sx) + mean_src_x
            # apply transformation to image
            from skimage.transform import warp
            row_scaleXinv = np.reshape(coord_init_pix_scaleXinv[:, 0], [nx, ny])
            src2d_scaleX = warp(src2d, np.array([row_scaleXinv, col]), order=1)

            # ============================================================
            # COLUMN-WISE REGISTRATION (Y dimension for each Xi)
            # ============================================================
            coord_init_pix_scaleY = np.copy(coord_init_pix)  # need to use np.copy to avoid copying pointer
            coord_init_pix_scaleYinv = np.copy(coord_init_pix)  # need to use np.copy to avoid copying pointer
            # coord_src2d_scaleXY = np.copy(coord_src2d_scaleX)  # need to use np.copy to avoid copying pointer
            # loop across columns (X dimension)
            for ix in range(nx):
                # retrieve 1D signal along Y
                src1d = src2d_scaleX[ix, :]
                dest1d = dest2d[ix, :]
                # make sure there are non-zero data in src or dest
                if np.any(src1d > th_nonzero) and np.any(dest1d > th_nonzero):
                    # retrieve min/max of non-zeros elements (edge of the segmentation)
                    # src1d_min, src1d_max = min(np.nonzero(src1d)[0]), max(np.nonzero(src1d)[0])
                    # dest1d_min, dest1d_max = min(np.nonzero(dest1d)[0]), max(np.nonzero(dest1d)[0])
                    # 1D matching between src_y and dest_y
                    # Ty = (dest1d_max + dest1d_min)/2 - (src1d_max + src1d_min)/2
                    # Sy = (dest1d_max - dest1d_min) / float(src1d_max - src1d_min)
                    # apply translation and scaling to coordinates in column
                    # get indices (in continuous space) at half-maximum of upward and downward slope
                    # src1d_min, src1d_max = find_index_halfmax(src1d)
                    # dest1d_min, dest1d_max = find_index_halfmax(dest1d)
                    src1d_min, src1d_max = np.min(np.where(src1d > th_nonzero)), np.max(np.where(src1d > th_nonzero))
                    dest1d_min, dest1d_max = np.min(np.where(dest1d > th_nonzero)), np.max(np.where(dest1d > th_nonzero))
                    # 1D matching between src_y and dest_y
                    mean_dest_y = (dest1d_max + dest1d_min) / 2
                    mean_src_y = (src1d_max + src1d_min) / 2
                    # Tx = (dest1d_max + dest1d_min)/2 - (src1d_max + src1d_min)/2
                    Sy = (dest1d_max - dest1d_min + 1) / float(src1d_max - src1d_min + 1)
                    # apply forward transformation (in pixel space)
                    # below: only for debugging purpose
                    # coord_src2d_scaleX = np.copy(coord_src2d)  # need to use np.copy to avoid copying pointer
                    # coord_src2d_scaleX[:, 0] = (coord_src2d[:, 0] - mean_src) * Sx + mean_dest
                    # coord_init_pix_scaleY = np.copy(coord_init_pix)  # need to use np.copy to avoid copying pointer
                    # coord_init_pix_scaleY[:, 0] = (coord_init_pix[:, 0] - mean_src ) * Sx + mean_dest
                    range_x = list(range(ix * ny, ix * ny + nx))
                    coord_init_pix_scaleY[range_x, 1] = (coord_init_pix[range_x, 1] - mean_src_y) * Sy + mean_dest_y
                    coord_init_pix_scaleYinv[range_x, 1] = (coord_init_pix[range_x, 1] - mean_dest_y) / float(Sy) + mean_src_y
            # apply transformation to image
            col_scaleYinv = np.reshape(coord_init_pix_scaleYinv[:, 1], [nx, ny])
            src2d_scaleXY = warp(src2d, np.array([row_scaleXinv, col_scaleYinv]), order=1)
            # regularize Y warping fields
            from skimage.filters import gaussian
            col_scaleY = np.reshape(coord_init_pix_scaleY[:, 1], [nx, ny])
            col_scaleYsmooth = gaussian(col_scaleY, smoothWarpXY)
            col_scaleYinvsmooth = gaussian(col_scaleYinv, smoothWarpXY)
            # apply smoothed transformation to image
            src2d_scaleXYsmooth = warp(src2d, np.array([row_scaleXinv, col_scaleYinvsmooth]), order=1)
            # reshape warping field as 1d
            coord_init_pix_scaleY[:, 1] = col_scaleYsmooth.ravel()
            coord_init_pix_scaleYinv[:, 1] = col_scaleYinvsmooth.ravel()
            # display
            if verbose == 2:
                # FIG 1
                plt.figure(figsize=(15, 3))
                # plot #1
                ax = plt.subplot(141)
                plt.imshow(np.swapaxes(src2d, 1, 0), cmap=plt.cm.gray, interpolation='none')
                plt.hold(True)  # add other layer
                plt.imshow(np.swapaxes(dest2d, 1, 0), cmap=plt.cm.copper, interpolation='none', alpha=0.5)
                plt.title('src')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(mean_dest_x - 15, mean_dest_x + 15)
                plt.ylim(mean_dest_y - 15, mean_dest_y + 15)
                ax.grid(True, color='w')
                # plot #2
                ax = plt.subplot(142)
                plt.imshow(np.swapaxes(src2d_scaleX, 1, 0), cmap=plt.cm.gray, interpolation='none')
                plt.hold(True)  # add other layer
                plt.imshow(np.swapaxes(dest2d, 1, 0), cmap=plt.cm.copper, interpolation='none', alpha=0.5)
                plt.title('src_scaleX')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(mean_dest_x - 15, mean_dest_x + 15)
                plt.ylim(mean_dest_y - 15, mean_dest_y + 15)
                ax.grid(True, color='w')
                # plot #3
                ax = plt.subplot(143)
                plt.imshow(np.swapaxes(src2d_scaleXY, 1, 0), cmap=plt.cm.gray, interpolation='none')
                plt.hold(True)  # add other layer
                plt.imshow(np.swapaxes(dest2d, 1, 0), cmap=plt.cm.copper, interpolation='none', alpha=0.5)
                plt.title('src_scaleXY')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(mean_dest_x - 15, mean_dest_x + 15)
                plt.ylim(mean_dest_y - 15, mean_dest_y + 15)
                ax.grid(True, color='w')
                # plot #4
                ax = plt.subplot(144)
                plt.imshow(np.swapaxes(src2d_scaleXYsmooth, 1, 0), cmap=plt.cm.gray, interpolation='none')
                plt.hold(True)  # add other layer
                plt.imshow(np.swapaxes(dest2d, 1, 0), cmap=plt.cm.copper, interpolation='none', alpha=0.5)
                plt.title('src_scaleXYsmooth (s=' + str(smoothWarpXY) + ')')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(mean_dest_x - 15, mean_dest_x + 15)
                plt.ylim(mean_dest_y - 15, mean_dest_y + 15)
                ax.grid(True, color='w')
                # save figure
                plt.savefig(os.path.join(path_qc, 'register2d_columnwise_image_z' + str(iz) + '.png'))
                plt.close()

            # ============================================================
            # CALCULATE TRANSFORMATIONS
            # ============================================================
            # calculate forward transformation (in physical space)
            coord_init_phy_scaleX = np.array(im_dest.transfo_pix2phys(coord_init_pix_scaleX))
            coord_init_phy_scaleY = np.array(im_dest.transfo_pix2phys(coord_init_pix_scaleY))
            # calculate inverse transformation (in physical space)
            coord_init_phy_scaleXinv = np.array(im_src.transfo_pix2phys(coord_init_pix_scaleXinv))
            coord_init_phy_scaleYinv = np.array(im_src.transfo_pix2phys(coord_init_pix_scaleYinv))
            # compute displacement per pixel in destination space (for forward warping field)
            warp_x[:, :, iz] = np.array([coord_init_phy_scaleXinv[i, 0] - coord_init_phy[i, 0] for i in range(nx * ny)]).reshape((nx, ny))
            warp_y[:, :, iz] = np.array([coord_init_phy_scaleYinv[i, 1] - coord_init_phy[i, 1] for i in range(nx * ny)]).reshape((nx, ny))
            # compute displacement per pixel in source space (for inverse warping field)
            warp_inv_x[:, :, iz] = np.array([coord_init_phy_scaleX[i, 0] - coord_init_phy[i, 0] for i in range(nx * ny)]).reshape((nx, ny))
            warp_inv_y[:, :, iz] = np.array([coord_init_phy_scaleY[i, 1] - coord_init_phy[i, 1] for i in range(nx * ny)]).reshape((nx, ny))

    # Generate forward warping field (defined in destination space)
    generate_warping_field(fname_dest, warp_x, warp_y, fname_warp, verbose)
    # Generate inverse warping field (defined in source space)
    generate_warping_field(fname_src, warp_inv_x, warp_inv_y, fname_warp_inv, verbose)


def register2d(fname_src, fname_dest, fname_mask='', fname_warp='warp_forward.nii.gz',
               fname_warp_inv='warp_inverse.nii.gz',
               paramreg=Paramreg(step='0', type='im', algo='Translation', metric='MI', iter='5', shrink='1', smooth='0',
                   gradStep='0.5'),
               ants_registration_params={'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '',
                                         'translation': '', 'bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                         'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'},
               verbose=0):
    """
    Slice-by-slice registration of two images.

    :param fname_src: name of moving image (type: string)
    :param fname_dest: name of fixed image (type: string)
    :param fname_mask: name of mask file (type: string) (parameter -x of antsRegistration)
    :param fname_warp: name of output 3d forward warping field
    :param fname_warp_inv: name of output 3d inverse warping field
    :param paramreg: Class Paramreg()
    :param ants_registration_params: dict: specific algorithm's parameters for antsRegistration
    :param verbose:
    :return:
        if algo==translation:
            x_displacement: list of translation along x axis for each slice (type: list)
            y_displacement: list of translation along y axis for each slice (type: list)
        if algo==rigid:
            x_displacement: list of translation along x axis for each slice (type: list)
            y_displacement: list of translation along y axis for each slice (type: list)
            theta_rotation: list of rotation angle in radian (and in ITK's coordinate system) for each slice (type: list)
        if algo==affine or algo==syn or algo==bsplinesyn:
            creation of two 3D warping fields (forward and inverse) that are the concatenations of the slice-by-slice
            warps.
    """
    # set metricSize
    if paramreg.metric == 'MI':
        metricSize = '32'  # corresponds to number of bins
    else:
        metricSize = '4'  # corresponds to radius (for CC, MeanSquares...)

    # Get image dimensions and retrieve nz
    sct.printv('\nGet image dimensions of destination image...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_dest).dim
    sct.printv('.. matrix size: ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)
    sct.printv('.. voxel size:  ' + str(px) + 'mm x ' + str(py) + 'mm x ' + str(pz) + 'mm', verbose)

    # Split input volume along z
    sct.printv('\nSplit input volume...', verbose)
    im_src = Image(fname_src)
    split_source_list = split_data(im_src, 2)
    for im in split_source_list:
        im.save()

    # Split destination volume along z
    sct.printv('\nSplit destination volume...', verbose)
    im_dest = Image(fname_dest)
    split_dest_list = split_data(im_dest, 2)
    for im in split_dest_list:
        im.save()

    # Split mask volume along z
    if fname_mask != '':
        sct.printv('\nSplit mask volume...', verbose)
        im_mask = Image('mask.nii.gz')
        split_mask_list = split_data(im_mask, 2)
        for im in split_mask_list:
            im.save()

    # initialization
    if paramreg.algo in ['Translation']:
        x_displacement = [0 for i in range(nz)]
        y_displacement = [0 for i in range(nz)]
        theta_rotation = [0 for i in range(nz)]
    if paramreg.algo in ['Rigid', 'Affine', 'BSplineSyN', 'SyN']:
        list_warp = []
        list_warp_inv = []

    # loop across slices
    for i in range(nz):
        # set masking
        sct.printv('Registering slice ' + str(i) + '/' + str(nz - 1) + '...', verbose)
        num = numerotation(i)
        prefix_warp2d = 'warp2d_' + num
        # if mask is used, prepare command for ANTs
        if fname_mask != '':
            masking = ['-x', 'mask_Z' + num + '.nii.gz']
        else:
            masking = []
        # main command for registration
        # TODO fixup isct_ants* parsers
        cmd = ['isct_antsRegistration',
         '--dimensionality', '2',
         '--transform', paramreg.algo + '[' + str(paramreg.gradStep) + ants_registration_params[paramreg.algo.lower()] + ']',
         '--metric', paramreg.metric + '[dest_Z' + num + '.nii' + ',src_Z' + num + '.nii' + ',1,' + metricSize + ']',  #[fixedImage,movingImage,metricWeight +nb_of_bins (MI) or radius (other)
         '--convergence', str(paramreg.iter),
         '--shrink-factors', str(paramreg.shrink),
         '--smoothing-sigmas', str(paramreg.smooth) + 'mm',
         '--output', '[' + prefix_warp2d + ',src_Z' + num + '_reg.nii]',    #--> file.mat (contains Tx,Ty, theta)
         '--interpolation', 'BSpline[3]',
         '--verbose', '1',
        ] + masking
        # add init translation
        if not paramreg.init == '':
            init_dict = {'geometric': '0', 'centermass': '1', 'origin': '2'}
            cmd += ['-r', '[dest_Z' + num + '.nii' + ',src_Z' + num + '.nii,' + init_dict[paramreg.init] + ']']

        try:
            # run registration
            sct.run(cmd, is_sct_binary=True)

            if paramreg.algo in ['Translation']:
                file_mat = prefix_warp2d + '0GenericAffine.mat'
                matfile = loadmat(file_mat, struct_as_record=True)
                array_transfo = matfile['AffineTransform_double_2_2']
                x_displacement[i] = array_transfo[4][0]  # Tx in ITK'S coordinate system
                y_displacement[i] = array_transfo[5][0]  # Ty  in ITK'S and fslview's coordinate systems
                theta_rotation[i] = asin(array_transfo[2])  # angle of rotation theta in ITK'S coordinate system (minus theta for fslview)

            if paramreg.algo in ['Rigid', 'Affine', 'BSplineSyN', 'SyN']:
                # List names of 2d warping fields for subsequent merge along Z
                file_warp2d = prefix_warp2d + '0Warp.nii.gz'
                file_warp2d_inv = prefix_warp2d + '0InverseWarp.nii.gz'
                list_warp.append(file_warp2d)
                list_warp_inv.append(file_warp2d_inv)

            if paramreg.algo in ['Rigid', 'Affine']:
                # Generating null 2d warping field (for subsequent concatenation with affine transformation)
                # TODO fixup isct_ants* parsers
                sct.run(['isct_antsRegistration',
                 '-d', '2',
                 '-t', 'SyN[1,1,1]',
                 '-c', '0',
                 '-m', 'MI[dest_Z' + num + '.nii,src_Z' + num + '.nii,1,32]',
                 '-o', 'warp2d_null',
                 '-f', '1',
                 '-s', '0',
                ], is_sct_binary=True)
                # --> outputs: warp2d_null0Warp.nii.gz, warp2d_null0InverseWarp.nii.gz
                file_mat = prefix_warp2d + '0GenericAffine.mat'
                # Concatenating mat transfo and null 2d warping field to obtain 2d warping field of affine transformation
                sct.run(['isct_ComposeMultiTransform', '2', file_warp2d, '-R', 'dest_Z' + num + '.nii', 'warp2d_null0Warp.nii.gz', file_mat], is_sct_binary=True)
                sct.run(['isct_ComposeMultiTransform', '2', file_warp2d_inv, '-R', 'src_Z' + num + '.nii', 'warp2d_null0InverseWarp.nii.gz', '-i', file_mat], is_sct_binary=True)

        # if an exception occurs with ants, take the last value for the transformation
        # TODO: DO WE NEED TO DO THAT??? (julien 2016-03-01)
        except Exception as e:
            sct.printv('ERROR: Exception occurred.\n' + str(e), 1, 'error')

    # Merge warping field along z
    sct.printv('\nMerge warping fields along z...', verbose)

    if paramreg.algo in ['Translation']:
        # convert to array
        x_disp_a = np.asarray(x_displacement)
        y_disp_a = np.asarray(y_displacement)
        theta_rot_a = np.asarray(theta_rotation)
        # Generate warping field
        generate_warping_field(fname_dest, x_disp_a, y_disp_a, fname_warp=fname_warp)  #name_warp= 'step'+str(paramreg.step)
        # Inverse warping field
        generate_warping_field(fname_src, -x_disp_a, -y_disp_a, fname_warp=fname_warp_inv)

    if paramreg.algo in ['Rigid', 'Affine', 'BSplineSyN', 'SyN']:
        # concatenate 2d warping fields along z
        concat_warp2d(list_warp, fname_warp, fname_dest)
        concat_warp2d(list_warp_inv, fname_warp_inv, fname_src)


def numerotation(nb):
    """Indexation of number for matching fslsplit's index.

    Given a slice number, this function returns the corresponding number in fslsplit indexation system.

    input:
        nb: the number of the slice (type: int)

    output:
        nb_output: the number of the slice for fslsplit (type: string)
    """
    if nb < 0:
        logger.error('ERROR: the number is negative.')
        sys.exit(status=2)
    elif -1 < nb < 10:
        nb_output = '000' + str(nb)
    elif 9 < nb < 100:
        nb_output = '00' + str(nb)
    elif 99 < nb < 1000:
        nb_output = '0' + str(nb)
    elif 999 < nb < 10000:
        nb_output = str(nb)
    elif nb > 9999:
        logger.error('ERROR: the number is superior to 9999.')
        sys.exit(status = 2)
    return nb_output


def generate_warping_field(fname_dest, warp_x, warp_y, fname_warp='warping_field.nii.gz', verbose=1):
    """
    Generate an ITK warping field
    :param fname_dest:
    :param warp_x:
    :param warp_y:
    :param fname_warp:
    :param verbose:
    :return:
    """
    sct.printv('\nGenerate warping field...', verbose)

    # Get image dimensions
    # sct.printv('Get destination dimension', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_dest).dim
    # sct.printv('  matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)
    # sct.printv('  voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm', verbose)

    # initialize
    data_warp = np.zeros((nx, ny, nz, 1, 3))

    # fill matrix
    data_warp[:, :, :, 0, 0] = -warp_x  # need to invert due to ITK conventions
    data_warp[:, :, :, 0, 1] = -warp_y  # need to invert due to ITK conventions

    # save warping field
    im_dest = load(fname_dest)
    hdr_dest = im_dest.get_header()
    hdr_warp = hdr_dest.copy()
    hdr_warp.set_intent('vector', (), '')
    hdr_warp.set_data_dtype('float32')
    img = Nifti1Image(data_warp, None, hdr_warp)
    save(img, fname_warp)
    sct.printv(' --> ' + fname_warp, verbose)

    #
    # file_dest = load(fname_dest)
    # hdr_file_dest = file_dest.get_header()
    # hdr_warp = hdr_file_dest.copy()
    #
    #
    # # Center of rotation
    # if center_rotation == None:
    #     x_a = int(round(nx/2))
    #     y_a = int(round(ny/2))
    # else:
    #     x_a = center_rotation[0]
    #     y_a = center_rotation[1]
    #
    # # Calculate displacement for each voxel
    # data_warp = np.zeros(((((nx, ny, nz, 1, 3)))))
    # vector_i = [[[i-x_a], [j-y_a]] for i in range(nx) for j in range(ny)]
    #
    # # if theta_rot == None:
    # #     # for translations
    # #     for k in range(nz):
    # #         matrix_rot_a = np.asarray([[cos(theta_rot[k]), - sin(theta_rot[k])], [-sin(theta_rot[k]), -cos(theta_rot[k])]])
    # #         tmp = matrix_rot_a + array(((-1, 0), (0, 1)))
    # #         result = dot(tmp, array(vector_i).T[0]) + array([[x_trans[k]], [y_trans[k]]])
    # #         for i in range(ny):
    # #             data_warp[i, :, k, 0, 0] = result[0][i*nx:i*nx+ny]
    # #             data_warp[i, :, k, 0, 1] = result[1][i*nx:i*nx+ny]
    #
    # # else:
    #     # For rigid transforms (not optimized)
    #     # if theta_rot != None:
    # # TODO: this is not optimized! can do better!
    # for k in range(nz):
    #     for i in range(nx):
    #         for j in range(ny):
    #             data_warp[i, j, k, 0, 0] = (cos(theta_rot[k]) - 1) * (i - x_a) - sin(theta_rot[k]) * (j - y_a) + x_trans[k]
    #             data_warp[i, j, k, 0, 1] = - sin(theta_rot[k]) * (i - x_a) - (cos(theta_rot[k]) - 1) * (j - y_a) + y_trans[k]
    #             data_warp[i, j, k, 0, 2] = 0
    #
    # # Generate warp file as a warping field
    # hdr_warp.set_intent('vector', (), '')
    # hdr_warp.set_data_dtype('float32')
    # img = Nifti1Image(data_warp, None, hdr_warp)
    # save(img, fname)
    # sct.printv('\nDone! Warping field generated: '+fname, verbose)


def angle_between(a, b):
    """
    compute angle in radian between a and b. Throws an exception if a or b has zero magnitude.
    :param a:
    :param b:
    :return:
    """
    # TODO: check if extreme value that can make the function crash-- use "try"
    # from numpy.linalg import norm
    # from numpy import dot
    # import math
    arccosInput = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    # sct.printv(arccosInput)
    arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
    arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
    sign_angle = np.sign(np.cross(a, b))
    # sct.printv(sign_angle)
    return sign_angle * acos(arccosInput)

    # @xl_func("numpy_row v1, numpy_row v2: float")
    # def py_ang(v1, v2):
    # """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    # cosang = np.dot(a, b)
    # sinang = la.norm(np.cross(a, b))
    # return np.arctan2(sinang, cosang)


def compute_pca(data2d):
    """
    Compute PCA using sklearn
    :param data2d: 2d array. PCA will be computed on non-zeros values.
    :return:
        coordsrc: 2d array: centered non-zero coordinates
        pca: object: PCA result.
        centermass: 2x1 array: 2d coordinates of the center of mass
    """
    # round it and make it int (otherwise end up with values like 10-7)
    data2d = data2d.round().astype(int)
    # get non-zero coordinates, and transpose to obtain nx2 dimensions
    coordsrc = np.array(data2d.nonzero()).T
    # get center of mass
    centermass = coordsrc.mean(0)
    # center data
    coordsrc = coordsrc - centermass
    # normalize data
    coordsrc /= coordsrc.std()
    # Performs PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, copy=False, whiten=False)
    pca.fit(coordsrc)
    # pca_score = pca.explained_variance_ratio_
    # V = pca.components_
    return coordsrc, pca, centermass


def find_index_halfmax(data1d):
    """
    Find the two indices at half maximum for a bell-type curve (non-parametric). Uses center of mass calculation.
    :param data1d:
    :return: xmin, xmax
    """
    # normalize data between 0 and 1
    data1d = data1d / float(np.max(data1d))
    # loop across elements and stops when found 0.5
    for i in range(len(data1d)):
        if data1d[i] > 0.5:
            break
    # compute center of mass to get coordinate at 0.5
    xmin = i - 1 + (0.5 - data1d[i - 1]) / float(data1d[i] - data1d[i - 1])
    # continue for the descending slope
    for i in range(i, len(data1d)):
        if data1d[i] < 0.5:
            break
    # compute center of mass to get coordinate at 0.5
    xmax = i - 1 + (0.5 - data1d[i - 1]) / float(data1d[i] - data1d[i - 1])
    # display
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(src1d)
    # plt.plot(xmin, 0.5, 'o')
    # plt.plot(xmax, 0.5, 'o')
    # plt.savefig('./normalize1d.png')
    return xmin, xmax


def find_angle_hog(image, centermass, px, py, angle_range=10):
    """Finds the angle of an image based on the method described by Sun, “Symmetry Detection Using Gradient Information.”
     Pattern Recognition Letters 16, no. 9 (September 1, 1995): 987–96, and improved by N. Pinon
     inputs :
        - image : 2D numpy array to find symmetry axis on
        - centermass: tuple of floats indicating the center of mass of the image
        - px, py, dimensions of the pixels in the x and y direction
        - angle_range : float or None, in deg, the angle will be search in the range [-angle_range, angle_range], if None angle angle might be returned
     outputs :
        - angle_found : float, angle found by the method
        - conf_score : confidence score of the method (Actually a WIP, did not provide sufficient results to be used)
    """

    # param that can actually be tweeked to influence method performance :
    sigma = 10  # influence how far away pixels will vote for the orientation, if high far away pixels vote will count more, if low only closest pixels will participate
    nb_bin = 360  # number of angle bins for the histogram, can be more or less than 360, if high, a higher precision might be achieved but there is the risk of
    kmedian_size = 5

    # Normalization of sigma relative to pixdim :
    sigmax = sigma / px
    sigmay = sigma / py
    if nb_bin % 2 != 0:  # necessary to have even number of bins
        nb_bin = nb_bin - 1
    if angle_range is None:
        angle_range = 90

    # Constructing mask based on center of mass that will influence the weighting of the orientation histogram
    nx, ny = image.shape
    xx, yy = np.mgrid[:nx, :ny]
    seg_weighted_mask = np.exp(
        -(((xx - centermass[0]) ** 2) / (2 * (sigmax ** 2)) + ((yy - centermass[1]) ** 2) / (2 * (sigmay ** 2))))

    # Acquiring the orientation histogram :
    grad_orient_histo = gradient_orientation_histogram(image, nb_bin=nb_bin, seg_weighted_mask=seg_weighted_mask)
    # Bins of the histogram :
    repr_hist = np.linspace(-(np.pi - 2 * np.pi / nb_bin), (np.pi - 2 * np.pi / nb_bin), nb_bin - 1)
    # Smoothing of the histogram, necessary to avoid digitization effects that will favor angles 0, 45, 90, -45, -90:
    grad_orient_histo_smooth = circular_filter_1d(grad_orient_histo, kmedian_size, kernel='median')  # fft than square than ifft to calculate convolution
    # Computing the circular autoconvolution of the histogram to obtain the axis of symmetry of the histogram :
    grad_orient_histo_conv = circular_conv(grad_orient_histo_smooth, grad_orient_histo_smooth)
    # Restraining angle search to the angle range :
    index_restrain = int(np.ceil(np.true_divide(angle_range, 180) * nb_bin))
    center = (nb_bin - 1) // 2
    grad_orient_histo_conv_restrained = grad_orient_histo_conv[center - index_restrain + 1:center + index_restrain + 1]
    # Finding the symmetry axis by searching for the maximum in the autoconvolution of the histogram :
    index_angle_found = np.argmax(grad_orient_histo_conv_restrained) + (nb_bin // 2 - index_restrain)
    angle_found = repr_hist[index_angle_found] / 2
    angle_found_score = np.amax(grad_orient_histo_conv_restrained)
    # Finding other maxima to compute confidence score
    arg_maxs = argrelmax(grad_orient_histo_conv_restrained, order=kmedian_size, mode='wrap')[0]
    # Confidence score is the ratio of the 2 first maxima :
    if len(arg_maxs) > 1:
        conf_score = angle_found_score / grad_orient_histo_conv_restrained[arg_maxs[1]]
    else:
        conf_score = angle_found_score / np.mean(grad_orient_histo_conv)  # if no other maxima  in the region ratio of the maximum to the mean

    return angle_found, conf_score


def gradient_orientation_histogram(image, nb_bin, seg_weighted_mask=None):
    """ This function takes an image as an input and return its orientation histogram
    inputs :
        - image : the image to compute the orientation histogram from, a 2D numpy array
        - nb_bin : the number of bins of the histogram, an int, for instance 360 for bins 1 degree large (can be more or less than 360)
        - seg_weighted_mask : optional, mask weighting the histogram count, base on segmentation, 2D numpy array between 0 and 1
    outputs :
        - grad_orient_histo : the histogram of the orientations of the image, a 1D numpy array of length nb_bin"""

    h_kernel = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]]) / 4.0
    v_kernel = h_kernel.T

    # Normalization by median, to resolve scaling problems
    median = np.median(image)
    if median != 0:
        image = image / median

    # x and y gradients of the image
    gradx = ndimage.convolve(image, v_kernel)
    grady = ndimage.convolve(image, h_kernel)

    # orientation gradient
    orient = np.arctan2(grady, gradx)  # results are in the range -pi pi

    # weight by gradient magnitude :  this step seems dumb, it alters the angles
    grad_mag = ((np.abs(gradx.astype(object)) ** 2 + np.abs(grady.astype(object)) ** 2) ** 0.5)  # weird data type manipulation, cannot explain why it failed without it
    if np.max(grad_mag) != 0:
        grad_mag = grad_mag / np.max(grad_mag)  # to have map between 0 and 1 (and keep consistency with the seg_weihting map if provided)

    if seg_weighted_mask is not None:
        weighting_map = np.multiply(seg_weighted_mask, grad_mag)  # include weightning by segmentation
    else:
        weighting_map = grad_mag

    # compute histogram :
    grad_orient_histo = np.histogram(np.concatenate(orient), bins=nb_bin - 1, range=(-(np.pi - np.pi / nb_bin), (np.pi - np.pi / nb_bin)),
                                     weights=np.concatenate(weighting_map))

    return grad_orient_histo[0].astype(float)  # return only the values of the bins, not the bins (we know them)


def circular_conv(signal1, signal2):
    """takes two 1D numpy array and do a circular convolution with them
    inputs :
        - signal1 : 1D numpy array
        - signal2 : 1D numpy array, same length as signal1
    output :
        - signal_conv : 1D numpy array, result of circular convolution of signal1 and signal2"""

    if signal1.shape != signal2.shape:
        raise Exception("The two signals for circular convolution do not have the same shape")

    signal2_extended = np.concatenate((signal2, signal2, signal2))  # replicate signal at both ends

    signal_conv_extended = np.convolve(signal1, signal2_extended, mode="same")  # median filtering

    signal_conv = signal_conv_extended[len(signal1):2*len(signal1)]  # truncate back the signal

    return signal_conv


def circular_filter_1d(signal, window_size, kernel='gaussian'):

    """ This function filters circularly the signal inputted with a median filter of inputted size, in this context
    circularly means that the signal is wrapped around and then filtered
    inputs :
        - signal : 1D numpy array
        - window_size : size of the kernel, an int
    outputs :
        - signal_smoothed : 1D numpy array, same size as signal"""

    signal_extended = np.concatenate((signal, signal, signal))  # replicate signal at both ends
    if kernel == 'gaussian':
        signal_extended_smooth = ndimage.gaussian_filter(signal_extended, window_size)  # gaussian
    elif kernel == 'median':
        signal_extended_smooth = medfilt(signal_extended, window_size)  # median filtering
    else:
        raise Exception("Unknow type of kernel")

    signal_smoothed = signal_extended_smooth[len(signal):2*len(signal)]  # truncate back the signal

    return signal_smoothed
