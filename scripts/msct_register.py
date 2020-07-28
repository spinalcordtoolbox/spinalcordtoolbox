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
from spinalcordtoolbox.register.register import *

import sct_utils as sct
import sct_apply_transfo
import sct_concat_transfo
from sct_convert import convert
from sct_image import split_data, concat_warp2d
from msct_register_landmarks import register_landmarks

logger = logging.getLogger(__name__)

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
        step = paramregmulti.steps[str(i_step)]
        sct.printv('\n--\nESTIMATE TRANSFORMATION FOR STEP #' + str(i_step), param.verbose)
        # identify which is the src and dest
        if step.type == 'im':
            src = ['src.nii']
            dest = ['dest_RPI.nii']
            interp_step = ['spline']
        elif step.type == 'seg':
            src = ['src_seg.nii']
            dest = ['dest_seg_RPI.nii']
            interp_step = ['nn']
        elif step.type == 'imseg':
            src = ['src.nii', 'src_seg.nii']
            dest = ['dest_RPI.nii', 'dest_seg_RPI.nii']
            interp_step = ['spline', 'nn']
        elif step.type == 'label':
            src = ['src_label.nii']
            dest = ['dest_label_RPI.nii']
            interp_step = ['nn']
        else:
            sct.printv('ERROR: Wrong image type: {}'.format(step.type), 1, 'error')

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
        warp_forward_out, warp_inverse_out = register(src=src, dest=dest, step=step, param=param)

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
    sct.generate_output_file(os.path.join(path_tmp, "warp_src2dest.nii.gz"), fname_output_warp, verbose=param.verbose)

    # generate: inverse warping field
    if generate_warpinv:
        fname_output_warpinv = os.path.join(path_out, 'warp_' + file_dest + '2' + file_src + '.nii.gz')
        sct.generate_output_file(os.path.join(path_tmp, "warp_dest2src.nii.gz"), fname_output_warpinv, verbose=param.verbose)
    else:
        fname_output_warpinv = None

    # Delete temporary files
    if param.remove_temp_files:
        sct.printv('\nRemove temporary files...', param.verbose)
        sct.rmtree(path_tmp, verbose=param.verbose)

    return fname_src2dest, fname_dest2src, fname_output_warp, fname_output_warpinv


# register images
# ==========================================================================================
def register(src, dest, step, param):
    """
    Register src onto dest image. Output affine transformations that need to be inverted will have the prefix "-".
    """
    # initiate default parameters of antsRegistration transformation
    ants_registration_params = {'rigid': '', 'affine': '', 'compositeaffine': '', 'similarity': '', 'translation': '',
                                'bspline': ',10', 'gaussiandisplacementfield': ',3,0',
                                'bsplinedisplacementfield': ',5,10', 'syn': ',3,0', 'bsplinesyn': ',1,3'}

    output = ''  # default output if problem

    # If the input type is either im or seg, we can convert the input list into a string for improved code clarity
    if not step.type == 'imseg':
        src = src[0]
        dest = dest[0]

    # display arguments
    sct.printv('Registration parameters:', param.verbose)
    sct.printv('  type ........... ' + step.type, param.verbose)
    sct.printv('  algo ........... ' + step.algo, param.verbose)
    sct.printv('  slicewise ...... ' + step.slicewise, param.verbose)
    sct.printv('  metric ......... ' + step.metric, param.verbose)
    sct.printv('  iter ........... ' + step.iter, param.verbose)
    sct.printv('  smooth ......... ' + step.smooth, param.verbose)
    sct.printv('  laplacian ...... ' + step.laplacian, param.verbose)
    sct.printv('  shrink ......... ' + step.shrink, param.verbose)
    sct.printv('  gradStep ....... ' + step.gradStep, param.verbose)
    sct.printv('  deformation .... ' + step.deformation, param.verbose)
    sct.printv('  init ........... ' + step.init, param.verbose)
    sct.printv('  poly ........... ' + step.poly, param.verbose)
    sct.printv('  filter_size .... ' + str(step.filter_size), param.verbose)
    sct.printv('  dof ............ ' + step.dof, param.verbose)
    sct.printv('  smoothWarpXY ... ' + step.smoothWarpXY, param.verbose)
    sct.printv('  rot_method ..... ' + step.rot_method, param.verbose)

    # set metricSize
    if step.metric == 'MI':
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

    # # landmark-based registration
    if step.type in ['label']:
        warp_forward_out, warp_inverse_out = register_step_label(
         src=src,
         dest=dest,
         step=step,
         verbose=param.verbose,
        )

    elif step.algo == 'slicereg':
        warp_forward_out, warp_inverse_out = register_step_ants_slice_regularized_registration(
         src=src,
         dest=dest,
         step=step,
         metricSize=metricSize,
         fname_mask=fname_mask,
         verbose=param.verbose,
        )

    # ANTS 3d
    elif step.algo.lower() in ants_registration_params and step.slicewise == '0': # FIXME [AJ]
        warp_forward_out, warp_inverse_out = register_step_ants_registration(
         src=src,
         dest=dest,
         step=step,
         masking=masking,
         ants_registration_params=ants_registration_params,
         padding=param.padding,
         metricSize=metricSize,
         verbose=param.verbose,
        )

    # ANTS 2d
    elif step.algo.lower() in ants_registration_params and step.slicewise == '1': # FIXME [AJ]
        warp_forward_out, warp_inverse_out = register_step_slicewise_ants(
         src=src,
         dest=dest,
         step=step,
         ants_registration_params=ants_registration_params,
         fname_mask=fname_mask,
         remove_temp_files=param.remove_temp_files,
         verbose=param.verbose,
        )

    # slice-wise transfo
    elif step.algo in ['centermass', 'centermassrot', 'columnwise']:
        # check if user provided a mask-- if so, inform it will be ignored
        if fname_mask:
            sct.printv('\nWARNING: algo ' + step.algo + ' will ignore the provided mask.\n', 1, 'warning')

        warp_forward_out, warp_inverse_out = register_step_slicewise(
         src=src,
         dest=dest,
         step=step,
         ants_registration_params=ants_registration_params,
         remove_temp_files=param.remove_temp_files,
         verbose=param.verbose,
        )

    else:
        sct.printv('\nERROR: algo ' + step.algo + ' does not exist. Exit program\n', 1, 'error')

    if not os.path.isfile(warp_forward_out):
        # no forward warping field for rigid and affine
        sct.printv('\nERROR: file ' + warp_forward_out + ' doesn\'t exist (or is not a file).\n' + output +
                   '\nERROR: ANTs failed. Exit program.\n', 1, 'error')
    elif not os.path.isfile(warp_inverse_out) and \
            step.algo not in ['rigid', 'affine', 'translation'] and \
            step.type not in ['label']:
        # no inverse warping field for rigid and affine
        sct.printv('\nERROR: file ' + warp_inverse_out + ' doesn\'t exist (or is not a file).\n' + output +
                   '\nERROR: ANTs failed. Exit program.\n', 1, 'error')
    else:
        # rename warping fields
        if (step.algo.lower() in ['rigid', 'affine', 'translation'] and
                step.slicewise == '0'):
            # if ANTs is used with affine/rigid --> outputs .mat file
            warp_forward = 'warp_forward_' + str(step.step) + '.mat'
            os.rename(warp_forward_out, warp_forward)
            warp_inverse = '-warp_forward_' + str(step.step) + '.mat'
        elif step.type in ['label']:
            # if label-based registration is used --> outputs .txt file
            warp_forward = 'warp_forward_' + str(step.step) + '.txt'
            os.rename(warp_forward_out, warp_forward)
            warp_inverse = '-warp_forward_' + str(step.step) + '.txt'
        else:
            warp_forward = 'warp_forward_' + str(step.step) + '.nii.gz'
            warp_inverse = 'warp_inverse_' + str(step.step) + '.nii.gz'
            os.rename(warp_forward_out, warp_forward)
            os.rename(warp_inverse_out, warp_inverse)

    return warp_forward, warp_inverse