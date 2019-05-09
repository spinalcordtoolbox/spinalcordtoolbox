#!/usr/bin/env python
#
# This program straightens the spinal cord of an anatomic image, apply a smoothing in the z dimension and apply
# the inverse warping field to get back the curved spinal cord but smoothed.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Simon Levy
# Modified: 2014-09-01
#
# About the license: see the file LICENSE.TXT
#########################################################################################


# TODO: maybe no need to convert RPI at the beginning because strainghten spinal cord already does it!

from __future__ import absolute_import

import sys, os, time

import numpy as np

import sct_utils as sct
import sct_maths
import spinalcordtoolbox.image as msct_image
from sct_convert import convert
from msct_parser import Parser

# PARAMETERS
class Param:
    # The constructor
    def __init__(self):
        self.algo_fitting = 'bspline'  # Fitting algorithm for centerline. See sct_straighten_spinalcord.

    # update constructor with user's parameters
    def update(self, param_user):
        # list_objects = param_user.split(',')
        for object in param_user:
            if len(object) < 2:
                sct.printv('ERROR: Wrong usage.', 1, type='error')
            obj = object.split('=')
            setattr(self, obj[0], obj[1])


# PARSER
# ==========================================================================================
def get_parser():
    # Initialize the parser
    parser = Parser(__file__)

    # initialize default parameters
    param_default = Param()

    parser.usage.set_description('Smooth the spinal cord along its centerline. Steps are:\n'
                                 '1) Spinal cord is straightened (using centerline),\n'
                                 '2) a Gaussian kernel is applied in the superior-inferior direction,\n'
                                 '3) then cord is de-straightened as originally.\n')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image to smooth",
                      mandatory=True,
                      example='data.nii.gz')
    parser.add_option(name="-s",
                      type_value="file",
                      description="Spinal cord centerline or segmentation",
                      mandatory=True,
                      example='data_centerline.nii.gz')
    parser.add_option(name="-c",
                      type_value=None,
                      description="Spinal cord centerline or segmentation",
                      mandatory=False,
                      deprecated_by='-s')
    parser.add_option(name="-smooth",
                      type_value=[[','], 'float'],
                      description='Sigma (standard deviation) of the smoothing Gaussian kernel (in mm). For isotropic '
                                  'smoothing you only need to specify a value (e.g. 2). For anisotropic smoothing '
                                  'specify a value for each axis, separated with a comma. The order should follow axes '
                                  'Right-Left, Antero-Posterior, Superior-Inferior (e.g.: 1,1,3). For no smoothing, set '
                                  'value to 0.',
                      mandatory=False,
                      default_value=[0, 0, 3])
    parser.add_option(name='-param',
                      type_value=[[','], 'str'],
                      description="Advanced parameters. Assign value with \"=\"; Separate params with \",\"\n"
                                  "algo_fitting {bspline, polyfit}: Algorithm for curve fitting. For more information, see sct_straighten_spinalcord. Default="+ param_default.algo_fitting + ".\n",
                      mandatory=False)
    parser.usage.addSection('MISC')
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')
    return parser


# MAIN
# ==========================================================================================
def main(args=None):

    # Initialization
    param = Param()
    start_time = time.time()

    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    fname_anat = arguments['-i']
    fname_centerline = arguments['-s']
    if '-smooth' in arguments:
        sigma = arguments['-smooth']
    if '-param' in arguments:
        param.update(arguments['-param'])
    if '-r' in arguments:
        remove_temp_files = int(arguments['-r'])
    verbose = int(arguments.get('-v'))
    sct.init_sct(log_level=verbose, update=True)  # Update log level

    # Display arguments
    sct.printv('\nCheck input arguments...')
    sct.printv('  Volume to smooth .................. ' + fname_anat)
    sct.printv('  Centerline ........................ ' + fname_centerline)
    sct.printv('  Sigma (mm) ........................ ' + str(sigma))
    sct.printv('  Verbose ........................... ' + str(verbose))

    # Check that input is 3D:
    from spinalcordtoolbox.image import Image
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_anat).dim
    dim = 4  # by default, will be adjusted later
    if nt == 1:
        dim = 3
    if nz == 1:
        dim = 2
    if dim == 4:
        sct.printv('WARNING: the input image is 4D, please split your image to 3D before smoothing spinalcord using :\n'
                   'sct_image -i ' + fname_anat + ' -split t -o ' + fname_anat, verbose, 'warning')
        sct.printv('4D images not supported, aborting ...', verbose, 'error')

    # Extract path/file/extension
    path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)
    path_centerline, file_centerline, ext_centerline = sct.extract_fname(fname_centerline)

    path_tmp = sct.tmp_create(basename="smooth_spinalcord", verbose=verbose)

    # Copying input data to tmp folder
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    sct.copy(fname_anat, os.path.join(path_tmp, "anat" + ext_anat))
    sct.copy(fname_centerline, os.path.join(path_tmp, "centerline" + ext_centerline))

    # go to tmp folder
    curdir = os.getcwd()
    os.chdir(path_tmp)

    # convert to nii format
    convert('anat' + ext_anat, 'anat.nii')
    convert('centerline' + ext_centerline, 'centerline.nii')

    # Change orientation of the input image into RPI
    sct.printv('\nOrient input volume to RPI orientation...')
    fname_anat_rpi = msct_image.Image("anat.nii") \
     .change_orientation("RPI", generate_path=True) \
     .save() \
     .absolutepath

    # Change orientation of the input image into RPI
    sct.printv('\nOrient centerline to RPI orientation...')
    fname_centerline_rpi = msct_image.Image("centerline.nii") \
     .change_orientation("RPI", generate_path=True) \
     .save() \
     .absolutepath

    # Straighten the spinal cord
    # straighten segmentation
    sct.printv('\nStraighten the spinal cord using centerline/segmentation...', verbose)
    cache_sig = sct.cache_signature(input_files=[fname_anat_rpi, fname_centerline_rpi],
                                    input_params={"x": "spline"})
    cachefile = os.path.join(curdir, "straightening.cache")
    if sct.cache_valid(cachefile, cache_sig) and os.path.isfile(os.path.join(curdir, 'warp_curve2straight.nii.gz')) and os.path.isfile(os.path.join(curdir, 'warp_straight2curve.nii.gz')) and os.path.isfile(os.path.join(curdir, 'straight_ref.nii.gz')):
        # if they exist, copy them into current folder
        sct.printv('Reusing existing warping field which seems to be valid', verbose, 'warning')
        sct.copy(os.path.join(curdir, 'warp_curve2straight.nii.gz'), 'warp_curve2straight.nii.gz')
        sct.copy(os.path.join(curdir, 'warp_straight2curve.nii.gz'), 'warp_straight2curve.nii.gz')
        sct.copy(os.path.join(curdir, 'straight_ref.nii.gz'), 'straight_ref.nii.gz')
        # apply straightening
        sct.run(['sct_apply_transfo', '-i', fname_anat_rpi, '-w', 'warp_curve2straight.nii.gz', '-d', 'straight_ref.nii.gz', '-o', 'anat_rpi_straight.nii', '-x', 'spline'], verbose)
    else:
        sct.run(['sct_straighten_spinalcord', '-i', fname_anat_rpi, '-o', 'anat_rpi_straight.nii', '-s', fname_centerline_rpi, '-x', 'spline', '-param', 'algo_fitting='+param.algo_fitting], verbose)
        sct.cache_save(cachefile, cache_sig)
        # move warping fields locally (to use caching next time)
        sct.copy('warp_curve2straight.nii.gz', os.path.join(curdir, 'warp_curve2straight.nii.gz'))
        sct.copy('warp_straight2curve.nii.gz', os.path.join(curdir, 'warp_straight2curve.nii.gz'))

    # Smooth the straightened image along z
    sct.printv('\nSmooth the straightened image...')
    sigma_smooth = ",".join([str(i) for i in sigma])
    sct_maths.main(args=['-i', 'anat_rpi_straight.nii',
                         '-smooth', sigma_smooth,
                         '-o', 'anat_rpi_straight_smooth.nii',
                         '-v', '0'])
    # Apply the reversed warping field to get back the curved spinal cord
    sct.printv('\nApply the reversed warping field to get back the curved spinal cord...')
    sct.run(['sct_apply_transfo', '-i', 'anat_rpi_straight_smooth.nii', '-o', 'anat_rpi_straight_smooth_curved.nii', '-d', 'anat.nii', '-w', 'warp_straight2curve.nii.gz', '-x', 'spline'], verbose)

    # replace zeroed voxels by original image (issue #937)
    sct.printv('\nReplace zeroed voxels by original image...', verbose)
    nii_smooth = Image('anat_rpi_straight_smooth_curved.nii')
    data_smooth = nii_smooth.data
    data_input = Image('anat.nii').data
    indzero = np.where(data_smooth == 0)
    data_smooth[indzero] = data_input[indzero]
    nii_smooth.data = data_smooth
    nii_smooth.save('anat_rpi_straight_smooth_curved_nonzero.nii')

    # come back
    os.chdir(curdir)

    # Generate output file
    sct.printv('\nGenerate output file...')
    sct.generate_output_file(os.path.join(path_tmp, "anat_rpi_straight_smooth_curved_nonzero.nii"),
                             file_anat + '_smooth' + ext_anat)

    # Remove temporary files
    if remove_temp_files == 1:
        sct.printv('\nRemove temporary files...')
        sct.rmtree(path_tmp)

    # Display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: ' + str(int(np.round(elapsed_time))) + 's\n')

    sct.display_viewer_syntax([file_anat, file_anat + '_smooth'], verbose=verbose)


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    main()
