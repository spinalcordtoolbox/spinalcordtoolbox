#!/usr/bin/env python
#######################################################################################################################
#
# Merge images. See details in function "merge_images".
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Dominique Eden, Sara Dupont
# Modified: 2017-03-17
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

# TODO: parameter "almost_zero" might case problem if merging data with very low values (e.g. MD from diffusion)

# Python imports
import sys
import os
import shutil
import numpy as np
# SCT imports
from msct_parser import Parser
import sct_utils as sct
import sct_apply_transfo
import msct_image
import sct_maths


class Param:
    def __init__(self):
        self.fname_out = 'merged_images.nii.gz'
        self.interp = 'linear'
        self.rm_tmp = True
        self.verbose = 1
        self.almost_zero = 0.00000001


# PARSER
# ==========================================================================================
def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Merge images to the same space')
    parser.add_option(name="-i",
                      type_value=[[','], 'file'],
                      description="Input images",
                      mandatory=True)
    parser.add_option(name="-d",
                      type_value='file',
                      description="Destination image",
                      mandatory=True)
    parser.add_option(name="-w",
                      type_value=[[','], 'file'],
                      description="List of warping fields from input images to destination image",
                      mandatory=True)
    parser.add_option(name="-x",
                      type_value='str',
                      description="interpolation for warping the input images to the destination image",
                      mandatory=False,
                      default_value=Param().interp)

    parser.add_option(name="-o",
                      type_value='file_output',
                      description="Output image",
                      mandatory=False,
                      default_value=Param().fname_out)

    '''
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False)
    '''
    parser.usage.addSection('MISC')
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value=str(int(Param().rm_tmp)),
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="Verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value=str(Param().verbose))

    return parser


def merge_images(list_fname_src, fname_dest, list_fname_warp, param):
    """
    Merge multiple source images onto destination space. All images are warped to the destination space and then added.
    To deal with overlap during merging (e.g. one voxel in destination image is shared with two input images), the
    resulting voxel is divided by the sum of the partial volume of each image. For example, if src(x,y,z)=1 is mapped to
    dest(i,j,k) with a partial volume of 0.5 (because destination voxel is bigger), then its value after linear interpolation
    will be 0.5. To account for partial volume, the resulting voxel will be: dest(i,j,k) = 0.5*0.5/0.5 = 0.5.
    Now, if two voxels overlap in the destination space, let's say: src(x,y,z)=1 and src2'(x',y',z')=1, then the
    resulting value will be: dest(i,j,k) = (0.5*0.5 + 0.5*0.5) / (0.5+0.5) = 0.5. So this function acts like a weighted
    average operator, only in destination voxels that share multiple source voxels.

    Parameters
    ----------
    list_fname_src
    fname_dest
    list_fname_warp
    param

    Returns
    -------

    """

    # create temporary folder
    path_tmp = sct.tmp_create()

    # get dimensions of destination file
    nii_dest = msct_image.Image(fname_dest)

    # initialize variables
    data = np.zeros([nii_dest.dim[0], nii_dest.dim[1], nii_dest.dim[2], len(list_fname_src)])
    partial_volume = np.zeros([nii_dest.dim[0], nii_dest.dim[1], nii_dest.dim[2], len(list_fname_src)])
    data_merge = np.zeros([nii_dest.dim[0], nii_dest.dim[1], nii_dest.dim[2]])

    # loop across files
    i_file = 0
    for fname_src in list_fname_src:

        # apply transformation src --> dest
        sct_apply_transfo.main(args=[
            '-i', fname_src,
            '-d', fname_dest,
            '-w', list_fname_warp[i_file],
            '-x', param.interp,
            '-o', 'src_' + str(i_file) + '_template.nii.gz',
            '-v', param.verbose])

        # create binary mask from input file by assigning one to all non-null voxels
        sct_maths.main(args=[
            '-i', fname_src,
            '-bin', str(param.almost_zero),
            '-o', 'src_' + str(i_file) + 'native_bin.nii.gz'])

        # apply transformation to binary mask to compute partial volume
        sct_apply_transfo.main(args=[
            '-i', 'src_' + str(i_file) + 'native_bin.nii.gz',
            '-d', fname_dest,
            '-w', list_fname_warp[i_file],
            '-x', param.interp,
            '-o', 'src_' + str(i_file) + '_template_partialVolume.nii.gz'])

        # open data
        data[:, :, :, i_file] = msct_image.Image('src_' + str(i_file) + '_template.nii.gz').data
        partial_volume[:, :, :, i_file] = msct_image.Image('src_' + str(i_file) + '_template_partialVolume.nii.gz').data
        i_file += 1

    # merge files using partial volume information (and convert nan resulting from division by zero to zeros)
    data_merge = np.divide(np.sum(data * partial_volume, axis=3), np.sum(partial_volume, axis=3))
    data_merge = np.nan_to_num(data_merge)

    # write result in file
    nii_dest.data = data_merge
    nii_dest.setFileName(param.fname_out)
    nii_dest.save()

    # remove temporary folder
    if param.rm_tmp:
        shutil.rmtree(path_tmp)


# MAIN
# ==========================================================================================
def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # create param objects
    param = Param()

    # get parser
    parser = get_parser()
    arguments = parser.parse(args)

    # set param arguments ad inputted by user
    list_fname_src = arguments["-i"]
    fname_dest = arguments["-d"]
    list_fname_warp = arguments["-w"]
    param.fname_out = arguments["-o"]

    # if '-ofolder' in arguments:
    #     path_results = arguments['-ofolder']
    if '-x' in arguments:
        param.interp = arguments['-x']
    if '-r' in arguments:
        param.rm_tmp = bool(int(arguments['-r']))
    if '-v' in arguments:
        param.verbose = arguments['-v']

    # check if list of input files and warping fields have same length
    assert len(list_fname_src) == len(list_fname_warp), "ERROR: list of files are not of the same length"

    # merge src images to destination image
    try:
        merge_images(list_fname_src, fname_dest, list_fname_warp, param)
    except Exception as e:
        sct.printv(str(e), 1, 'error')

    sct.printv('Done ! to view your results, type: ', param.verbose, 'normal')
    sct.printv('fslview ' + fname_dest + ' ' + os.path.abspath(param.fname_out) + ' &\n', param.verbose, 'info')

if __name__ == "__main__":
    sct.start_stream_logger()
    main()
