#!/usr/bin/env python
#######################################################################################################################
#
#
# Merge images
#
# ----------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Dominique Eden, Sara Dupont
# Modified: 2017-03-17
#
# About the license: see the file LICENSE.TXT
########################################################################################################################

# TODO: do averaging only in overlapping voxels (ie: excluding zero voxels), otherwise the mean is wrong.
# TODO: in compute: do NOT copy files like that, because if using nii this will fail. use "convert" or do not copy
# TODO: do not copy certain files that are now opened and processed with numpy (save time)
# TODO: parameter "almost_zero" might case problem if merging data with very low values (e.g. MD from diffusion)

# Python imports
import sys
import shutil
import numpy as np
# SCT imports
from msct_parser import Parser
import sct_utils as sct
import sct_apply_transfo
import sct_image
import sct_maths

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


class Param:
    def __init__(self):
        self.fname_out = 'merged_images.nii.gz'
        self.interp = 'nn'
        self.rm_tmp = True
        self.verbose = 1
        self.almost_zero = 0.00000001


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
    nii_dest = sct_image.Image(fname_dest)

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
            '-o', 'src_'+str(i_file)+'_template.nii.gz',
            '-v', param.verbose])

        # create binary mask from input file by assigning one to all non-null voxels
        sct_maths.main(args=[
            '-i', fname_src,
            '-bin', str(param.almost_zero),
            '-o', 'src_'+str(i_file)+'native_bin.nii.gz'])

        # apply transformation to binary mask to compute partial volume
        sct_apply_transfo.main(args=[
            '-i', 'src_'+str(i_file)+'native_bin.nii.gz',
            '-d', fname_dest,
            '-w', list_fname_warp[i_file],
            '-x', param.interp,
            '-o', 'src_'+str(i_file)+'_template_partialVolume.nii.gz'])

        # open data
        data[:, :, :, i_file] = sct_image.Image('src_'+str(i_file)+'_template.nii.gz').data
        partial_volume[:, :, :, i_file] = sct_image.Image('src_'+str(i_file)+'_template_partialVolume.nii.gz').data
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

    #
    #
    # #copy input files to tmp folder
    # list_fname_src_tmp = []
    # for i, fname_src in enumerate(list_fname_src):
    #     fname_src_new= 'input_'+str(i)+'.nii.gz'
    #     shutil.copy(fname_src, path_tmp + fname_src_new)
    #     list_fname_src_tmp.append(fname_src_new)
    #
    # fname_dest_tmp = 'dest.nii.gz'
    # shutil.copy(fname_dest, path_tmp + fname_dest_tmp)
    #
    # list_fname_warp_tmp = []
    # for i, fname_warp in enumerate(list_fname_warp):
    #     fname_warp_new = 'warp_'+str(i)+'.nii.gz'
    #     shutil.copy(fname_warp, path_tmp + fname_warp_new)
    #     list_fname_warp_tmp.append(fname_warp_new)
    #
    # # go to tmp folder
    # path_wd = os.getcwd()
    # os.chdir(path_tmp)
    #
    # # warp src images to dest
    # list_fname_reg = warp_images(list_fname_src_tmp, fname_dest_tmp, list_fname_warp_tmp, interp=param.interp, param=param)
    #
    # # merge images
    # fname_merged = merge_images(list_fname_reg, param=param)
    #
    # # go back to original working directory
    # os.chdir(path_wd)
    # sct.generate_output_file(path_tmp+fname_merged, param.fname_out)

#
#
# def warp_images(list_fname_src, fname_dest, list_fname_warp, interp='nn', param=Param()):
#     list_fname_out = []
#     for fname_src, fname_warp in zip(list_fname_src, list_fname_warp):
#         fname_out = sct.add_suffix(fname_src, '_reg')
#         sct_apply_transfo.main(args=['-i', fname_src,
#                                      '-d', fname_dest,
#                                      '-w', fname_warp,
#                                      '-x', interp,
#                                      '-o', fname_out,
#                                      '-v', param.verbose])
#         list_fname_out.append(fname_out)
#     return list_fname_out

#
# def merge_images(list_fname_to_merge, param=Param()):
#
#
#
#
#     str_concat = ','.join(list_fname_to_merge)
#
#     # run SCT Image concatenation
#     fname_concat = 'concat_image.nii.gz'
#     sct_image.main(args=['-i', str_concat,
#                          '-concat', 't',
#                          '-o', fname_concat,
#                          '-v', param.verbose])
#     # run SCT Math mean
#     fname_merged = 'merged_image.nii.gz'
#     sct_maths.main(args=['-i', fname_concat,
#                          '-mean', 't',
#                          '-o', fname_merged,
#                          '-v', param.verbose])
#     return fname_merged

########################################################################################################################
# ------------------------------------------------------  MAIN ------------------------------------------------------- #
########################################################################################################################

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # create param objects
    param = Param()

    # get parser
    parser = get_parser()
    arguments = parser.parse(args)

    # set param arguments ad inputted by user
    list_fname_src= arguments["-i"]
    fname_dest = arguments["-d"]
    list_fname_warp = arguments["-w"]
    param.fname_out = arguments["-o"]

    if '-ofolder' in arguments:
        path_results= arguments['-ofolder']
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
    sct.printv('fslview '+param.fname_out+' &\n', param.verbose, 'info')

if __name__ == "__main__":
    main()
