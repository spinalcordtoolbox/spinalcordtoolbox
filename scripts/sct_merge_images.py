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
import sys, os, shutil
from msct_parser import Parser
import sct_utils as sct
import sct_apply_transfo, sct_image, sct_maths

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

    parser.add_option(name="-o",
                      type_value='file_creation',
                      description="Output image",
                      mandatory=False,
                      default_value='merged_images.nii.gz')

    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False)

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
        self.rm_tmp = True
        self.verbose = 1


def compute(list_fname_src, fname_dest, list_fname_warp, fname_out):
    # create temporary folder
    path_tmp = sct.tmp_create()

    #copy input files to tmp folder
    list_fname_src_tmp = []
    for fname_src in list_fname_src:
        fname_src = sct.extract_fname(fname_src)[1:]
        shutil.copy(fname_src, path_tmp + fname_src)
        list_fname_src_tmp.append(fname_src)

    fname_dest_tmp = sct.extract_fname(fname_dest)[1:]
    shutil.copy(fname_dest, path_tmp + fname_dest_tmp)

    list_fname_warp_tmp = []
    for fname_warp in list_fname_warp:
        fname_warp = sct.extract_fname(fname_warp)[1:]
        shutil.copy(fname_warp, path_tmp + fname_warp)
        list_fname_warp_tmp.append(fname_warp)

    # go to tmp folder
    path_wd = os.getwd('.')
    os.chdir(path_tmp)

    # warp src images to dest
    list_fname_reg = warp_images(list_fname_src_tmp, fname_dest_tmp, list_fname_warp_tmp)

    # merge images
    fname_merged = merge_images(list_fname_reg)

    # go back to original working directory
    os.chdir(path_wd)
    shutil.copy(path_tmp+fname_merged, fname_out)



def warp_images(list_fname_src, fname_dest, list_fname_warp, interp='nn'):
    list_fname_out = []
    for fname_src, fname_warp in zip(list_fname_src, list_fname_warp):
        fname_out = sct.add_suffix(fname_src, '_reg')
        sct_apply_transfo.main(args=['-i', fname_src,
                                     '-d', fname_dest,
                                     '-w', fname_warp,
                                     '-x', interp,
                                     '-o', fname_out])
        list_fname_out.append(fname_out)
    return list_fname_out


def merge_images(list_fname_to_merge):
    str_concat = ','.join(list_fname_to_merge)

    # run SCT Image concatenation
    fname_concat = 'concat_image.nii.gz'
    sct_image.main(args=['-i', str_concat,
                         '-concat', 't',
                         '-o', fname_concat])
    # run SCT Math mean
    fname_merged = 'merged_image.nii.gz'
    sct_maths.main(args=['-i', fname_concat,
                         '-mean', 't',
                         '-o', fname_merged])
    return fname_merged

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
    fname_out = arguments["-o"]

    if '-ofolder' in arguments:
        path_results= arguments['-ofolder']
    if '-r' in arguments:
        param.rm_tmp= bool(int(arguments['-r']))
    if '-v' in arguments:
        param.verbose= arguments['-v']

    assert len(list_fname_src) == len(list_fname_warp), "ERROR: list of files are not of the same length"

    compute(list_fname_src, fname_dest, list_fname_warp, fname_out)

if __name__ == "__main__":
    main()
