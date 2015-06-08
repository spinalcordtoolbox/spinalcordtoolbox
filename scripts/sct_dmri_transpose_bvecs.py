#!/usr/bin/env python
#=======================================================================================================================
# Convert bvecs file to column, in case they are in line.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad
# Modified: 2014-09-01
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import os


# init
#fname_in = '/Users/julien/MRI/david_cadotte/2013-10-19_multiparametric/bvecs.txt'


# Extracts path, file and extension
#=======================================================================================================================
def extract_fname(fname):
    # extract path
    path_fname = os.path.dirname(fname)+'/'
    # check if only single file was entered (without path)
    if path_fname == '/':
        path_fname = ''
    # extract file and extension
    file_fname = fname
    file_fname = file_fname.replace(path_fname,'')
    file_fname, ext_fname = os.path.splitext(file_fname)
    # check if .nii.gz file
    if ext_fname == '.gz':
        file_fname = file_fname[0:len(file_fname)-4]
        ext_fname = ".nii.gz"
    return path_fname, file_fname, ext_fname


# Usage
#=======================================================================================================================
def usage():
    print '\n' \
    ''+os.path.basename(__file__)+'\n' \
    '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
    'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
    '\n'\
    'DESCRIPTION\n' \
    '  Convert bvecs file to column, in case they are in line.\n' \
    '\n' \
    'USAGE\n' \
    '  sct_dmri_transpose_bvecs -i <bvecs> \n' \
    '\n' \
    'MANDATORY ARGUMENTS\n' \
    '  -i <txt_file>        bvecs file.\n' \
    '\n' \
    'OPTIONAL ARGUMENTS\n' \
    '  -h                help. Show this message.\n' \
    '\n' \
    'EXAMPLE\n' \
    '  sct_dmri_transpose_bvecs -i bvec.txt \n' \

    sys.exit(2)



#=======================================================================================================================
# Main
#=======================================================================================================================


def main():
    # Check inputs
    path_func, file_func, ext_func = extract_fname(sys.argv[0])
    if len(sys.argv) < 2:
        usage()
    fname_in = sys.argv[1]

    # Extracts path, file and extension
    path_in, file_in, ext_in = extract_fname(fname_in)

    # read ASCII file
    print('Read file...')
    text_file = open(fname_in, 'r')
    bvecs = text_file.readlines()
    text_file.close()

    # Parse each line
    # TODO: find a better way to do it, maybe with string or numpy...
    lin0 = bvecs[0].split()
    lin1 = bvecs[1].split()
    lin2 = bvecs[2].split()

    # Write new file
    print('Transpose bvecs...')
    fname_out = path_in+file_in+'_t'+ext_in
    fid = open(fname_out,'w')
    for iCol in xrange(0, len(lin0)):
        fid.write(lin0[iCol]+' '+lin1[iCol]+' '+lin2[iCol]+'\n')
    fid.close()

    # Display
    print('File created: '+fname_out)

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()
