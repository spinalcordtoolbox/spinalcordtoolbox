#!/usr/bin/env python
#########################################################################################
# Convert bvecs file to column, in case they are in line.
#
#
# USAGE
# ---------------------------------------------------------------------------------------
#   sct_dmri_transpose_bvecs.py <bvecs>
#
#
# INPUT
# ---------------------------------------------------------------------------------------
# bvecs             bvecs ASCII file (FSL format).
#
#
# OUTPUT
# ---------------------------------------------------------------------------------------
# bvecs_col         bvecs in column.
#
#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
# EXTERNAL PYTHON PACKAGES
# none
#
# EXTERNAL SOFTWARE
# none
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
# Author: Julien Cohen-Adad
# Modified: 2013-10-19
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#########################################################################################

import sys
import os


# init
#fname_in = '/Users/julien/MRI/david_cadotte/2013-10-19_multiparametric/bvecs.txt'


# Extracts path, file and extension
#########################################################################################
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
#########################################################################################


# MAIN
#########################################################################################

# Check inputs
path_func, file_func, ext_func = extract_fname(sys.argv[0])
if len(sys.argv) < 2:
    print 'Usage: '+file_func+ext_func+' <bvecs>'
    sys.exit(1)
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

