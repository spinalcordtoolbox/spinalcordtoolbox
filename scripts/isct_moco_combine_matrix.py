#!/usr/bin/env python
#########################################################################################
#
# Code for combining matrices generated during motion correction.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
# Modified: 2014-07-02
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# check if needed Python libraries are already installed or not
import os
import getopt
import commands
import sys

try:
    import numpy as np
except ImportError:
    print '--- numpy not installed! Exit program. ---'
    sys.exit(2)

#=======================================================================================================================
# main
#=======================================================================================================================

def main():

    mat_2_combine = ''
    mat_final     = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:c:f:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-c'):
            mat_2_combine = arg
        elif opt in ('-f'):
            mat_final = arg

    # display usage if a mandatory argument is not provided
    if mat_2_combine=='' or mat_final=='':
        print '\n\nAll mandatory arguments are not provided \n'
        usage()

    sct_moco_combine_matrix(mat_2_combine,mat_final)


#=======================================================================================================================
# sct_moco_combine_matrix
#=======================================================================================================================
def sct_moco_combine_matrix(mat_2_combine,mat_final,verbose):

    if verbose:
        print '\n Combining Matrices...'
        print '------------------------------------------------------------------------------\n'

    m2c_fnames = [ fname for fname in os.listdir(mat_2_combine) if os.path.isfile(os.path.join(mat_2_combine,fname)) ]
    for fname in m2c_fnames:
        if os.path.isfile(os.path.join(mat_final,fname)):
            file =  open(os.path.join(mat_2_combine,fname))
            Matrix_m2c = np.loadtxt(file)
            file.close()
        
            file =  open(os.path.join(mat_final,fname))
            Matrix_f = np.loadtxt(file)
            file.close()
            Matrix_final = np.identity(4)
            Matrix_final[0:3,0:3] = Matrix_f[0:3,0:3]*Matrix_m2c[0:3,0:3]
            Matrix_final[0,3] = Matrix_f[0,3] + Matrix_m2c[0,3]
            Matrix_final[1,3] = Matrix_f[1,3] + Matrix_m2c[1,3]
            
            file =  open(os.path.join(mat_final,fname),'w')
            np.savetxt(os.path.join(mat_final,fname), Matrix_final, fmt="%s", delimiter='  ', newline='\n')
            file.close()

    if verbose:
        print '\n...done. Matrices are combined...'
        print '------------------------------------------------------------------------------\n'

#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print '\n' \
        '  '+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Combines the matrices.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -c <mat_2_combine> -f <mat_final>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -c                    matrix folder to combine with final matrix folder\n' \
        '  -f                    final matrix folder\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -h                    help. Show this message.\n' \
        '\n'\
        'EXAMPLE:\n' \
        '  '+os.path.basename(__file__)+' -c b0groups_param.mat -f mat_final \n'
    
    #Exit Program
    sys.exit(2)

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()