#!/usr/bin/env python
#########################################################################################
#
# Code for Spline regularization along T during motion correction.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
# Modified: 2014-07-02
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# check if needed Python libraries are already installed or not
import sys
import os
import glob
import commands
import getopt
import scipy.signal
import scipy.fftpack

try:
    import numpy as np
except ImportError:
    print '--- numpy not installed! Exit program. ---'
    sys.exit(2)

#=======================================================================================================================
# main
#=======================================================================================================================

def main():

    folder_mat = ''
    verbose = 0

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            folder_mat = arg
        elif opt in ('-v'):
            verbose = int(arg)

    if folder_mat=='':
        print 'All mandatory arguments are not provided!'
        usage()

    print 'Folder:', folder_mat

    folder_mat = folder_mat + '/'
    nz = len(glob.glob(folder_mat + 'mat.T0_Z*.txt'))
    nt = len(glob.glob(folder_mat + 'mat.T*_Z0.txt'))

    sct_moco_spline(folder_mat,nt,nz,verbose)

#=======================================================================================================================
# sct_moco_spline
#=======================================================================================================================
def sct_moco_spline(folder_mat,nt,nz,verbose):
    print '\n\n\n------------------------------------------------------------------------------'
    print 'Spline Regularization along T: Smoothing Patient Motion...'
    
    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    # append path that contains scripts, to be able to load modules
    sys.path.append(path_sct + '/scripts')
    import sct_utils as sct
    
    fname_mat = [[[] for i in range(nz)] for i in range(nt)]
    for iT in range(nt):
        for iZ in range(nz):
            fname_mat[iT][iZ] = folder_mat + 'mat.T' + str(iT) + '_Z' + str(iZ) + '.txt'

    #Copying the existing Matrices to another folder
    old_mat = folder_mat + 'old/'
    if not os.path.exists(old_mat): os.makedirs(old_mat)    
    cmd = 'cp ' + folder_mat + '*.txt ' + old_mat
    status, output = sct.run(cmd)

    print '\nloading matrices...'
    X = [[[] for i in range(nt)] for i in range(nz)]
    Y = [[[] for i in range(nt)] for i in range(nz)]
    X_smooth = [[[] for i in range(nt)] for i in range(nz)]
    Y_smooth = [[[] for i in range(nt)] for i in range(nz)]
    for iZ in range(nz):
        for iT in range(nt):
            file =  open(fname_mat[iT][iZ])
            Matrix = np.loadtxt(file)
            file.close()

            X[iZ][iT] = Matrix[0,3]
            Y[iZ][iT] = Matrix[1,3]

    # Generate motion splines
    print '\nGenerate motion splines...'
    T = np.arange(nt)
    if verbose==1:
        import pylab as pl

    for iZ in range(nz):

        frequency = scipy.fftpack.fftfreq(len(X[iZ][:]), d=1)
        spectrum = np.abs(scipy.fftpack.fft(X[iZ][:], n=None, axis=-1, overwrite_x=False))
        Wn = np.amax(frequency)/10
        N = 5              #Order of the filter
        b, a = scipy.signal.iirfilter(N, Wn, rp=None, rs=None, btype='low', analog=False, ftype='butter', output='ba')
        X_smooth[iZ][:] = scipy.signal.filtfilt(b, a, X[iZ][:], axis=-1, padtype=None)
                
        if verbose==1:
            pl.plot(T,X_smooth[iZ][:])
            pl.plot(T,X[iZ][:],marker='o',linestyle='None')
            pl.title('X')
            pl.show()

        frequency = scipy.fftpack.fftfreq(len(Y[iZ][:]), d=1)
        spectrum = np.abs(scipy.fftpack.fft(Y[iZ][:], n=None, axis=-1, overwrite_x=False))
        Wn = np.amax(frequency)/10
        N = 5              #Order of the filter
        b, a = scipy.signal.iirfilter(N, Wn, rp=None, rs=None, btype='low', analog=False, ftype='butter', output='ba')
        Y_smooth[iZ][:] = scipy.signal.filtfilt(b, a, Y[iZ][:], axis=-1, padtype=None)

        if verbose==1:
            pl.plot(T,Y_smooth[iZ][:])
            pl.plot(T,Y[iZ][:],marker='*', linestyle='None')
            pl.title('Y')
            pl.show()

    #Storing the final Matrices
    print '\nStoring the final Matrices...'
    for iZ in range(nz):
        for iT in range(nt):
            file =  open(fname_mat[iT][iZ])
            Matrix = np.loadtxt(file)
            file.close()
            
            Matrix[0,3] = X_smooth[iZ][iT]
            Matrix[1,3] = Y_smooth[iZ][iT]
            
            file =  open(fname_mat[iT][iZ],'w')
            np.savetxt(fname_mat[iT][iZ], Matrix, fmt='%.9e', delimiter='  ', newline='\n', header='', footer='', comments='#')
            file.close()

    print '\n...Done. Patient motion has been smoothed'
    print '------------------------------------------------------------------------------\n'

#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        ' Spline regularization along T \n' \
        '\nUSAGE: \n' \
        '  '+os.path.basename(__file__)+' -i <folder_path> \n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        '  -i           folder_path \n' \
        '\n'\
        'OPTIONAL ARGUMENTS\n' \
        '  -v           Set verbose=1 for plotting graphs. Default value is 0\n' \
        '  -h           help. Show this message.\n' \
        '\n'\
        'EXAMPLE:\n' \
        '  '+os.path.basename(__file__)+' -i mat_final -v 1 \n'
    
    #Exit Program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()