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
        opts, args = getopt.getopt(sys.argv[1:],'hi:g:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            folder_mat = arg
        elif opt in ('-g'):
            graph = arg
        elif opt in ('-v'):
            verbose = int(arg)

    if folder_mat=='':
        print 'All mandatory arguments are not provided!'
        usage()

    print 'Folder:', folder_mat

    folder_mat = folder_mat + '/'
    nz = len(glob.glob(folder_mat + 'mat.T0_Z*.txt'))
    nt = len(glob.glob(folder_mat + 'mat.T*_Z0.txt'))

    sct_moco_spline(folder_mat,nt,nz,verbose,graph)

#=======================================================================================================================
# sct_moco_spline
#=======================================================================================================================
def sct_moco_spline(folder_mat,nt,nz,verbose,index_b0 = [],graph=0):
    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    # append path that contains scripts, to be able to load modules
    sys.path.append(path_sct + '/scripts')
    import sct_utils as sct

    sct.printv('\n\n\n------------------------------------------------------------------------------',verbose)
    sct.printv('Spline Regularization along T: Smoothing Patient Motion...',verbose)
    
    fname_mat = [[[] for i in range(nz)] for i in range(nt)]
    for iT in range(nt):
        for iZ in range(nz):
            fname_mat[iT][iZ] = folder_mat + 'mat.T' + str(iT) + '_Z' + str(iZ) + '.txt'

    #Copying the existing Matrices to another folder
    old_mat = folder_mat + 'old/'
    if not os.path.exists(old_mat): os.makedirs(old_mat)    
    cmd = 'cp ' + folder_mat + '*.txt ' + old_mat
    status, output = sct.run(cmd, verbose)

    sct.printv('\nloading matrices...',verbose)
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
    sct.printv('\nGenerate motion splines...',verbose)
    T = np.arange(nt)
    if graph:
        import pylab as pl

    for iZ in range(nz):

#        frequency = scipy.fftpack.fftfreq(len(X[iZ][:]), d=1)
#        spectrum = np.abs(scipy.fftpack.fft(X[iZ][:], n=None, axis=-1, overwrite_x=False))
#        Wn = np.amax(frequency)/10
#        N = 5              #Order of the filter
#        b, a = scipy.signal.iirfilter(N, Wn, rp=None, rs=None, btype='low', analog=False, ftype='butter', output='ba')
#        X_smooth[iZ][:] = scipy.signal.filtfilt(b, a, X[iZ][:], axis=-1, padtype=None)

        spline = scipy.interpolate.UnivariateSpline(T, X[iZ][:], w=None, bbox=[None, None], k=3, s=None)
        X_smooth[iZ][:] = spline(T)
        
        if graph:
            pl.plot(T,X_smooth[iZ][:],label='spline_smoothing')
            pl.plot(T,X[iZ][:],marker='*',linestyle='None',label='original_val')
            if len(index_b0)!=0:
                T_b0 = [T[i_b0] for i_b0 in index_b0]
                X_b0 = [X[iZ][i_b0] for i_b0 in index_b0]
                pl.plot(T_b0,X_b0,marker='D',linestyle='None',color='k',label='b=0')
            pl.title('X')
            pl.grid()
            pl.legend()
            pl.show()

#        frequency = scipy.fftpack.fftfreq(len(Y[iZ][:]), d=1)
#        spectrum = np.abs(scipy.fftpack.fft(Y[iZ][:], n=None, axis=-1, overwrite_x=False))
#        Wn = np.amax(frequency)/10
#        N = 5              #Order of the filter
#        b, a = scipy.signal.iirfilter(N, Wn, rp=None, rs=None, btype='low', analog=False, ftype='butter', output='ba')
#        Y_smooth[iZ][:] = scipy.signal.filtfilt(b, a, Y[iZ][:], axis=-1, padtype=None)

        spline = scipy.interpolate.UnivariateSpline(T, Y[iZ][:], w=None, bbox=[None, None], k=3, s=None)
        Y_smooth[iZ][:] = spline(T)

        if graph:
            pl.plot(T,Y_smooth[iZ][:],label='spline_smoothing')
            pl.plot(T,Y[iZ][:],marker='*', linestyle='None',label='original_val')
            if len(index_b0)!=0:
                T_b0 = [T[i_b0] for i_b0 in index_b0]
                Y_b0 = [Y[iZ][i_b0] for i_b0 in index_b0]
                pl.plot(T_b0,Y_b0,marker='D',linestyle='None',color='k',label='b=0')
            pl.title('Y')
            pl.grid()
            pl.legend()
            pl.show()

    #Storing the final Matrices
    sct.printv('\nStoring the final Matrices...',verbose)
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

    sct.printv('\n...Done. Patient motion has been smoothed',verbose)
    sct.printv('------------------------------------------------------------------------------\n',verbose)

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
        '  -v           Set verbose=1 for printing text. Default value is 0\n' \
        '  -g           Set value to 1 for plotting graphs. Default value is 0\n' \
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