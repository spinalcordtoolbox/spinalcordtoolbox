#!/usr/bin/env python
#########################################################################################
#
# Eddy Current Correction.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Tanguy Duval, Julien Cohen-Adad
# Modified: 2014-07-02
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: add test.
# TODO: remove FSL dependency

# check if needed Python libraries are already installed or not
import sys
import os
import commands
import getopt
import time
import numpy as np
from msct_image import Image
import sct_utils as sct
# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')


fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '  # for faster processing, all outputs are in NIFTI


class Param:
    def __init__(self):
        self.fname_data                = ''
        self.fname_bvecs               = ''
        self.slicewise                 = 1
        self.output_path               = ''
        self.mat_eddy                  = ''
        self.min_norm                  = 0.001
        self.swapXY                    = 0
        self.cost_function_flirt       = 'normcorr'               # 'mutualinfo' | 'woods' | 'corratio' | 'normcorr' | 'normmi' | 'leastsquares'. Default is 'normcorr'.
        self.interp                    = 'trilinear'              #  Default is 'trilinear'. Additional options: trilinear,nearestneighbour,sinc,spline
        self.delete_tmp_files          = 1
        self.merge_back                = 1
        self.verbose                   = 0
        self.plot_graph                = 0


#=======================================================================================================================
# main
#=======================================================================================================================
def main():
    start_time = time.time()

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:c:b:g:m:o:p:r:s:v:')
    except getopt.GetoptError:
        usage()
    if not opts:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            param.fname_data = arg
        elif opt in ('-b'):
            param.fname_bvecs = arg
        elif opt in ('-c'):
            param.cost_function_flirt = arg
        elif opt in ('-g'):
            param.plot_graph = int(arg)
        elif opt in ('-m'):
            param.mat_eddy = arg
        elif opt in ('-o'):
            param.output_path = arg
        elif opt in ('-p'):
            param.interp = arg
        elif opt in ('-r'):
            param.delete_tmp_files = int(arg)
        elif opt in ('-s'):
            param.slicewise = int(arg)
        elif opt in ('-v'):
            param.verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if param.fname_data == '' or param.fname_bvecs == '':
        sct.printv('\n\nAll mandatory arguments are not provided \n')
        usage()

    if param.output_path == '':
        param.output_path = os.getcwd() + '/'

    # create temporary folder
    path_tmp = 'tmp.' + time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir ' + path_tmp, param.verbose)

    # go to tmp folder
    os.chdir(path_tmp)

    # run sct_eddy_correct
    eddy_correct(param)

    # come back to parent folder
    os.chdir('..')

    # Delete temporary files
    if param.delete_tmp_files == 1:
        sct.printv('\nDelete temporary files...')
        sct.run('rm -rf ' + path_tmp, param.verbose)

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('\nFinished! Elapsed time: ' + str(int(round(elapsed_time))) + 's')

#=======================================================================================================================
# Function eddy_correct
#=======================================================================================================================


def eddy_correct(param):

    sct.printv('\n\n\n\n===================================================', param.verbose)
    sct.printv('              Running: eddy_correct', param.verbose)
    sct.printv('===================================================\n', param.verbose)

    fname_data    = param.fname_data
    min_norm      = param.min_norm
    cost_function = param.cost_function_flirt
    verbose       = param.verbose

    sct.printv(('Input File:' + param.fname_data), verbose)
    sct.printv(('Bvecs File:' + param.fname_bvecs), verbose)

    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)

    if param.mat_eddy == '':
        param.mat_eddy = 'mat_eddy/'
    if not os.path.exists(param.mat_eddy):
        os.makedirs(param.mat_eddy)
    mat_eddy    = param.mat_eddy

    # Schedule file for FLIRT
    schedule_file = path_sct + '/flirtsch/schedule_TxTy_2mmScale.sch'
    sct.printv(('\n.. Schedule file: ' + schedule_file), verbose)

    # Swap X-Y dimension (to have X as phase-encoding direction)
    if param.swapXY == 1:
        sct.printv('\nSwap X-Y dimension (to have X as phase-encoding direction)', verbose)
        fname_data_new = 'tmp.data_swap'
        cmd = fsloutput + 'fslswapdim ' + fname_data + ' -y -x -z ' + fname_data_new
        status, output = sct.run(cmd, verbose)
        sct.printv(('\n.. updated data file name: ' + fname_data_new), verbose)
    else:
        fname_data_new = fname_data

    # Get size of data
    sct.printv('\nGet dimensions data...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_data).dim
    sct.printv('.. ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz) + ' x ' + str(nt), verbose)

    # split along T dimension
    sct.printv('\nSplit along T dimension...', verbose)
    from sct_image import split_data
    im_to_split = Image(fname_data_new + '.nii')
    im_split_list = split_data(im_to_split, 3)
    for im in im_split_list:
        im.save()

    # cmd = fsloutput + 'fslsplit ' + fname_data_new + ' ' + file_data + '_T'
    # status, output = sct.run(cmd,verbose)

    # Slice-wise or Volume based method
    if param.slicewise:
        nb_loops = nz
        file_suffix = []
        for iZ in range(nz):
            file_suffix.append('_Z' + str(iZ).zfill(4))
    else:
        nb_loops = 1
        file_suffix = ['']

    # Identify pairs of opposite gradient directions
    sct.printv('\nIdentify pairs of opposite gradient directions...', verbose)

    # Open bvecs file
    sct.printv('\nOpen bvecs file...', verbose)
    bvecs = []
    with open(param.fname_bvecs) as f:
        for line in f:
            bvecs_new = map(float, line.split())
            bvecs.append(bvecs_new)

    # Check if bvecs file is nx3
    if not len(bvecs[0][:]) == 3:
        sct.printv('.. WARNING: bvecs file is 3xn instead of nx3. Consider using sct_dmri_transpose_bvecs.', verbose)
        sct.printv('Transpose bvecs...', verbose)
        # transpose bvecs
        bvecs = zip(*bvecs)
    bvecs = np.array(bvecs)

    opposite_gradients_iT = []
    opposite_gradients_jT = []
    index_identified = []
    index_b0 = []
    for iT in range(nt - 1):
        if np.linalg.norm(bvecs[iT, :]) != 0:
            if iT not in index_identified:
                jT = iT + 1
                if np.linalg.norm((bvecs[iT, :] + bvecs[jT, :])) < min_norm:
                    sct.printv(('.. Opposite gradient for #' + str(iT) + ' is: #' + str(jT)), verbose)
                    opposite_gradients_iT.append(iT)
                    opposite_gradients_jT.append(jT)
                    index_identified.append(iT)
        else:
            index_b0.append(iT)
            sct.printv(('.. Opposite gradient for #' + str(iT) + ' is: NONE (b=0)'), verbose)
    nb_oppositeGradients = len(opposite_gradients_iT)
    sct.printv(('.. Number of gradient directions: ' + str(2 * nb_oppositeGradients) + ' (2*' + str(nb_oppositeGradients) + ')'), verbose)
    sct.printv('.. Index b=0: ' + str(index_b0), verbose)

    # =========================================================================
    #	Find transformation
    # =========================================================================
    for iN in range(nb_oppositeGradients):
        i_plus = opposite_gradients_iT[iN]
        i_minus = opposite_gradients_jT[iN]

        sct.printv(('\nFinding affine transformation between volumes #' + str(i_plus) + ' and #' + str(i_minus) + ' (' + str(iN) + '/' + str(nb_oppositeGradients) + ')'), verbose)
        sct.printv('------------------------------------------------------------------------------------\n', verbose)

        # Slicewise correction
        if param.slicewise:
            sct.printv('\nSplit volumes across Z...', verbose)
            fname_plus = file_data + '_T' + str(i_plus).zfill(4)
            fname_plus_Z = file_data + '_T' + str(i_plus).zfill(4) + '_Z'
            im_plus = Image(fname_plus + '.nii')
            im_plus_split_list = split_data(im_plus, 2)
            for im_p in im_plus_split_list:
                im_p.save()
            # cmd = fsloutput + 'fslsplit ' + fname_plus + ' ' + fname_plus_Z + ' -z'
            # status, output = sct.run(cmd,verbose)

            fname_minus = file_data + '_T' + str(i_minus).zfill(4)
            fname_minus_Z = file_data + '_T' + str(i_minus).zfill(4) + '_Z'
            im_minus = Image(fname_minus + '.nii')
            im_minus_split_list = split_data(im_minus, 2)
            for im_m in im_minus_split_list:
                im_m.save()            # cmd = fsloutput + 'fslsplit ' + fname_minus + ' ' + fname_minus_Z + ' -z'
            # status, output = sct.run(cmd,verbose)

        # loop across Z
        for iZ in range(nb_loops):
            fname_plus = file_data + '_T' + str(i_plus).zfill(4) + file_suffix[iZ]

            fname_minus = file_data + '_T' + str(i_minus).zfill(4) + file_suffix[iZ]
            # Find transformation on opposite gradient directions
            sct.printv('\nFind transformation for each pair of opposite gradient directions...', verbose)
            fname_plus_corr = file_data + '_T' + str(i_plus).zfill(4) + file_suffix[iZ] + '_corr_'
            omat = 'mat_' + file_data + '_T' + str(i_plus).zfill(4) + file_suffix[iZ] + '.txt'
            cmd = fsloutput + 'flirt -in ' + fname_plus + ' -ref ' + fname_minus + ' -paddingsize 3 -schedule ' + schedule_file + ' -verbose 2 -omat ' + omat + ' -cost ' + cost_function + ' -forcescaling'
            status, output = sct.run(cmd, verbose)

            file =  open(omat)
            Matrix = np.loadtxt(file)
            file.close()
            M = Matrix[0:4, 0:4]
            sct.printv(('.. Transformation matrix:\n' + str(M)), verbose)
            sct.printv(('.. Output matrix file: ' + omat), verbose)

            # Divide affine transformation by two
            sct.printv('\nDivide affine transformation by two...', verbose)
            A = (M - np.identity(4)) / 2
            Mplus = np.identity(4) + A
            omat_plus = mat_eddy + 'mat.T' + str(i_plus) + '_Z' + str(iZ) + '.txt'
            file =  open(omat_plus, 'w')
            np.savetxt(omat_plus, Mplus, fmt='%.6e', delimiter='  ', newline='\n', header='', footer='', comments='#')
            file.close()
            sct.printv(('.. Output matrix file (plus): ' + omat_plus), verbose)

            Mminus = np.identity(4) - A
            omat_minus = mat_eddy + 'mat.T' + str(i_minus) + '_Z' + str(iZ) + '.txt'
            file =  open(omat_minus, 'w')
            np.savetxt(omat_minus, Mminus, fmt='%.6e', delimiter='  ', newline='\n', header='', footer='', comments='#')
            file.close()
            sct.printv(('.. Output matrix file (minus): ' + omat_minus), verbose)

    # =========================================================================
    #	Apply affine transformation
    # =========================================================================

    sct.printv('\nApply affine transformation matrix', verbose)
    sct.printv('------------------------------------------------------------------------------------\n', verbose)

    for iN in range(nb_oppositeGradients):
        for iFile in range(2):
            if iFile == 0:
                i_file = opposite_gradients_iT[iN]
            else:
                i_file = opposite_gradients_jT[iN]

            for iZ in range(nb_loops):
                fname = file_data + '_T' + str(i_file).zfill(4) + file_suffix[iZ]
                fname_corr = fname + '_corr_' + '__div2'
                omat = mat_eddy + 'mat.T' + str(i_file) + '_Z' + str(iZ) + '.txt'
                cmd = fsloutput + 'flirt -in ' + fname + ' -ref ' + fname + ' -out ' + fname_corr + ' -init ' + omat + ' -applyxfm -paddingsize 3 -interp ' + param.interp
                status, output = sct.run(cmd, verbose)

    # =========================================================================
    #	Merge back across Z
    # =========================================================================

    sct.printv('\nMerge across Z', verbose)
    sct.printv('------------------------------------------------------------------------------------\n', verbose)

    for iN in range(nb_oppositeGradients):
        i_plus = opposite_gradients_iT[iN]
        fname_plus_corr = file_data + '_T' + str(i_plus).zfill(4) + '_corr_' + '__div2'
        cmd = fsloutput + 'fslmerge -z ' + fname_plus_corr

        for iZ in range(nz):
            fname_plus_Z_corr = file_data + '_T' + str(i_plus).zfill(4) + file_suffix[iZ] + '_corr_' + '__div2'
            cmd = cmd + ' ' + fname_plus_Z_corr
        status, output = sct.run(cmd, verbose)

        i_minus = opposite_gradients_jT[iN]
        fname_minus_corr = file_data + '_T' + str(i_minus).zfill(4) + '_corr_' + '__div2'
        cmd = fsloutput + 'fslmerge -z ' + fname_minus_corr

        for iZ in range(nz):
            fname_minus_Z_corr = file_data + '_T' + str(i_minus).zfill(4) + file_suffix[iZ] + '_corr_' + '__div2'
            cmd = cmd + ' ' + fname_minus_Z_corr
        status, output = sct.run(cmd, verbose)

    # =========================================================================
    #	Merge files back
    # =========================================================================
    sct.printv('\nMerge back across T...', verbose)
    sct.printv('------------------------------------------------------------------------------------\n', verbose)

    fname_data_corr = param.output_path + file_data + '_eddy'
    cmd = fsloutput + 'fslmerge -t ' + fname_data_corr
    path_tmp = os.getcwd()
    for iT in range(nt):
        if os.path.isfile((path_tmp + '/' + file_data + '_T' + str(iT).zfill(4) + '_corr_' + '__div2.nii')):
            fname_data_corr_3d = file_data + '_T' + str(iT).zfill(4) + '_corr_' + '__div2'
        elif iT in index_b0:
            fname_data_corr_3d = file_data + '_T' + str(iT).zfill(4)

        cmd = cmd + ' ' + fname_data_corr_3d
    status, output = sct.run(cmd, verbose)

    # Swap back X-Y dimensions
    if param.swapXY == 1:
        fname_data_final = fname_data
        sct.printv('\nSwap back X-Y dimensions', verbose)
        cmd = fsloutput + 'fslswapdim ' + fname_data_corr + ' -y -x -z ' + fname_data_final
        status, output = sct.run(cmd, verbose)
    else:
        fname_data_final = fname_data_corr

    sct.printv(('... File created: ' + fname_data_final), verbose)

    sct.printv('\n===================================================', verbose)
    sct.printv('              Completed: eddy_correct', verbose)
    sct.printv('===================================================\n\n\n', verbose)


#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print('\n'
        ' ' + os.path.basename(__file__) + '\n'
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n'
        '\n'
        'DESCRIPTION\n'
        'Correct Eddy-current distortions using pairs of DW images acquired at reversed gradient polarities'
        '\nUSAGE: \n'
        '  ' + os.path.basename(__file__) + ' -i <filename> -b <bvecs_file>\n'
        '\n'
        'MANDATORY ARGUMENTS\n'
        '  -i           input_file \n'
        '  -b           bvecs file \n'
        '\n'
        'OPTIONAL ARGUMENTS\n'
        '  -o           Specify Output path.\n'
        '  -s           Set value to 0 for volume based correction. Default value is 1 i.e slicewise correction\n'
        '  -m           matrix folder \n'
        '  -c           Cost function FLIRT - mutualinfo | woods | corratio | normcorr | normmi | leastsquares. Default is <normcorr>..\n'
        '  -p           Interpolation - Default is trilinear. Additional options: nearestneighbour,sinc,spline.\n'
        '  -g {0,1}     Set value to 1 for plotting graphs. Default value is 0 \n'
        '  -r           Set value to 0 for not deleting temp files. Default value is 1 \n'
        '  -v {0,1}     Set verbose=1 for sct.printv(ng text. Default value is 0 \n'
        '  -h           help. Show this message.\n'
        '\n'
        'EXAMPLE:\n'
        '  ' + os.path.basename(__file__) + ' -i KS_HCP34.nii -b KS_HCP_bvec.txt \n')

    # Exit Program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    sct.start_stream_logger()
    param = Param()
    # call main function
    main()
