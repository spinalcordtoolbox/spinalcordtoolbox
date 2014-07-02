#!/usr/bin/env python

# check if needed Python libraries are already installed or not
import sys
import os
import commands
import getopt
import time

try:
    import nibabel
except ImportError:
    print '--- nibabel not installed! Exit program. ---'
    sys.exit(2)
try:
    import numpy as np
except ImportError:
    print '--- numpy not installed! Exit program. ---'
    sys.exit(2)

# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
import sct_utils as sct

fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI

class eddy_class:
    def __init__(self):

        #============================================
        #Different Parameters
        #============================================
        self.fname_data                = ''
        self.fname_bvecs               = ''
        self.mat_eddy                  = ''
        self.min_norm                  = 0.001
        self.swapXY                    = 0
        self.cost_function_flirt       = 'normcorr'               # 'mutualinfo' | 'woods' | 'corratio' | 'normcorr' | 'normmi' | 'leastsquares'. Default is 'normcorr'.
        self.interp                    = 'trilinear'              #  Default is 'trilinear'. Additional options: trilinear,nearestneighbour,sinc,spline
        self.delete_tmp_files          = 1
        self.merge_back                = 1
        self.verbose                   = 0
#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    start_time = time.time()
    param = eddy_class()

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:b:m:c:p:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            param.fname_data = arg
        elif opt in ('-b'):
            param.fname_bvecs = arg
        elif opt in ('-m'):
            param.mat_eddy = arg
        elif opt in ('-c'):
            param.cost_function_flirt = arg
        elif opt in ('-p'):
            param.interp = arg
        elif opt in ('-v'):
            param.verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if param.fname_data=='' or param.fname_bvecs=='':
        print '\n\nAll mandatory arguments are not provided \n'
        usage()

    #Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(param.fname_data)

    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+ path_tmp)

    # go to tmp folder
    os.chdir(path_tmp)

    # run sct_eddy_correct
    sct_eddy_correct(param)

    # come back to parent folder
    os.chdir('..')

    # Delete temporary files
    if param.delete_tmp_files == 1:
        print '\nDelete temporary files...'
        sct.run('rm -rf '+ path_tmp)

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'

#=======================================================================================================================
# Function sct_eddy_correct
#=======================================================================================================================
def sct_eddy_correct(param):

    print '\n\n\n\n==================================================='
    print '              Running: sct_eddy_correct'
    print '===================================================\n'

    fname_data    = param.fname_data
    min_norm      = param.min_norm
    cost_function = param.cost_function_flirt
    
    print 'Input File:',param.fname_data
    print 'Bvecs File:',param.fname_bvecs    
    
    #Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)
    
    if param.mat_eddy=='': param.mat_eddy= path_data + 'mat_eddy/'
    if not os.path.exists(param.mat_eddy): os.makedirs(param.mat_eddy)
    mat_eddy    = param.mat_eddy
    
    #Schedule file for FLIRT
    schedule_file = path_sct + '/flirtsch/schedule_TxTy_2mmScale.sch'
    print '\n.. Schedule file: ',schedule_file

    #Swap X-Y dimension (to have X as phase-encoding direction)
    if param.swapXY==1:
        print '\nSwap X-Y dimension (to have X as phase-encoding direction)'
        fname_data_new = 'tmp.data_swap'
        cmd = fsloutput + 'fslswapdim ' + fname_data + ' -y -x -z ' + fname_data_new
        status, output = sct.run(cmd)
        print '\n.. updated data file name: ',fname_data_new
    else:
        fname_data_new = fname_data

    # Get size of data
    print '\nGet dimensions data...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_data)
    print '.. '+str(nx)+' x '+str(ny)+' x '+str(nz)+' x '+str(nt)

    # split along T dimension
    print '\nSplit along T dimension...'
    cmd = fsloutput + 'fslsplit ' + fname_data_new + ' ' + 'tmp.data_splitT'
    status, output = sct.run(cmd)
    numT = []
    for i in range(nt):
        if len(str(i))==1:
            numT.append('000' + str(i))
        elif len(str(i))==2:
            numT.append('00' + str(i))
        elif len(str(i))==3:
            numT.append('0' + str(i))
        else:
            numT.append(str(nt))

    nb_loops = nz
    file_suffix=[]
    for iZ in range(nz):
        file_suffix.append('_Z'+numT[iZ])

    # Identify pairs of opposite gradient directions
    print '\nIdentify pairs of opposite gradient directions...'

    # Open bvecs file
    print '\nOpen bvecs file...'
    bvecs = []
    with open(param.fname_bvecs) as f:
        for line in f:
            bvecs_new = map(float, line.split())
            bvecs.append(bvecs_new)

    # Check if bvecs file is nx3
    if not len(bvecs[0][:]) == 3:
        print '.. WARNING: bvecs file is 3xn instead of nx3. Consider using sct_dmri_transpose_bvecs.'
        print 'Transpose bvecs...'
        # transpose bvecs
        bvecs = zip(*bvecs)
    bvecs = np.array(bvecs)

    opposite_gradients_iT = []
    opposite_gradients_jT = []
    index_identified = []
    index_b0 = []
    for iT in range(nt-1):
        if np.linalg.norm(bvecs[iT,:])!=0:
            if iT not in index_identified:
                jT = iT+1
                if np.linalg.norm((bvecs[iT,:]+bvecs[jT,:]))<min_norm:
                    print '.. Opposite gradient for #',str(iT),' is: #',str(jT)
                    opposite_gradients_iT.append(iT)
                    opposite_gradients_jT.append(jT)
                    index_identified.append(iT)
        else:
            index_b0.append(iT)
            print '.. Opposite gradient for #',str(iT),' is: NONE (b=0)'
    nb_oppositeGradients = len(opposite_gradients_iT)
    print '.. Number of gradient directions: ',str(2*nb_oppositeGradients), ' (2*',str(nb_oppositeGradients),')'
    print '.. Index b=0: ',str(index_b0)


    # =========================================================================
    #	Find transformation
    # =========================================================================
    for iN in range(nb_oppositeGradients):
        i_plus = opposite_gradients_iT[iN]
        i_minus = opposite_gradients_jT[iN]

        print '\nFinding affine transformation between volumes #',str(i_plus),' and #',str(i_minus),' (',str(iN),'/',str(nb_oppositeGradients),')'
        print '------------------------------------------------------------------------------------\n'
        
        #Slicewise correction
        print '\nSplit volumes across Z...'
        fname_plus = 'tmp.data_splitT' + numT[i_plus]
        fname_plus_Z = 'tmp.data_splitT' + numT[i_plus] + '_Z'
        cmd = fsloutput + 'fslsplit ' + fname_plus + ' ' + fname_plus_Z + ' -z'
        status, output = sct.run(cmd)

        fname_minus = 'tmp.data_splitT' + numT[i_minus]
        fname_minus_Z = 'tmp.data_splitT' + numT[i_minus] + '_Z'
        cmd = fsloutput + 'fslsplit ' + fname_minus + ' ' + fname_minus_Z + ' -z'
        status, output = sct.run(cmd)

        #loop across Z
        for iZ in range(nb_loops):
            fname_plus = 'tmp.data_splitT' + numT[i_plus] + file_suffix[iZ]

            fname_minus = 'tmp.data_splitT' + numT[i_minus] + file_suffix[iZ]
            #Find transformation on opposite gradient directions
            print '\nFind transformation for each pair of opposite gradient directions...'
            fname_plus_corr = 'tmp.data_splitT' + numT[i_plus] + file_suffix[iZ] + '_corr_'
            omat = 'mat__tmp.data_splitT' + numT[i_plus] + file_suffix[iZ] + '.txt'
            cmd = fsloutput+'flirt -in '+fname_plus+' -ref '+fname_minus+' -paddingsize 3 -schedule '+schedule_file+' -verbose 2 -omat '+omat+' -cost '+cost_function+' -forcescaling'
            status, output = sct.run(cmd)

            file =  open(omat)
            Matrix = np.loadtxt(file)
            file.close()
            M = Matrix[0:4,0:4]
            print '.. Transformation matrix:\n',M
            print '.. Output matrix file: ',omat

            # Divide affine transformation by two
            print '\nDivide affine transformation by two...'
            A = (M - np.identity(4))/2
            Mplus = np.identity(4)+A
            omat_plus = mat_eddy + 'mat.T' + str(i_plus) + '_Z' + str(iZ) + '.txt'
            file =  open(omat_plus,'w')
            np.savetxt(omat_plus, Mplus, fmt='%.6e', delimiter='  ', newline='\n', header='', footer='', comments='#')
            file.close()
            print '.. Output matrix file (plus): ',omat_plus

            Mminus = np.identity(4)-A
            omat_minus = mat_eddy + 'mat.T' + str(i_minus) + '_Z' + str(iZ) + '.txt'
            file =  open(omat_minus,'w')
            np.savetxt(omat_minus, Mminus, fmt='%.6e', delimiter='  ', newline='\n', header='', footer='', comments='#')
            file.close()
            print '.. Output matrix file (minus): ',omat_minus

    # =========================================================================
    #	Apply affine transformation
    # =========================================================================

    print '\nApply affine transformation matrix'
    print '------------------------------------------------------------------------------------\n'

    for iN in range(nb_oppositeGradients):
        for iFile in range(2):
            if iFile==0:
                i_file = opposite_gradients_iT[iN]
            else:
                i_file = opposite_gradients_jT[iN]

            for iZ in range(nb_loops):
                fname = 'tmp.data_splitT' + numT[i_file] + file_suffix[iZ]
                fname_corr = fname + '_corr_' + '__div2'
                omat = mat_eddy + 'mat.T' + str(i_file) + '_Z' + str(iZ) + '.txt'
                cmd = fsloutput + 'flirt -in ' + fname + ' -ref ' + fname + ' -out ' + fname_corr + ' -init ' + omat + ' -applyxfm -paddingsize 3 -interp ' + param.interp
                status, output = sct.run(cmd)

    
    # =========================================================================
    #	Merge back across Z
    # =========================================================================

    print '\nMerge across Z'
    print '------------------------------------------------------------------------------------\n'

    for iN in range(nb_oppositeGradients):
        i_plus = opposite_gradients_iT[iN]
        fname_plus_corr = 'tmp.data_splitT' + numT[i_plus] + '_corr_' + '__div2'
        cmd = fsloutput + 'fslmerge -z ' + fname_plus_corr

        for iZ in range(nz):
            fname_plus_Z_corr = 'tmp.data_splitT' + numT[i_plus] + file_suffix[iZ] + '_corr_' + '__div2'
            cmd = cmd + ' ' + fname_plus_Z_corr
        status, output = sct.run(cmd)

        i_minus = opposite_gradients_jT[iN]
        fname_minus_corr = 'tmp.data_splitT' + numT[i_minus] + '_corr_' + '__div2'
        cmd = fsloutput + 'fslmerge -z ' + fname_minus_corr

        for iZ in range(nz):
            fname_minus_Z_corr = 'tmp.data_splitT' + numT[i_minus] + file_suffix[iZ] + '_corr_' + '__div2'
            cmd = cmd + ' ' + fname_minus_Z_corr
        status, output = sct.run(cmd)

    # =========================================================================
    #	Merge files back
    # =========================================================================
    print '\nMerge back across T...'
    print '------------------------------------------------------------------------------------\n'
    
    fname_data_corr = path_data + file_data + '_eddy'
    cmd = fsloutput + 'fslmerge -t ' + fname_data_corr
    path_tmp = os.getcwd()
    for iT in range(nt):
        if os.path.isfile((path_tmp + '/' + 'tmp.data_splitT' + numT[iT] + '_corr_' + '__div2.nii')):
            fname_data_corr_3d = 'tmp.data_splitT' + numT[iT] + '_corr_' + '__div2'
        elif iT in index_b0:
            fname_data_corr_3d = 'tmp.data_splitT' + numT[iT]
        
        cmd = cmd + ' ' + fname_data_corr_3d
    status, output = sct.run(cmd)

    #Swap back X-Y dimensions
    if param.swapXY==1:
        fname_data_final = fname_data
        print '\nSwap back X-Y dimensions'
        cmd = fsloutput_temp + 'fslswapdim ' + fname_data_corr + ' -y -x -z ' + fname_data_final
        status, output = sct.run(cmd)
    else:
        fname_data_final = fname_data_corr

    print '... File created: ',fname_data_final
    
    print '\n==================================================='
    print '              Completed: sct_eddy_correct'
    print '===================================================\n\n\n'

#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print '\n' \
        ' '+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        'Correct Eddy-current distortions using pairs of DW images acquired at reversed gradient polarities' \
        '\nUSAGE: \n' \
        '  '+os.path.basename(__file__)+' -i <filename> -b <bvecs_file>\n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        '  -i           input_file \n' \
        '  -b           bvecs file \n' \
        '\n'\
        'OPTIONAL ARGUMENTS\n' \
        '  -m           matrix folder \n' \
        '  -c           Cost function FLIRT - mutualinfo | woods | corratio | normcorr | normmi | leastsquares. Default is <normcorr>..\n' \
        '  -p           Interpolation - Default is trilinear. Additional options: nearestneighbour,sinc,spline.\n' \
        '  -h           help. Show this message.\n' \
        '\n'\
        'EXAMPLE:\n' \
        '  '+os.path.basename(__file__)+' -i KS_HCP34.nii -b KS_HCP_bvec.txt \n'
    
    #Exit Program
    sys.exit(2)

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()