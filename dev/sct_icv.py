#!/usr/bin/env python
#########################################################################################
# Extract raw intracranial volume in cubic millimeters.
# Method: bias field correction (N4) followed by SIENAx or RBM
# RBM (reverse MNI brain masking) is adapted from Keihaninejad (2010)
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: William Thong
# Modified: 2014-09-01
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os, sys, getopt, re


# Default parameters
# ==========================================================================================
class Param:
    def __init__(self):
        self.N4Correct = True  #Need to be True for correct -f value in BET
        self.ImageDimension = 3
        
        # MNI152 templates (T1 2mm)
        self.path_atlas='${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz'
        self.path_mask_wm='${FSLDIR}/data/standard/tissuepriors/avg152T1_white.img'
        self.path_mask_gm='${FSLDIR}/data/standard/tissuepriors/avg152T1_gray.img'        
        
        # default parameters of the main
        self.contrast = 't1'
        self.method = 'sienax'
        self.verbose = 1
        self.debugging=False
        
def main():
    
    # Initialization
    input_path = ''
    contrast = param.contrast
    output_path = ''
    method = param.method
    verbose = param.verbose

    # Check input parameters
    if param.debugging:
        pass
    else:
        try:
            opts, args = getopt.getopt(sys.argv[1:],'hi:c:o:d:v:')
        except getopt.GetoptError:
            usage()
        if not opts:
            usage()
        if not opts:
            # no option supplied
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ("-i"):
                input_path = arg
                exist_image(input_path)
            elif opt in ("-o"):
                output_path = arg
            elif opt in ("-c"):
                contrast = arg
            elif opt in ("-d"):
                method = arg
            elif opt in ("-v"):
                verbose = int(arg)


    # check mandatory arguments
    if input_path == '' :
        print('\nError: Wrong input path, "'+input_path+'" not valid')
        usage()
        
    if contrast != 't1' and contrast != 't2':
        print('\nError: Wrong contrast, "'+contrast+'" not valid')
        usage()
    
    if method != 'sienax' and method != 'rbm':
        print('\nError: Wrong method, "'+method+'" not valid')
        usage()

    # Normalize paths (remove trailing slash)
    input_path = os.path.normpath(input_path)

    # Extract path, file and extension
    path_input, file_input, ext_input = extract_fname(input_path)

    # define output_path
    if output_path=='':
        output_path=path_input
    else:
        output_path = os.path.normpath(output_path)+'/'
    
    #check existence of directories (if exists, removes subdirectories; if not, creates directory)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # print arguments
    if verbose:
        print 'Check input parameters...'
        print '.. Image:    '+input_path
        print '.. Contrast:    '+contrast
        print '.. Output directory:     '+output_path
        print '.. Method:    '+method
    

    ####################################
    # The core of the code starts here #
    ####################################

    #-----------------------------------
    # Field correction with N4
    # Note that N3 is implemented in BET, use -B parameter
    #-----------------------------------
    if param.N4Correct:
        cmd='N4BiasFieldCorrection -d '+str(param.ImageDimension)+' -i '+input_path+' -o '+output_path+file_input+'_n4'+ext_input
        print(">> "+cmd)
        os.system(cmd)
        file_input = file_input+'_n4'
    
    if method=='sienax':
        #-----------------------------------    
        # Parameters for bet and use SIENAx
        #-----------------------------------    
        # -f Fractional intensity threshold (smaller values give larger brain) equals 0.2 for both contrasts
        # -R Robust brain centre estimation (iterates BET several times)
        if contrast == 't2':
            frac_int = 0.2
            # note that -t2 parameter is important in SIENAx to estimate white matter volume properly
            cmd='sienax '+output_path+file_input+ext_input+' -o '+output_path+file_input+'_sienax -B "-f '+str(frac_int)+' -R" -t2' 
            print(">> "+cmd)
            os.system(cmd)
        elif contrast == 't1':
            frac_int = 0.2
            cmd='sienax '+output_path+file_input+ext_input+' -o '+output_path+file_input+'_sienax -B "-f '+str(frac_int)+' -R"'
            print(">> "+cmd)
            os.system(cmd)    
        
        #-----------------------------------    
        # Read SIENAx report to extract raw brain ICV in cubic millimeters and write final value in icv.txt
        #-----------------------------------   
        report = parse_report(output_path+file_input+'_sienax/report.sienax')
        
        print('Writting icv value in "'+output_path+'icv.txt"...')
        fo = open(output_path+"icv.txt", "wb")
        fo.write(str(report['brain']['raw']))
        fo.close()
        
        print('\n\n\nEstimated brain volume: '+str(report['brain']['raw'])+' mm3\n\n\n')    
        
        print('\n\n\nDone!')
    
    if method=='rbm':
        # Brain extraction
        frac_int = 0.2
        file_output='tmp.brain.'+file_input
        cmd = 'bet '+output_path+file_input+ext_input+' '+output_path+file_output+ext_input+' -R -f '+str(frac_int)
        print(">> "+cmd)
        os.system(cmd)
           
        # Swap dimensions to correspond to the template
        swapping = get_orient(output_path+file_output+ext_input)
        file_output='tmp.brain.reorient.'+file_input
        cmd = 'fslswapdim '+output_path+'tmp.brain.'+file_input+ext_input+' '+swapping+' '+output_path+file_output+ext_input
        print(">> "+cmd)
        os.system(cmd)
        
        # Resample in 2mm to match template voxel size 
        cmd = 'isct_c3d '+output_path+file_output+ext_input+' -resample-mm 2.0x2.0x2.0mm '+output_path+file_output+'_2mm'+ext_input
        print(">> "+cmd)
        os.system(cmd)
        file_output=file_output+'_2mm'
        source_img=file_output
        
        # Registration on template        
        cmd = 'flirt -dof 6 -ref '+param.path_atlas+' -in '+output_path+file_output+ext_input+' -out '+output_path+file_output+'_reg'+ext_input+' -omat '+output_path+'tmp.'+file_input+'_affine_transf.mat'
        print(">> "+cmd)
        os.system(cmd)
        file_output = file_output+'_reg'
        
        # Histogram Normalization of gm template (note: 245 is the max intensity in the MNI152 gm template)
        cmd = 'fslmaths '+param.path_mask_gm+' -div 245 '+output_path+'tmp.brain_gm'+ext_input
        print(">> "+cmd)
        os.system(cmd)

        # Histogram Normalization of wm template (note: 253 is the max intensity in the MNI152 wm template)
        cmd = 'fslmaths '+param.path_mask_wm+' -div 253 '+output_path+'tmp.brain_wm'+ext_input
        print(">> "+cmd)
        os.system(cmd)        
        
        # Apply binary mask for gm
        cmd = 'fslmaths '+output_path+file_output+ext_input+' -mul '+output_path+'tmp.brain_gm'+ext_input+' '+output_path+file_output+'_gm'+ext_input
        print(">> "+cmd)
        os.system(cmd)
        
        # Apply binary mask for wm
        cmd = 'fslmaths '+output_path+file_output+ext_input+' -mul '+output_path+'tmp.brain_wm'+ext_input+' '+output_path+file_output+'_wm'+ext_input
        print(">> "+cmd)
        os.system(cmd)
        
        # Merge gm and wm
        cmd = 'fslmaths '+output_path+file_output+'_gm'+ext_input+' -add '+output_path+file_output+'_wm'+ext_input+' '+output_path+file_output+'_masked'+ext_input
        print(">> "+cmd)
        os.system(cmd)
        
        # Threshold the previous merge by the mean intensity value of non zeros voxels
        p = os.popen('fslstats '+output_path+file_output+'_masked'+ext_input+' -M')
        s = p.readline()
        p.close()
        cmd = 'fslmaths '+output_path+file_output+'_masked'+ext_input+' -thr '+str(float(s)*6./5)+' '+output_path+file_output+'_masked_thr'+ext_input
        print(">> "+cmd)
        os.system(cmd)        
        file_output = file_output+'_masked_thr'        
              
        # invert transformation matrix
        cmd = 'convert_xfm -omat '+output_path+'tmp.'+file_input+'_affine_inverse_transf.mat -inverse '+output_path+'tmp.'+file_input+'_affine_transf.mat'
        print(">> "+cmd)
        os.system(cmd)      
        
        # apply inverse transmation matrix 
        cmd = 'flirt -ref '+output_path+source_img+ext_input+' -in '+output_path+file_output+ext_input+' -out '+output_path+file_output+'_inv'+ext_input+' -init '+output_path+'tmp.'+file_input+'_affine_inverse_transf.mat -applyxfm'
        print(">> "+cmd)
        os.system(cmd)
        file_output = file_output+'_inv'
        
        # reslice in initial space
        cmd = 'isct_c3d '+output_path+file_input+ext_input+' '+output_path+file_output+ext_input+' -reslice-identity -o '+output_path+file_input+'_brain'+ext_input
        print(">> "+cmd)
        os.system(cmd)

        # write final icv in icv.txt
        print('Writting icv value in "'+output_path+'icv.txt"...')
        p = os.popen('fslstats '+output_path+file_input+'_brain'+ext_input+' -V')
        s = p.readline()
        p.close()
        
        fo = open(output_path+"icv.txt", "wb")
        fo.write(s.split(' ',1)[0])
        fo.close()
        
        print('\n\n\nEstimated brain volume: '+s.split(' ',1)[0]+' mm3\n\n\n')
        
        # remove all the tempopary files created
        print('\nDelete temporary files...')
        cmd = 'rm '+output_path+'tmp.*'
        print(">> "+cmd)
        os.system(cmd)
        
        print('\n\n\nDone!')

        
def usage():
    """ Print usage. """
    path_func, file_func, ext_func = extract_fname(sys.argv[0])
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  This program extracts raw intracranial volume in cubic millimeters.\n' \
        '  Method: bias field correction (N4) followed by SIENAx or RBM. RBM (reverse MNI brain masking) is adapted from Keihaninejad (2010).' \
        '\n' \
        'USAGE\n' \
        '   '+file_func+ext_func+' -i <inputvol> [options]\n\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -i inputvol          image to extract values from\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -c contrast          image contrast. t1 or t2 (default=t1), e.g. -c t1\n' \
        '  -o output            set output directory (create directory if not exists) \n' \
        '  -d method            method to estimate ICV. sienax or rbm (default=sienax), e.g. -d rbm \n' \
        '  -v verbose           verbose. 0 or 1. (default=1).\n' \
        '\n' \

    sys.exit(2)


def exist_image(fname):
    """ Check existence of a file. """
    if os.path.isfile(fname) or os.path.isfile(fname+'.nii') or os.path.isfile(fname+'.nii.gz'):
        pass
    else:
        print('\nERROR: '+fname+' does not exist. Exit program.\n')
        sys.exit(2)

def extract_fname(fname):
    """ Extracts path, file and extension. """
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
    
def get_orient(fname):
    """ Get the orientation (NEUROLOGICAL or RADIOLOGICAL) and return the correct swapping. """
    print("Orientation of the image:")
    p = os.popen('fslorient '+fname)
    s = p.readline()
    p.close()
    print(s)
    if 'NEUROLOGICAL' in s:
        swap = 'LR PA IS'
    elif 'RADIOLOGICAL' in s:
        swap = 'RL PA IS'
    else:
        print("Error could not find the orientation with fslorient")
    
    return swap
        
def parse_report(path):
    """ Return the volume informations contained in the SIENAX report. This
        is a dictionary with keys "grey", "white", and "brain". 
        The informations for the different tissues is a dictionary with the
        normalized and raw values, in cubic millimeters.
        
        adapted from: http://code.google.com/p/medipy/source/browse/plugins/fsl/sienax.py
        see license: http://code.google.com/p/medipy/source/browse/LICENSE
    """
    
    report = {}
    
    fd = open(path)
    for line in fd.readlines() :        
        for tissue in ["GREY", "WHITE", "BRAIN"] :
            pattern = tissue + r"\s+([\d+\.]+)\s+([\d+\.]+)"
            measure = re.match(pattern, line)
            if measure :
                normalized = float(measure.group(1))
                raw = float(measure.group(2))
                report[tissue.lower()] = {"normalized" : normalized, "raw" : raw}
                continue
        
        vscale = re.match("VSCALING ([\d\.]+)", line)
        if vscale :
            report["vscale"] = float(vscale.group(1))
    
    return report

# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    param = Param()
    main()
