#!/usr/bin/env python
#########################################################################################
# Extract raw intracranial volume in cubic millimeters.
# Method: bias field correction (N4) followed by SIENAx
#
# USAGE
# ---------------------------------------------------------------------------------------
#   sct_icv.py -i <inputvol> -c <contrast> [options]
#
#
# INPUT
# ---------------------------------------------------------------------------------------
# -i inputvol          volume. Can be nii or nii.gz
# -c contrast          (t1|t2) contrast
# -o output-dir        set output directory (create directory if not exists) 
# -v verbose           (0|1)
#
# OUTPUT
# ---------------------------------------------------------------------------------------
# text file with the value of raw intracranial volume in cubic millimeters
# SIENAx report in separate folder
#
#
# DEPENDENCIES
# ---------------------------------------------------------------------------------------
#
# EXTERNAL SOFTWARE
# - FSL (SIENAx) <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/SIENA>
# - ANTS (N4BiasFieldCorrection) <http://stnava.github.io/ANTs/>
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
# Author: William Thong
# Modified: 2013-12-13 18:00
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os, sys, getopt, re



# Fixed parameters
# ==========================================================================================
N4Correct=True
ImageDimension=3
sienax=False
RBM=True
path_atlas='${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz'
path_mask_mni='${FSLDIR}/data/standard/MNI152_T1_2mm_brain_mask.nii.gz'

def main():
    
    # Initialization
    input_path = ''
    contrast = ''
    output_path = ''
    verbose = 1
    debugging=False
    # Check input parameters
    if debugging:
        pass
    else:
        try:
            opts, args = getopt.getopt(sys.argv[1:],'hi:c:o:v:')
        except getopt.GetoptError:
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
            elif opt in ("-v"):
                verbose = int(arg)


    # check mandatory arguments
    if input_path == '' or contrast == '' :
        usage()

    # Normalize paths (remove trailing slash)
    input_path = os.path.normpath(input_path)

    # Extract path, file and extension
    path_input, file_input, ext_input = extract_fname(input_path)

    # define output_path
    if output_path=='':
        output_path=path_input
    else:
        output_path = os.path.normpath(output_path)
    
    #check existence of directories (if exists, removes subdirectories; if not, creates directory)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # print arguments
    if verbose:
        print 'Check input parameters...'
        print '.. Image:    '+input_path
        print '.. Contrast:    '+str(contrast)
        print '.. Output directory:     '+output_path
    

    ####################################
    # The core of the code starts here #
    ####################################

    #-----------------------------------
    # Field correction with N4
    # Note that N3 is implemented in BET, use -B parameter
    #-----------------------------------
    if N4Correct:
        cmd='N4BiasFieldCorrection -d '+str(ImageDimension)+' -i '+input_path+' -o ' + os.path.join(output_path, file_input + '_n4'+ext_input)
        print(">> "+cmd)
        os.system(cmd)
        file_input = file_input+'_n4'
    
    if sienax:
        #-----------------------------------    
        # Parameters for bet and use SIENAx
        #-----------------------------------    
        # -f Fractional intensity threshold (smaller values give larger brain) equals 0.2 for both contrasts
        # -R Robust brain centre estimation (iterates BET several times)
        if contrast == 't2':
            frac_int = 0.2
            # note that -t2 parameter is important in SIENAx to estimate white matter volume properly
            cmd='sienax '+os.path.join(output_path, file_input+ext_input) +' -o '+os.path.join(output_path, file_input+'_sienax') + ' -B "-f '+str(frac_int)+' -R" -t2' 
            print(">> "+cmd)
            os.system(cmd)
        elif contrast == 't1':
            frac_int = 0.2
            cmd='sienax '+os.path.join(output_path, file_input+ext_input) +' -o '+os.path.join(output_path, file_input+'_sienax') + ' -B "-f '+str(frac_int)+' -R"'
            print(">> "+cmd)
            os.system(cmd)    
        
        #-----------------------------------    
        # Read SIENAx report to extract raw brain ICV in cubic millimeters
        #-----------------------------------   
        report = parse_report(os.path.join(output_path, file_input+'_sienax', 'report.sienax'))
        
        fo = open(os.path.join(output_path, "icv.txt"), "wb")
        fo.write(str(report['brain']['raw']))
        fo.close()
    
    if RBM:
        # Brain extraction
        frac_int = 0.2
        file_output='tmp.brain.'+file_input
        cmd = 'bet '+os.path.join(output_path, file_input+ext_input) + ' '+ os.path.join(output_path, file_output+ext_input) +' -R -f '+str(frac_int)
        print(">> "+cmd)
        os.system(cmd)
        
        # Swap dimension to correspond to the template
        file_output='tmp.brain.RLPAIS.'+file_input
        cmd = 'fslswapdim '+os.path.join(output_path, 'tmp.brain.'+file_input+ext_input) +' LR PA IS '+ os.path.join(output_path, file_output+ext_input)
        print(">> "+cmd)
        os.system(cmd)
        
        # Resample in 2mm
        cmd = 'c3d '+ os.path.join(output_path, file_output+ext_input) +' -resample-mm 2.0x2.0x2.0mm '+ os.path.join(output_path, file_output+'_2mm'+ext_input)
        print(">> "+cmd)
        os.system(cmd)
        file_output=file_output+'_2mm' 
        
        # Registration on template        
        cmd = 'flirt -ref '+path_atlas+' -in '+ os.path.join(output_path, file_output+ext_input) +' -out '+ os.path.join(output_path, file_output+'_reg'+ext_input) +' -omat '+ os.path.join(output_path, 'tmp.'+file_input+'_affine_transf.mat')
        print(">> "+cmd)
        os.system(cmd)
        file_output = file_output+'_reg'   
        
        # Apply binary mask
        cmd = 'fslmaths '+ os.path.join(output_path, file_output+ext_input)+' -mas '+path_mask_mni+' '+ os.path.join(output_path, file_output+'_masked'+ext_input)
        print(">> "+cmd)
        os.system(cmd)
        file_output = file_output+'_masked'
        
        # invert transformation matrix
        cmd = 'convert_xfm -omat '+ os.path.join(output_path, 'tmp.'+file_input+'_affine_inverse_transf.mat') + ' -inverse '+ os.path.join(output_path, 'tmp.'+file_input+'_affine_transf.mat')
        print(">> "+cmd)
        os.system(cmd)      
        
        # apply inverse transmation matrix 
        cmd = 'flirt -ref '+os.path.join(output_path, file_input+ext_input) +' -in '+ os.path.join(output_path, file_output+ext_input) +' -out '+ os.path.join(output_path, file_input+'_brain'+ext_input) +' -init '+ os.path.join(output_path, 'tmp.'+file_input+'_affine_inverse_transf.mat') + ' -applyxfm'
        print(">> "+cmd)
        os.system(cmd)

        p = os.popen('fslstats '+os.path.join(output_path, file_input+'_brain'+ext_input) +' -V')
        s = p.readline()
        p.close()
        
        fo = open(os.path.join(output_path, "icv.txt"), "wb")
        fo.write(s)
        fo.close()

        
def usage():
    """ Print usage """
    path_func, file_func, ext_func = extract_fname(sys.argv[0])
    print '\nUsage:\n' \
        '   '+file_func+ext_func+' -i <inputvol> -c <contrast> [options]\n\n' \
        'Options:\n' \
        '  -i inputvol          image to extract values from\n' \
        '  -c contrast          image contrast. t1 or t2, e.g. -c t1\n' \
        '  -o output            set output directory (create directory if not exists) \n' \
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
    path_fname = os.path.dirname(fname)
    # extract file and extension
    file_fname = os.path.basename(fname)
    file_fname, ext_fname = os.path.splitext(file_fname)
    # check if .nii.gz file
    if ext_fname == '.gz':
        file_fname = file_fname[0:len(file_fname)-4]
        ext_fname = ".nii.gz"
    return path_fname, file_fname, ext_fname
        
def parse_report(path):
    """ Return the volume informations contained in the SIENAX report. This
        is a dictionary with keys "grey", "white", and "brain". 
        The informations for the different tissues is a dictionary with the
        normalized and raw values, in cubic millimeters.
        
        adapted from: http://code.google.com/p/medipy/source/browse/plugins/fsl/sienax.py
        see licence: http://code.google.com/p/medipy/source/browse/LICENSE
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
    main()
