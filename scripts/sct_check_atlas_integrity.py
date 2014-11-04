#!/usr/bin/env python
#########################################################################################
#
# Module converting image files
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# Created: 2014-09-22
#
# Dependences:
#   minc-toolkit - http://www.bic.mni.mcgill.ca/ServicesSoftware/ServicesSoftwareMincToolKit
#
# TO DO:
# - check if minc-toolkit is installed. If not, convert files using nibabel
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys
import commands
import getopt
import sct_utils as sct
import nibabel as nib
import numpy as np


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1
        self.file_info_label = 'info_label.txt'
        self.threshold_atlas = 0.25
        self.threshold_GM = 0.25
        self.fname_seg = ''
        self.fname_GM = ''

# constants
ALMOST_ZERO = 0.0000001


# main
#=======================================================================================================================
def main():
    
    # Initialization
    path_atlas = ''
    
    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hi:s:m:g:t:v:')
        except getopt.GetoptError:
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ('-i'):
                path_atlas = arg
            elif opt in ('-s'):
                param.fname_seg = arg
            elif opt in ('-m'):
                param.fname_GM = arg
            elif opt in ('-t'):
                param.threshold_atlas = float(arg)
            elif opt in ('-g'):
                param.threshold_GM = float(arg)
            elif opt in ('-v'):
                param.verbose = int(arg)
    
    # display usage if a mandatory argument is not provided
    if path_atlas == '' and not param.debug:
        # get path of the toolbox
        status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
        path_atlas = path_sct+'/data/atlas/'
        param.fname_seg = path_sct+'/data/template/MNI-Poly-AMU_cord.nii.gz'
        param.fname_GM = path_sct+'/data/template/MNI-Poly-AMU_GM.nii.gz'

    # print arguments
    if param.verbose:
        print 'Check input parameters...'
        print '.. Atlas folder path:                '+path_atlas
        print '.. Spinal cord segmentation path:    '+param.fname_seg
        print '.. Spinal cord gray matter path:     '+param.fname_GM
        print '.. Atlas threshold=                  '+str(param.threshold_atlas)

    # Check for end-caracter of folder path
    if path_atlas[-1] != "/": path_atlas=path_atlas+"/";

    # Check folder existence
    sct.printv('\nCheck atlas existence...', param.verbose)
    sct.check_file_exist(path_atlas)
    
    # Extract atlas info
    atlas_id, atlas_name, atlas_file = read_label_file(path_atlas)
    nb_tracts_total = len(atlas_id)

    # Load atlas
    sct.printv('\nLoad atlas...', param.verbose)
    atlas = np.empty([nb_tracts_total], dtype=object)  # labels(nb_labels_total, x, y, z)
    for i_atlas in range(0, nb_tracts_total):
        atlas[i_atlas] = nib.load(path_atlas+atlas_file[i_atlas]).get_data()

    # Check integrity
    sct.printv('\nCheck atlas integrity...', param.verbose)
    check_integrity(atlas, atlas_id, atlas_name)


#=======================================================================================================================
# Read label.txt file which is located inside label folder
#=======================================================================================================================
def read_label_file(path_info_label):

    # file name of info_label.txt
    fname_label = path_info_label+param.file_info_label

    # Check info_label.txt existence
    sct.check_file_exist(fname_label)

    # Read file
    f = open(fname_label)

    # Extract all lines in file.txt
    lines = [lines for lines in f.readlines() if lines.strip()]

    # separate header from (every line starting with "#")
    lines = [lines[i] for i in range(0, len(lines)) if lines[i][0] != '#']

    # read each line
    label_id = []
    label_name = []
    label_file = []
    for i in range(0, len(lines)-1):
        line = lines[i].split(',')
        label_id.append(int(line[0]))
        label_name.append(line[1])
        label_file.append(line[2][:-1].strip())
    # An error could occur at the last line (deletion of the last character of the .txt file), the 5 following code
    # lines enable to avoid this error:
    line = lines[-1].split(',')
    label_id.append(int(line[0]))
    label_name.append(line[1])
    line[2]=line[2]+' '
    label_file.append(line[2].strip())

    # check if all files listed are present in folder. If not, WARNING.
    print '\nCheck existence of all files listed in '+param.file_info_label+' ...'
    for fname in label_file:
        if os.path.isfile(path_info_label+fname) or os.path.isfile(path_info_label+fname + '.nii') or \
                os.path.isfile(path_info_label+fname + '.nii.gz'):
            print('  OK: '+path_info_label+fname)
            pass
        else:
            print('  WARNING: ' + path_info_label+fname + ' does not exist but is listed in '
                  +param.file_info_label+'.\n')

    # Close file.txt
    f.close()

    return [label_id, label_name, label_file]


#=======================================================================================================================
# Check integrity of the atlas
#=======================================================================================================================
def check_integrity(atlas, atlas_id, atlas_name, method='wath'):

    nb_tracts = len(atlas) # number of tracts

    # Get dimensions of the atlas
    sct.printv('\nGet dimensions of atlas...', param.verbose)
    nx_atlas, ny_atlas, nz_atlas = atlas[0].shape
    sct.printv('.. '+str(nx_atlas)+' x '+str(ny_atlas)+' x '+str(nz_atlas)+' x '+str(nb_tracts), param.verbose)


    # if user asks for binary regions, binarize atlas
    if method == 'bin':
        for i in range(0, nb_tracts):
            atlas[i][atlas[i] < param.threshold_atlas] = 0
            atlas[i][atlas[i] >= param.threshold_atlas] = 1

    # if user asks for thresholded weighted-average, threshold atlas
    if method == 'wath':
        for i in range(0, nb_tracts):
            atlas[i][atlas[i] < param.threshold_atlas] = 0

    # Does all the tracts are present?
    tracts_are_present = True
    sct.printv('\nDoes all the tracts are present in the atlas?',param.verbose)
    sum_tract = []
    for i_atlas in range(0, nb_tracts):
        sum_tract.append(np.sum(atlas[i_atlas]))
        if sum_tract[i_atlas] < ALMOST_ZERO:
            sct.printv('The tract #'+str(atlas_id[i_atlas])+atlas_name[i_atlas]+' is non-existent',param.verbose)
            tracts_are_present = False
    if tracts_are_present: sct.printv('All the tracts are present.',param.verbose)


    # Does any tract gets out the spinal cord?
    if param.fname_seg != '':
        # Loading spinal cord segmentation
        segmentation = nib.load(param.fname_seg).get_data()
        sct.printv('\nGet dimensions of segmentation image...', param.verbose)
        nx_seg, ny_seg, nz_seg = segmentation.shape
        sct.printv('.. ' + str(nx_seg) + ' x ' + str(ny_seg) + ' x ' + str(nz_seg), param.verbose)

        # Check dimensions consistency between atlas and segmentation image
        if (nx_seg, ny_seg, nz_seg) != (nx_atlas, ny_atlas, nz_atlas):
            print '\nERROR: Segmentation image and the atlas DO NOT HAVE SAME DIMENSIONS.'
            sys.exit(2)

        tracts_are_inside_SC = True
        total_outside = 0
        total_sum_tracts = 0
        sct.printv('\nDoes any tract gets out the spinal cord?',param.verbose)
        ind_seg_outside_cord = segmentation<=ALMOST_ZERO
        for i_atlas in range(0, nb_tracts):
            ind_atlas_positive = atlas[i_atlas]>=ALMOST_ZERO
            sum_tract_outside_SC = np.sum(atlas[i_atlas][ind_atlas_positive & ind_seg_outside_cord])
            sum_tract = np.sum(atlas[i_atlas][ind_atlas_positive])
            if sum_tract_outside_SC > ALMOST_ZERO:
                percentage_out = float(sum_tract_outside_SC/sum_tract)
                sct.printv('The tract #'+str(atlas_id[i_atlas])+atlas_name[i_atlas]+' gets out the spinal cord of '+str(round(percentage_out*100,2))+'%',param.verbose)
                tracts_are_inside_SC = False
                total_outside += sum_tract_outside_SC
            total_sum_tracts += sum_tract
        if tracts_are_inside_SC: sct.printv('All the tracts are inside the spinal cord.',param.verbose)
        else:
            total_percentage_out = float(total_outside/total_sum_tracts)
            sct.printv('\nTotal percentage of present tracts outside the spinal cord: ' + str(round(total_percentage_out*100, 2)) + '%', param.verbose)


    # Does any tract overlaps the spinal cord gray matter?
    if param.fname_GM != '':
        # Loading spinal cord gray matter
        graymatter = nib.load(param.fname_GM).get_data()
        sct.printv('\nGet dimensions of gray matter image...', param.verbose)
        nx_gm, ny_gm, nz_gm = graymatter.shape
        sct.printv('.. ' + str(nx_gm) + ' x ' + str(ny_gm) + ' x ' + str(nz_gm), param.verbose)

        # Check dimensions consistency between atlas and spinal cord gray matter image
        if (nx_gm, ny_gm, nz_gm) != (nx_atlas, ny_atlas, nz_atlas):
            print '\nERROR: Gray matter image and the atlas DO NOT HAVE SAME DIMENSIONS.'
            sys.exit(2)
    
        tracts_overlap_GM = False
        total_overlaps = 0
        total_sum_tracts = 0
        sct.printv('\nDoes any tract overlaps the spinal cord gray matter?',param.verbose)
        ind_GM = graymatter>=param.threshold_GM
        for i_atlas in range(0, nb_tracts):
            ind_atlas_positive = atlas[i_atlas]>=ALMOST_ZERO
            sum_tract_overlap_GM = np.sum(atlas[i_atlas][ind_atlas_positive & ind_GM])
            sum_tract = np.sum(atlas[i_atlas])
            if sum_tract_overlap_GM > ALMOST_ZERO:
                percentage_overlap = float(sum_tract_overlap_GM/sum_tract)
                sct.printv('The tract #'+str(atlas_id[i_atlas])+atlas_name[i_atlas]+' overlaps the spinal cord gray matter of '+str(round(percentage_overlap*100,2))+'%',param.verbose)
                tracts_overlap_GM = True
                total_overlaps += sum_tract_overlap_GM
            total_sum_tracts += sum_tract
        if not tracts_overlap_GM: sct.printv('No tract overlaps the spinal cord gray matter.',param.verbose)
        else:
            total_percentage_overlap = float(total_overlaps/total_sum_tracts)
            sct.printv('\nTotal percentage of present tracts overlapping grey matter: ' + str(round(total_percentage_overlap*100, 2)) + '%', param.verbose)

# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>
        
DESCRIPTION
Check the integrity of the spinal cord internal structure (tracts) atlas.

USAGE
"""+os.path.basename(__file__)+""" -i <atlas>
            
    MANDATORY ARGUMENTS
    -i <folder>           atlas folder path.
            
    OPTIONAL ARGUMENTS
    -s <segmentation>     segmentation image path.
    -m <graymatter>       gray matter image path.
    -t [0,1]              float, atlas threshold. Default="""+str(param_default.threshold_atlas)+"""
    -g [0,1]              float, gray matter image threshold. Default="""+str(param_default.threshold_GM)+"""
    -v {0,1}              verbose. Default="""+str(param_default.verbose)+"""
    -h                    help. Show this message.
    """
    # exit program
    sys.exit(2)



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()