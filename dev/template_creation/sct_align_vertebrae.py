#!/usr/bin/env python
#########################################################################################
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Touati, Benjamin De Leener
# Created: 2014-08-11
# Modified: 2015-01-16
# 
# This script aligns the vertabral levels from the input image on the template space.
# The tempalte space has fixed vertebral levels.
# The script uses ANTs registration with SyN transformation and is restricted to
# axial deformation (in the z direction).
# 
# Exemple: template generation
# Requirements are an image with incremental labels (using fslview, label=1, then use sct_label_utils with -t increment) at the center of each vertebral body
# You may start at PMJ then C3 to avoid unprecision of C1 and C2 vertebral bodies.
#
# About the license: see the file LICENSE.TXT
#########################################################################################


#DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.verbose = 1
        self.final_warp = ''
        self.compose = 0
        
# check if needed Python libraries are already installed or not
import sys,commands
import getopt


# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))

import numpy as np
import sct_utils as sct
import os
import time
from commands import getstatusoutput
def main():
    
    
    # get path of the toolbox
    status, path_sct = getstatusoutput('echo $SCT_DIR')
    #print path_sct


    #Initialization
    fname = ''
    landmark = ''
    verbose = param.verbose
    output_name = 'aligned.nii.gz'
    template_landmark = ''
    final_warp = param.final_warp
    compose = param.compose
    transfo = 'affine'
        
    try:
         opts, args = getopt.getopt(sys.argv[1:],'hi:l:o:R:t:w:c:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts :
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            fname = arg
        elif opt in ("-l"):
            landmark = arg       
        elif opt in ("-o"):
            output_name = arg  
        elif opt in ("-R"):
            template_landmark = arg
        elif opt in ("-t"):
            transfo = arg    
        elif opt in ("-w"):
            final_warp = arg
        elif opt in ("-c"):
            compose = int(arg)                          
        elif opt in ('-v'):
            verbose = int(arg)
    
    # display usage if a mandatory argument is not provided
    if fname == '' or landmark == '' or template_landmark == '' :
        usage()
        
    if final_warp not in ['','spline','NN']:
        usage()
        
    if transfo not in ['affine', 'bspline', 'SyN', 'nurbs']:
        usage()       
    
    # check existence of input files
    print'\nCheck if file exists ...'
    
    sct.check_file_exist(fname)
    sct.check_file_exist(landmark)
    sct.check_file_exist(template_landmark)
    
    
        
    # Display arguments
    print'\nCheck input arguments...'
    print'  Input volume ...................... '+fname
    print'  Verbose ........................... '+str(verbose)

    if transfo == 'affine':
        print 'Creating cross using input landmarks\n...'
        sct.run('sct_label_utils -i ' + landmark + ' -o ' + 'cross_native.nii.gz -t cross ' )
    
        print 'Creating cross using template landmarks\n...'
        sct.run('sct_label_utils -i ' + template_landmark + ' -o ' + 'cross_template.nii.gz -t cross ' )
    
        print 'Computing affine transformation between subject and destination landmarks\n...'
        os.system('isct_ANTSUseLandmarkImagesToGetAffineTransform cross_template.nii.gz cross_native.nii.gz affine n2t.txt')
        warping = 'n2t.txt'
    elif transfo == 'nurbs':
        warping_subject2template = 'warp_subject2template.nii.gz'
        warping_template2subject = 'warp_template2subject.nii.gz'
        tmp_name = 'tmp.' + time.strftime("%y%m%d%H%M%S")
        sct.run('mkdir ' + tmp_name)
        tmp_abs_path = os.path.abspath(tmp_name)
        sct.run('cp ' + landmark + ' ' + tmp_abs_path)
        os.chdir(tmp_name)

        from msct_image import Image
        image_landmark = Image(landmark)
        image_template = Image(template_landmark)
        landmarks_input = image_landmark.getNonZeroCoordinates(sorting='value')
        landmarks_template = image_template.getNonZeroCoordinates(sorting='value')
        min_value = min([int(landmarks_input[0].value), int(landmarks_template[0].value)])
        max_value = max([int(landmarks_input[-1].value), int(landmarks_template[-1].value)])
        nx, ny, nz, nt, px, py, pz, pt = image_landmark.dim

        displacement_subject2template, displacement_template2subject = [], []
        for value in range(min_value, max_value+1):
            is_in_input = False
            coord_input = None
            for coord in landmarks_input:
                if int(value) == int(coord.value):
                    coord_input = coord
                    is_in_input = True
                    break
            is_in_template = False
            coord_template = None
            for coord in landmarks_template:
                if int(value) == int(coord.value):
                    coord_template = coord
                    is_in_template = True
                    break
            if is_in_template and is_in_input:
                displacement_subject2template.append([0.0, coord_input.z, coord_template.z - coord_input.z])
                displacement_template2subject.append([0.0, coord_template.z, coord_input.z - coord_template.z])

        # create displacement field
        from numpy import zeros
        from nibabel import Nifti1Image, save
        data_warp_subject2template = zeros((nx, ny, nz, 1, 3))
        data_warp_template2subject = zeros((nx, ny, nz, 1, 3))
        hdr_warp = image_template.hdr.copy()
        hdr_warp.set_intent('vector', (), '')
        hdr_warp.set_data_dtype('float32')

        # approximate displacement with nurbs
        from msct_smooth import b_spline_nurbs__todo_path_qc_eq_curdir
        displacement_z = [item[1] for item in displacement_subject2template]
        displacement_x = [item[2] for item in displacement_subject2template]
        verbose = 1
        displacement_z, displacement_y, displacement_y_deriv, displacement_z_deriv = b_spline_nurbs__todo_path_qc_eq_curdir(displacement_x, displacement_z, None, nbControl=None, verbose=verbose, all_slices=True)

        arg_min_z, arg_max_z = np.argmin(displacement_y), np.argmax(displacement_y)
        min_z, max_z = int(displacement_y[arg_min_z]), int(displacement_y[arg_max_z])
        displac = []
        for index, iz in enumerate(displacement_y):
            displac.append([iz, displacement_z[index]])
        for iz in range(0, min_z):
            displac.append([iz, displacement_z[arg_min_z]])

        for iz in range(max_z, nz):
            displac.append([iz, displacement_z[arg_max_z]])

        for item in displac:
            if 0 <= item[0] < nz:
                data_warp_template2subject[:, :, item[0], 0, 2] = item[1] * pz

        displacement_z = [item[1] for item in displacement_template2subject]
        displacement_x = [item[2] for item in displacement_template2subject]
        verbose = 1
        displacement_z, displacement_y, displacement_y_deriv, displacement_z_deriv = b_spline_nurbs__todo_path_qc_eq_curdir(displacement_x, displacement_z, None, nbControl=None, verbose=verbose, all_slices=True)

        arg_min_z, arg_max_z = np.argmin(displacement_y), np.argmax(displacement_y)
        min_z, max_z = int(displacement_y[arg_min_z]), int(displacement_y[arg_max_z])
        displac = []
        for index, iz in enumerate(displacement_y):
            displac.append([iz, displacement_z[index]])
        for iz in range(0, min_z):
            displac.append([iz, displacement_z[arg_min_z]])
        for iz in range(max_z, nz):
            displac.append([iz, displacement_z[arg_max_z]])

        for item in displac:
            data_warp_subject2template[:, :, item[0], 0, 2] = item[1] * pz

        img = Nifti1Image(data_warp_template2subject, None, hdr_warp)
        save(img, warping_template2subject)
        sct.printv('\nDONE ! Warping field generated: ' + warping_template2subject, verbose)
        img = Nifti1Image(data_warp_subject2template, None, hdr_warp)
        save(img, warping_subject2template)
        sct.printv('\nDONE ! Warping field generated: ' + warping_subject2template, verbose)

        # Copy warping into parent folder
        sct.run('cp ' + warping_subject2template + ' ../' + warping_subject2template)
        sct.run('cp ' + warping_template2subject + ' ../' + warping_template2subject)
        warping = warping_subject2template

        os.chdir('..')
        remove_temp_files = True
        if remove_temp_files:
            sct.run('rm -rf ' + tmp_name)

    elif transfo == 'SyN':
        warping = 'warp_subject2template.nii.gz'
        tmp_name = 'tmp.'+time.strftime("%y%m%d%H%M%S")
        sct.run('mkdir '+tmp_name)
        tmp_abs_path = os.path.abspath(tmp_name)
        sct.run('cp ' + landmark + ' ' + tmp_abs_path)
        os.chdir(tmp_name)

        # sct.run('sct_label_utils -i '+landmark+' -t dist-inter')
        # sct.run('sct_label_utils -i '+template_landmark+' -t plan -o template_landmarks_plan.nii.gz -c 5')
        # sct.run('sct_crop_image -i template_landmarks_plan.nii.gz -o template_landmarks_plan_cropped.nii.gz -start 0.35,0.35 -end 0.65,0.65 -dim 0,1')
        # sct.run('sct_label_utils -i '+landmark+' -t plan -o landmarks_plan.nii.gz -c 5')
        # sct.run('sct_crop_image -i landmarks_plan.nii.gz -o landmarks_plan_cropped.nii.gz -start 0.35,0.35 -end 0.65,0.65 -dim 0,1')
        # sct.run('isct_antsRegistration --dimensionality 3 --transform SyN[0.5,3,0] --metric MeanSquares[template_landmarks_plan_cropped.nii.gz,landmarks_plan_cropped.nii.gz,1] --convergence 400x200 --shrink-factors 4x2 --smoothing-sigmas 4x2mm --restrict-deformation 0x0x1 --output [landmarks_reg,landmarks_reg.nii.gz] --interpolation NearestNeighbor --float')
        # sct.run('isct_c3d -mcs landmarks_reg0Warp.nii.gz -oo warp_vecx.nii.gz warp_vecy.nii.gz warp_vecz.nii.gz')
        # sct.run('isct_c3d warp_vecz.nii.gz -resample 200% -o warp_vecz_r.nii.gz')
        # sct.run('isct_c3d warp_vecz_r.nii.gz -smooth 0x0x3mm -o warp_vecz_r_sm.nii.gz')
        # sct.run('sct_crop_image -i warp_vecz_r_sm.nii.gz -o warp_vecz_r_sm_line.nii.gz -start 0.5,0.5 -end 0.5,0.5 -dim 0,1 -b 0')
        # sct.run('sct_label_utils -i warp_vecz_r_sm_line.nii.gz -t plan_ref -o warp_vecz_r_sm_line_extended.nii.gz -c 0 -r '+template_landmark)
        # sct.run('isct_c3d '+template_landmark+' warp_vecx.nii.gz -reslice-identity -o warp_vecx_res.nii.gz')
        # sct.run('isct_c3d '+template_landmark+' warp_vecy.nii.gz -reslice-identity -o warp_vecy_res.nii.gz')
        # sct.run('isct_c3d warp_vecx_res.nii.gz warp_vecy_res.nii.gz warp_vecz_r_sm_line_extended.nii.gz -omc 3 '+warping)  # no x?


        #new
        #put labels of the subject at the center of the image (for plan xOy)
        import nibabel
        from copy import copy
        file_labels_input = nibabel.load(landmark)
        hdr_labels_input = file_labels_input.get_header()
        data_labels_input = file_labels_input.get_data()
        data_labels_middle = copy(data_labels_input)
        data_labels_middle *= 0
        from msct_image import Image
        nx, ny, nz, nt, px, py, pz, pt = Image(landmark).dim
        X,Y,Z = data_labels_input.nonzero()
        x_middle = int(round(nx/2.0))
        y_middle = int(round(ny/2.0))

        #put labels of the template at the center of the image (for plan xOy)  #probably not necessary as already done by average labels
        file_labels_template = nibabel.load(template_landmark)
        hdr_labels_template = file_labels_template.get_header()
        data_labels_template = file_labels_template.get_data()
        data_template_middle = copy(data_labels_template)
        data_template_middle *= 0

        x, y, z = data_labels_template.nonzero()

        max_num = min([len(z), len(Z)])
        index_sort = np.argsort(Z)
        index_sort = index_sort[::-1]
        X = X[index_sort]
        Y = Y[index_sort]
        Z = Z[index_sort]
        index_sort = np.argsort(z)
        index_sort = index_sort[::-1]
        x = x[index_sort]
        y = y[index_sort]
        z = z[index_sort]

        for i in range(max_num):
            data_labels_middle[x_middle, y_middle, Z[i]] = data_labels_input[X[i], Y[i], Z[i]]
        img = nibabel.Nifti1Image(data_labels_middle, None, hdr_labels_input)
        nibabel.save(img, 'labels_input_middle_xy.nii.gz')

        for i in range(max_num):
            data_template_middle[x_middle, y_middle, z[i]] = data_labels_template[x[i], y[i], z[i]]
        img_template = nibabel.Nifti1Image(data_template_middle, None, hdr_labels_template)
        nibabel.save(img_template, 'labels_template_middle_xy.nii.gz')


        #estimate Bspline transform to register to template
        sct.run('isct_ANTSUseLandmarkImagesToGetBSplineDisplacementField labels_template_middle_xy.nii.gz labels_input_middle_xy.nii.gz '+ warping+' 40x40x1 5 5 0')

        # select centerline of warping field according to z and extend it
        sct.run('isct_c3d -mcs '+warping+' -oo warp_vecx.nii.gz warp_vecy.nii.gz warp_vecz.nii.gz')
        #sct.run('isct_c3d warp_vecz.nii.gz -resample 200% -o warp_vecz_r.nii.gz')
        #sct.run('isct_c3d warp_vecz.nii.gz -smooth 0x0x3mm -o warp_vecz_r_sm.nii.gz')
        sct.run('sct_crop_image -i warp_vecz.nii.gz -o warp_vecz_r_sm_line.nii.gz -start 0.5,0.5 -end 0.5,0.5 -dim 0,1 -b 0')
        sct.run('sct_label_utils -i warp_vecz_r_sm_line.nii.gz -t plan_ref -o warp_vecz_r_sm_line_extended.nii.gz -r '+template_landmark)
        sct.run('isct_c3d '+template_landmark+' warp_vecx.nii.gz -reslice-identity -o warp_vecx_res.nii.gz')
        sct.run('isct_c3d '+template_landmark+' warp_vecy.nii.gz -reslice-identity -o warp_vecy_res.nii.gz')
        sct.run('isct_c3d warp_vecx_res.nii.gz warp_vecy_res.nii.gz warp_vecz_r_sm_line_extended.nii.gz -omc 3 '+warping)

        # check results
        #dilate first labels
        sct.run('fslmaths labels_input_middle_xy.nii.gz -dilF landmark_dilated.nii.gz') #new
        sct.run('sct_apply_transfo -i landmark_dilated.nii.gz -o label_moved.nii.gz -d labels_template_middle_xy.nii.gz -w '+warping+' -x nn')
        #undilate
        sct.run('sct_label_utils -i label_moved.nii.gz -t cubic-to-point -o label_moved_2point.nii.gz')
        sct.run('sct_label_utils -i labels_template_middle_xy.nii.gz -r label_moved_2point.nii.gz -o template_removed.nii.gz -t remove')
        #end new


        # check results
        #dilate first labels

        #sct.run('fslmaths '+landmark+' -dilF landmark_dilated.nii.gz') #old
        #sct.run('sct_apply_transfo -i landmark_dilated.nii.gz -o label_moved.nii.gz -d '+template_landmark+' -w '+warping+' -x nn') #old



        #undilate
        #sct.run('sct_label_utils -i label_moved.nii.gz -t cubic-to-point -o label_moved_2point.nii.gz') #old

        #sct.run('sct_label_utils -i '+template_landmark+' -r label_moved_2point.nii.gz -o template_removed.nii.gz -t remove') #old


        # # sct.run('sct_apply_transfo -i '+landmark+' -o label_moved.nii.gz -d '+template_landmark+' -w '+warping+' -x nn')
        # # sct.run('sct_label_utils -i '+template_landmark+' -r label_moved.nii.gz -o template_removed.nii.gz -t remove')
        # # status, output = sct.run('sct_label_utils -i label_moved.nii.gz -r template_removed.nii.gz -t MSE')

        status, output = sct.run('sct_label_utils -i label_moved_2point.nii.gz -r template_removed.nii.gz -t MSE')
        sct.printv(output,1,'info')
        remove_temp_files = False
        if os.path.isfile('error_log_label_moved.txt'):
            remove_temp_files = False
            with open('log.txt', 'a') as log_file:
                log_file.write('Error for '+fname+'\n')
        # Copy warping into parent folder
        sct.run('cp '+ warping+' ../'+warping)

        os.chdir('..')
        if remove_temp_files:
            sct.run('rm -rf '+tmp_name)



    # if transfo == 'bspline' :
    #     print 'Computing bspline transformation between subject and destination landmarks\n...'
    #     sct.run('isct_ANTSUseLandmarkImagesToGetBSplineDisplacementField cross_template.nii.gz cross_native.nii.gz warp_ntotemp.nii.gz 5x5x5 3 2 0')
    #     warping = 'warp_ntotemp.nii.gz'
        
    # if final_warp == '' :    
    #     print 'Apply transfo to input image\n...'
    #     sct.run('isct_antsApplyTransforms 3 ' + fname + ' ' + output_name + ' -r ' + template_landmark + ' -t ' + warping + ' -n Linear')
        
    # if final_warp == 'NN':
    #     print 'Apply transfo to input image\n...'
    #     sct.run('isct_antsApplyTransforms 3 ' + fname + ' ' + output_name + ' -r ' + template_landmark + ' -t ' + warping + ' -n NearestNeighbor')
    if final_warp == 'spline':
        print 'Apply transfo to input image\n...'
        sct.run('sct_apply_transfo -i ' + fname + ' -o ' + output_name + ' -d ' + template_landmark + ' -w ' + warping + ' -x spline')


    # Remove warping
    #os.remove(warping)

    # if compose :
        
    #     print 'Computing affine transformation between subject and destination landmarks\n...'
    #     sct.run('isct_ANTSUseLandmarkImagesToGetAffineTransform cross_template.nii.gz cross_native.nii.gz affine n2t.txt')
    #     warping_affine = 'n2t.txt'
        
        
    #     print 'Apply transfo to input landmarks\n...'
    #     sct.run('isct_antsApplyTransforms 3 ' + cross_native + ' cross_affine.nii.gz -r ' + template_landmark + ' -t ' + warping_affine + ' -n NearestNeighbor')
        
    #     print 'Computing transfo between moved landmarks and template landmarks\n...'
    #     sct.run('isct_ANTSUseLandmarkImagesToGetBSplineDisplacementField cross_template.nii.gz cross_affine.nii.gz warp_affine2temp.nii.gz 5x5x5 3 2 0')
    #     warping_bspline = 'warp_affine2temp.nii.gz'
        
    #     print 'Composing transformations\n...'
    #     sct.run('isct_ComposeMultiTransform 3 warp_full.nii.gz -r ' + template_landmark + ' ' + warping_bspline + ' ' + warping_affine)
    #     warping_concat = 'warp_full.nii.gz'
        
    #     if final_warp == '' :    
    #         print 'Apply concat warp to input image\n...'
    #         sct.run('isct_antsApplyTransforms 3 ' + fname + ' ' + output_name + ' -r ' + template_landmark + ' -t ' + warping_concat + ' -n Linear')
        
    #     if final_warp == 'NN':
    #         print 'Apply concat warp to input image\n...'
    #         sct.run('isct_antsApplyTransforms 3 ' + fname + ' ' + output_name + ' -r ' + template_landmark + ' -t ' + warping_concat + ' -n NearestNeighbor')
        
    #     if final_warp == 'spline':
    #         print 'Apply concat warp to input image\n...'
    #         sct.run('isct_antsApplyTransforms 3 ' + fname + ' ' + output_name + ' -r ' + template_landmark + ' -t ' + warping_concat + ' -n BSpline[3]')
          
    
    
    print '\nFile created : ' + output_name
  
    
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
Takes an input volume, a mask containing labels in the spinalcord at several levels, a mask
containning the same labels in the space you want to push into. It registers your input image
using transformation between landmark images.

USAGE
  """+os.path.basename(__file__)+"""  -i <input_volume> -l <landmark_native> -R <template_landmark>

MANDATORY ARGUMENTS
  -i <input_volume>         input image. No Default value
  -l <landmark_native>      mask with labels. No Default Value
  -R <template_landmark>    mask with labels in template space. No Default Value


OPTIONAL ARGUMENTS
  -o <output_name>          output name. Default : aligned.nii.gz
  -t {affine,bspline,SyN}   type of initial transformation. Default : affine
  -w {NN,spline}            final warp interpolation. Default : trilinear
  -c {0,1}                  compose affine and bspline transformation. Default="""+str(param.compose)+"""
  -v {0,1}                  verbose. Default="""+str(param.verbose)+"""
  -h                        help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i volume_image.nii.gz -l landmarks_native.nii.gz -R landmarks_template.nii.gz -t bspline -w NN\n"""

    # exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()






