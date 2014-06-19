#!/usr/bin/env python
#########################################################################################
#
# Perform various types of processing from the spinal cord segmentation (e.g. extract centerline, compute CSA, etc.).
# (extract_centerline) extract the spinal cord centerline from the segmentation. Output file is an image in the same
# space as the segmentation.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Touati, Gabriel Mangeat
# Created: 2014-05-24
#
# About the license: see the file LICENSE.TXT
#########################################################################################



# DEFAULT PARAMETERS



class param:
    ## The constructor
    def __init__(self):
        self.debug              = 0
        self.verbose            = 1 # verbose
        self.step               = 1 # step of discretized plane in mm
        self.remove_temp_files  = 1

import re
import math
import sys
import getopt
import os
import commands
import numpy as np
import time
import sct_utils as sct
from sct_nurbs import NURBS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import imsave
try:
    import nibabel
except ImportError:
    print '--- nibabel not installed! Exit program. ---'
    sys.exit(2)

# MAIN
# ==========================================================================================
def main():
    
    # Initialization
    path_script = os.path.dirname(__file__)
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI
    # THIS DOES NOT WORK IN MY LAPTOP: path_sct = os.environ['SCT_DIR'] # path to spinal cord toolbox
    #path_sct = path_script[:-8] # TODO: make it cleaner!
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
    fname_segmentation = ''
    name_process = ''
    processes = ['extract_centerline','compute_CSA']
    verbose = param.verbose
    start_time = time.time()
    remove_temp_files = param.remove_temp_files
    
    # Parameters for debug mode
    if param.debug:
        fname_segmentation = path_sct+'/testing/data/errsm_23/t2/t2_manual_segmentation.nii.gz'
        verbose = 1
        remove_temp_files = 0
    
    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:p:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            fname_segmentation = arg
        elif opt in ("-p"):
            name_process = arg
        elif opt in ('-v'):
            verbose = int(arg)
    
    # display usage if a mandatory argument is not provided
    if fname_segmentation == '' or name_process == '':
        usage()
	
    # display usage if the requested process is not available
    if name_process not in processes:
        usage()
	
    # check existence of input files
    sct.check_file_exist(fname_segmentation)
	
    # print arguments
    print '\nCheck parameters:'
    print '.. segmentation file:             '+fname_segmentation
	
    if name_process == 'extract_centerline':
        extract_centerline(fname_segmentation)
    
    if name_process == 'compute_CSA':
        compute_CSA(fname_segmentation)
    
    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'

# End of Main


# EXTRACT_CENTERLINE
# ==========================================================================================

def extract_centerline(fname_segmentation):
    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)
	
    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp)
    
    # copy files into tmp folder
    sct.run('cp '+fname_segmentation+' '+path_tmp)
    
    # go to tmp folder
    os.chdir(path_tmp)
    
    remove_temp_files = param.remove_temp_files
    
    # Change orientation of the input segmentation into RPI
    print '\nOrient segmentation image to RPI orientation...'
    fname_segmentation_orient = 'tmp.segmentation_rpi' + ext_data
    sct.run('sct_orientation -i ' + file_data+ext_data + ' -o ' + fname_segmentation_orient + ' -orientation RPI')
	
    # Extract orientation of the input segmentation
    status,sct_orientation_output = sct.run('sct_orientation -i ' + file_data+ext_data + ' -get')
    orientation = sct_orientation_output[-3:]
    print '\nOrientation of segmentation image: ' + orientation
	
    # Get size of data
    print '\nGet dimensions data...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_segmentation_orient)
    print '.. '+str(nx)+' x '+str(ny)+' y '+str(nz)+' z '+str(nt)
	
    print '\nOpen segmentation volume...'
    file = nibabel.load(fname_segmentation_orient)
    data = file.get_data()
    hdr = file.get_header()
	
    # Extract min and max index in Z direction
    X, Y, Z = (data>0).nonzero()
    print data
    min_z_index, max_z_index = min(Z), max(Z)
    x_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
    y_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
    z_centerline = [iz for iz in range(min_z_index, max_z_index+1)]
    # Extract segmentation points and average per slice
    for iz in range(min_z_index, max_z_index+1):
        x_seg, y_seg = (data[:,:,iz]>0).nonzero()
        x_centerline[iz-min_z_index] = np.mean(x_seg)
        y_centerline[iz-min_z_index] = np.mean(y_seg)
    for k in range(len(X)):
	    data[X[k],Y[k],Z[k]] = 0
    # Fit the centerline points with splines and return the new fitted coordinates
    x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)


    # Create an image with the centerline
    for iz in range(min_z_index, max_z_index+1):
	    data[round(x_centerline_fit[iz-min_z_index]),round(y_centerline_fit[iz-min_z_index]),iz] = 1

    # Write the centerline image in RPI orientation
    hdr.set_data_dtype('uint8') # set imagetype to uint8
    print '\nWrite NIFTI volumes...'
    img = nibabel.Nifti1Image(data, None, hdr)
    nibabel.save(img, 'tmp.centerline.nii')
    sct.generate_output_file('tmp.centerline.nii','./',file_data+'_centerline',ext_data)

    del data

    # come back to parent folder
    os.chdir('..')

    # Change orientation of the output centerline into input orientation
    print '\nOrient centerline image to input orientation: ' + orientation
    fname_segmentation_orient = 'tmp.segmentation_rpi' + ext_data
    sct.run('sct_orientation -i ' + path_tmp+'/'+file_data+'_centerline'+ext_data + ' -o ' + file_data+'_centerline'+ext_data + ' -orientation ' + orientation)


    # Remove temporary files
    if remove_temp_files == 1 :
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp)


    # to view results
    print '\nTo view results, type:'
    print 'fslview '+file_data+'_centerline &\n'

# End of extract_centerline


# COMPUTE_CSA
# ==========================================================================================


def compute_CSA(fname_segmentation):
    
    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)
	
    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp)
    
    # copy files into tmp folder
    sct.run('cp '+fname_segmentation+' '+path_tmp)
    
    # go to tmp folder
    os.chdir(path_tmp)
    
    remove_temp_files = param.remove_temp_files
    step = param.step
    
    # # Change orientation of the input segmentation into RPI
    print '\nOrient segmentation image to RPI orientation...'
    fname_segmentation_orient = 'tmp.segmentation_rpi' + ext_data
    sct.run('sct_orientation -i ' + file_data+ext_data + ' -o ' + fname_segmentation_orient + ' -orientation RPI')
	
    # Get size of data
    print '\nGet dimensions data...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_segmentation_orient)
    print '.. '+str(nx)+' x '+str(ny)+' y '+str(nz)+' z '+str(nt)
	
    print '\nOpen segmentation volume...'
    file = nibabel.load(fname_segmentation_orient)
    data = file.get_data()
    hdr = file.get_header()
    
    x_scale=hdr['pixdim'][1]
    y_scale=hdr['pixdim'][2]
    z_scale=hdr['pixdim'][3]
    
    #
    # Extract min and max index in Z direction
    X, Y, Z = (data>0).nonzero()
    coords = np.array([str([X[i],Y[i],Z[i]]) for i in range(0,len(Z))]) #don't know why but finding strings in array of array of strings is WAY fater than doing the same with integers
    #coords = [[X[i],Y[i],Z[i]] for i in range(0,len(Z))]
    
    min_z_index, max_z_index = min(Z), max(Z)
    x_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
    y_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
    z_centerline = [iz for iz in range(min_z_index, max_z_index+1)]
    
    # Extract segmentation points and average per slice
    for iz in range(min_z_index, max_z_index+1):
        x_seg, y_seg = (data[:,:,iz]>0).nonzero()
        x_centerline[iz-min_z_index] = np.mean(x_seg)
        y_centerline[iz-min_z_index] = np.mean(y_seg)
    
	
    # Fit the centerline points with splines and return the new fitted coordinates
    x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)
    
    # fig=plt.figure()
    #    ax=Axes3D(fig)
    #    ax.plot(x_centerline,y_centerline,z_centerline,zdir='z')
    #    ax.plot(x_centerline_fit,y_centerline_fit,z_centerline,zdir='z')
    #    plt.show()
    
    # step = min([x_scale,y_scale])
    # print step
    x=np.array([1,0,0])
    y=np.array([0,1,0])
    z=np.array([0,0,1])
    
    
    print('\nComputing CSA...')
    sections=[0 for i in range(0,max_z_index-min_z_index+1)]
    
    for iz in range(0,len(z_centerline)):
        
        a = x_centerline_deriv[iz]
        b = y_centerline_deriv[iz]
        c = z_centerline_deriv[iz]
        x_center = x_centerline_fit[iz]
        y_center = y_centerline_fit[iz]
        z_center = z_centerline[iz]
        d = -(a*x_center+b*y_center+c*z_center)
        
        normal=normalize(np.array([a,b,c]))
        
        basis_1 = normalize(np.cross(normal,x)) # use of x in order to get orientation of each plane, basis_1 is in the plane ax+by+cz+d=0
        basis_2 = normalize(np.cross(normal,basis_1)) # third vector of base
        
        angle = np.arccos(np.dot(normal,z))
        max_diameter = (max([(max(X)-min(X))*x_scale,(max(Y)-min(Y))*y_scale])*np.sqrt(2))/(np.cos(angle)) # maximum dimension of the tilted plane
        
        plane = np.zeros((int(max_diameter/step),int(max_diameter/step)))  ## discretized plane which will be filled with 0/1
        plane_grid = np.linspace(-int(max_diameter/2),int(max_diameter/2),(max_diameter/step)) # how the plane will be skimmed through
        
        cpt=0
        
        for i_b1 in plane_grid :
            
            for i_b2 in plane_grid :    # we go through the plane
                
                point = np.array([x_center*x_scale,y_center*y_scale,z_center*z_scale]) + i_b1*basis_1 +i_b2*basis_2
                coord_voxel = str([ int(round(point[0]/x_scale)), int(round(point[1]/y_scale)), int(round(point[2]/z_scale))])  ## to which voxel belongs each point of the plane
                #coord_voxel = [ int(round(point[0]/x_scale)), int(round(point[1]/y_scale)), int(round(point[2]/z_scale))]  ## to which voxel belongs each point of the plane
                
                if (coord_voxel in coords) is True :  ## if this voxel is 1
                    
                    plane[i_b1+int(max_diameter/2)][i_b2+int(max_diameter/2)]=1
                    cpt = cpt+1
        
        
        
        sections[iz]=cpt*step*step  # number of voxels that are in the intersection of each plane and the nonzeros values of segmentation, times the area of one cell of the discretized plane
        
        print sections[iz]
    
    #os.chdir('..')
    #sct.run('mkdir JPG_Results')
    #os.chdir('JPG_Results')
    #imsave('plane_' + str(iz) + '.jpg', plane)     # if you want ot save the images with the sections
    #os.chdir('..')
    #os.chdir('path_tmp')
    
    #print sections
    
    
    ## plotting results
    
    fig=plt.figure()
    plt.plot(z_centerline*z_scale, sections)
    plt.show()
    
    
    # come back to parent folder
    os.chdir('..')
    
    # creating output text file
    print('\nGenerating output text file...')
    file = open('Cross_Area_Sections.txt','w')
    file.write('List of Cross Section Areas for each z slice\n')
    
    for i in range(min_z_index, max_z_index+1):
        file.write('\nz = ' + str(i*z_scale) + ' mm -> CSA = ' + str(sections[i]) + ' mm^2')

    file.close()

    # Remove temporary files
    if remove_temp_files == 1 :
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp)

# End of compute_CSA


#=======================================================================================================================
# B-Spline fitting
#=======================================================================================================================

def b_spline_centerline(x_centerline,y_centerline,z_centerline):
    
    print '\nFitting centerline using B-spline approximation...'
    points = [[x_centerline[n],y_centerline[n],z_centerline[n]] for n in range(len(x_centerline))]
    nurbs = NURBS(3,3000,points) # BE very careful with the spline order that you choose : if order is too high ( > 4 or 5) you need to set a higher number of Control Points (cf sct_nurbs ). For the third argument (number of points), give at least len(z_centerline)+500 or higher
    
    P = nurbs.getCourbe3D()
    x_centerline_fit=P[0]
    y_centerline_fit=P[1]
    Q = nurbs.getCourbe3D_deriv()
    x_centerline_deriv=Q[0]
    y_centerline_deriv=Q[1]
    z_centerline_deriv=Q[2]
    
    return x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv


def normalize(vect):
    """take an 1x3 matrix vector and return the normalised vector"""
    norm=np.linalg.norm(vect)
    return vect/norm

# def find_in_list(small_list,list_lists):
#     """find a list in a list of lists"""
#     test=0
#     cpt2=0
#     while (test==0) & (cpt2<len(coords)):                      ## find in list
#         if (int(coord_voxel[0])==int(coords[cpt2][0]))&(int(coord_voxel[1])==int(coords[cpt2][1]))&(int(coord_voxel[2])==int(coords[cpt2][2])):
#             cpt=cpt+1
#             test=1
#         cpt2=cpt2+1




# Print usage
# ==========================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Perform various types of processing from the spinal cord segmentation (e.g. extract centerline, compute CSA,' \
        ' etc.).\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <segmentation> -p <process>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -i <segmentation>          segmentation data\n' \
        '  -p <process>               process to perform {extract_centerline},{compute_CSA}\n' \
        '\n' \
        'OPTIONAL ARGUMENTS\n' \
        '  -v <0,1>                   verbose. Default='+str(param.verbose)+'.\n'
    
    # exit program
    sys.exit(2)

# START PROGRAM
# =========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()
