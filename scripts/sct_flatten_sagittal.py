#!/usr/bin/env python
#########################################################################################
#
# Flatten spinal cord in sagittal plane.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad
# Modified: 2014-06-02
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: check input param with -s flag

## Default parameters
class param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.interp = 'sinc'  # final interpolation
        self.deg_poly = 10  # maximum degree of polynomial function for fitting centerline.
        self.remove_temp_files = 1  # remove temporary files

# check if needed Python libraries are already installed or not
import os
import getopt
import sys
import commands
import sct_utils as sct
from sct_nurbs import NURBS
from sct_utils import fsloutput
import nibabel
import numpy


#=======================================================================================================================
# main
#=======================================================================================================================
def main():
    
    # Initialization
    fname_anat = ''
    fname_centerline = ''
    centerline_fitting = 'splines'
    remove_temp_files = param.remove_temp_files
    interp = param.interp
    degree_poly = param.deg_poly
    
    # extract path of the script
    path_script = os.path.dirname(__file__)+'/'
    
    # Parameters for debug mode
    if param.debug == 1:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        fname_anat = path_sct_data+'/t2/t2.nii.gz'
        fname_centerline = path_sct_data+'/t2/t2_seg.nii.gz'
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    
    # Check input param
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:c:r:d:f:s:')
    except getopt.GetoptError as err:
        print str(err)
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            fname_anat = arg
        elif opt in ('-c'):
            fname_centerline = arg
        elif opt in ('-r'):
            remove_temp_files = int(arg)
        elif opt in ('-d'):
            degree_poly = int(arg)
        elif opt in ('-f'):
            centerline_fitting = str(arg)
        elif opt in ('-s'):
            interp = str(arg)
    
    # display usage if a mandatory argument is not provided
    if fname_anat == '' or fname_centerline == '':
        usage()
    
    # check existence of input files
    sct.check_file_exist(fname_anat)
    sct.check_file_exist(fname_centerline)
    
    # extract path/file/extension
    path_anat, file_anat, ext_anat = sct.extract_fname(fname_anat)
    
    # Display arguments
    print '\nCheck input arguments...'
    print '  Input volume ...................... '+fname_anat
    print '  Centerline ........................ '+fname_centerline
    print ''
    
    # Get input image orientation
    status, output = sct.run('sct_orientation -i ' + fname_anat + ' -get')
    input_image_orientation = output[-3:]

    # Reorient input data into RL PA IS orientation
    sct.run('sct_orientation -i '+fname_anat+' -o tmp.anat_orient.nii -orientation RPI')
    sct.run('sct_orientation -i '+fname_centerline+' -o tmp.centerline_orient.nii -orientation RPI')

    # Open centerline
    #==========================================================================================
    print '\nGet dimensions of input centerline...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('tmp.centerline_orient.nii')
    print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
    print '.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm'
    
    print '\nOpen centerline volume...'
    file = nibabel.load('tmp.centerline_orient.nii')
    data = file.get_data()

    X, Y, Z = (data>0).nonzero()
    min_z_index, max_z_index = min(Z), max(Z)
    
    
    # loop across z and associate x,y coordinate with the point having maximum intensity
    x_centerline = [0 for iz in range(min_z_index, max_z_index+1, 1)]
    y_centerline = [0 for iz in range(min_z_index, max_z_index+1, 1)]
    z_centerline = [iz for iz in range(min_z_index, max_z_index+1, 1)]

    # Two possible scenario:
    # 1. the centerline is probabilistic: each slices contains voxels with the probability of containing the centerline [0:...:1]
    # We only take the maximum value of the image to aproximate the centerline.
    # 2. The centerline/segmentation image contains many pixels per slice with values {0,1}.
    # We take all the points and approximate the centerline on all these points.

    X, Y, Z = ((data<1)*(data>0)).nonzero() # X is empty if binary image
    if (len(X) > 0): # Scenario 1
        for iz in range(min_z_index, max_z_index+1, 1):
            x_centerline[iz-min_z_index], y_centerline[iz-min_z_index] = numpy.unravel_index(data[:,:,iz].argmax(), data[:,:,iz].shape)
    else: # Scenario 2
        for iz in range(min_z_index, max_z_index+1, 1):
            x_seg, y_seg = (data[:,:,iz]>0).nonzero()
            if len(x_seg) > 0:
                x_centerline[iz-min_z_index] = numpy.mean(x_seg)
                y_centerline[iz-min_z_index] = numpy.mean(y_seg)

    # TODO: find a way to do the previous loop with this, which is more neat:
    # [numpy.unravel_index(data[:,:,iz].argmax(), data[:,:,iz].shape) for iz in range(0,nz,1)]
    
    # clear variable
    del data
    
    # Fit the centerline points with the kind of curve given as argument of the script and return the new smoothed coordinates
    if centerline_fitting == 'splines':
        x_centerline_fit, y_centerline_fit = b_spline_centerline(x_centerline,y_centerline,z_centerline)
    elif centerline_fitting == 'polynome':
        x_centerline_fit, y_centerline_fit = polynome_centerline(x_centerline,y_centerline,z_centerline)

    #==========================================================================================
    # Split input volume
    print '\nSplit input volume...'
    sct.run(sct.fsloutput + 'fslsplit tmp.anat_orient.nii tmp.anat_z -z')
    file_anat_split = ['tmp.anat_z'+str(z).zfill(4) for z in range(0,nz,1)]

    # initialize variables
    file_mat_inv_cumul = ['tmp.mat_inv_cumul_z'+str(z).zfill(4) for z in range(0,nz,1)]
    z_init = min_z_index
    displacement_max_z_index = x_centerline_fit[z_init-min_z_index]-x_centerline_fit[max_z_index-min_z_index]

    # write centerline as text file
    print '\nGenerate fitted transformation matrices...'
    file_mat_inv_cumul_fit = ['tmp.mat_inv_cumul_fit_z'+str(z).zfill(4) for z in range(0,nz,1)]
    for iz in range(min_z_index, max_z_index+1, 1):
        # compute inverse cumulative fitted transformation matrix
        fid = open(file_mat_inv_cumul_fit[iz], 'w')
        if (x_centerline[iz-min_z_index] == 0 and y_centerline[iz-min_z_index] == 0):
            displacement = 0
        else:
            displacement = x_centerline_fit[z_init-min_z_index]-x_centerline_fit[iz-min_z_index]
        fid.write('%i %i %i %f\n' %(1, 0, 0, displacement) )
        fid.write('%i %i %i %f\n' %(0, 1, 0, 0) )
        fid.write('%i %i %i %i\n' %(0, 0, 1, 0) )
        fid.write('%i %i %i %i\n' %(0, 0, 0, 1) )
        fid.close()

    # we complete the displacement matrix in z direction
    for iz in range(0, min_z_index, 1):
        fid = open(file_mat_inv_cumul_fit[iz], 'w')
        fid.write('%i %i %i %f\n' %(1, 0, 0, 0) )
        fid.write('%i %i %i %f\n' %(0, 1, 0, 0) )
        fid.write('%i %i %i %i\n' %(0, 0, 1, 0) )
        fid.write('%i %i %i %i\n' %(0, 0, 0, 1) )
        fid.close()
    for iz in range(max_z_index+1, nz, 1):
        fid = open(file_mat_inv_cumul_fit[iz], 'w')
        fid.write('%i %i %i %f\n' %(1, 0, 0, displacement_max_z_index) )
        fid.write('%i %i %i %f\n' %(0, 1, 0, 0) )
        fid.write('%i %i %i %i\n' %(0, 0, 1, 0) )
        fid.write('%i %i %i %i\n' %(0, 0, 0, 1) )
        fid.close()

    # apply transformations to data
    print '\nApply fitted transformation matrices...'
    file_anat_split_fit = ['tmp.anat_orient_fit_z'+str(z).zfill(4) for z in range(0,nz,1)]
    for iz in range(0, nz, 1):
        # forward cumulative transformation to data
        sct.run(fsloutput+'flirt -in '+file_anat_split[iz]+' -ref '+file_anat_split[iz]+' -applyxfm -init '+file_mat_inv_cumul_fit[iz]+' -out '+file_anat_split_fit[iz]+' -interp '+interp)

    # Merge into 4D volume
    print '\nMerge into 4D volume...'
    sct.run(fsloutput+'fslmerge -z tmp.anat_orient_fit tmp.anat_orient_fit_z*')

    # Reorient data as it was before
    print '\nReorient data back into native orientation...'
    sct.run('sct_orientation -i tmp.anat_orient_fit.nii -o tmp.anat_orient_fit_reorient.nii -orientation '+input_image_orientation)

    # Generate output file (in current folder)
    print '\nGenerate output file (in current folder)...'
    sct.generate_output_file('tmp.anat_orient_fit_reorient.nii', file_anat+'_flatten'+ext_anat)

    # Delete temporary files
    if remove_temp_files == 1:
        print '\nDelete temporary files...'
        sct.run('rm -rf tmp.*')

    # to view results
    print '\nDone! To view results, type:'
    print 'fslview '+file_anat+ext_anat+' '+file_anat+'_flatten'+ext_anat+' &\n'


#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print 'USAGE: \n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Flatten the spinal cord in the sagittal plane (to make nice pictures).\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <source> -c <centerline>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -i <source>       input volume.\n' \
        '  -c                centerline.\n' \
        '\n'\
        'OPTIONAL ARGUMENTS\n' \
        '  -s {nearestneighbour, trilinear, sinc}       final interpolation. Default='+str(param.interp)+'\n' \
        '  -d <deg>          degree of fitting polynome. Default='+str(param.deg_poly)+'\n' \
        '  -r {0, 1}         remove temporary files. Default='+str(param.remove_temp_files)+'\n' \
        '  -h                help. Show this message.\n' \
        '\n'\
        'EXAMPLE:\n' \
        '  sct_flatten_sagittal -i t2.nii.gz -c centerline.nii.gz\n'
    sys.exit(2)

def b_spline_centerline(x_centerline,y_centerline,z_centerline):
    """Give a better fitting of the centerline than the method 'spline_centerline' using b-splines"""
    
    
    points = [[x_centerline[n],y_centerline[n],z_centerline[n]] for n in range(len(x_centerline))]
    
    nurbs = NURBS(3,len(z_centerline)*3,points) # for the third argument (number of points), give at least len(z_centerline)
    # (len(z_centerline)+500 or 1000 is ok)
    P = nurbs.getCourbe3D()
    x_centerline_fit=P[0]
    y_centerline_fit=P[1]
    
    return x_centerline_fit, y_centerline_fit



def polynome_centerline(x_centerline,y_centerline,z_centerline):
    """Fit polynomial function through centerline"""
    
    # Fit centerline in the Z-X plane using polynomial function
    print '\nFit centerline in the Z-X plane using polynomial function...'
    coeffsx = np.polyfit(z_centerline, x_centerline, deg=5)
    polyx = np.poly1d(coeffsx)
    x_centerline_fit = np.polyval(polyx, z_centerline)
    
    #Fit centerline in the Z-Y plane using polynomial function
    print '\nFit centerline in the Z-Y plane using polynomial function...'
    coeffsy = np.polyfit(z_centerline, y_centerline, deg=5)
    polyy = np.poly1d(coeffsy)
    y_centerline_fit = np.polyval(polyy, z_centerline)
    
    
    return x_centerline_fit,y_centerline_fit

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()
