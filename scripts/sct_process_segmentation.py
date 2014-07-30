#!/usr/bin/env python
#########################################################################################
#
# Perform various types of processing from the spinal cord segmentation (e.g. extract centerline, compute CSA, etc.).
# (extract_centerline) extract the spinal cord centerline from the segmentation. Output file is an image in the same
# space as the segmentation.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Touati, Gabriel Mangeat
# Modified: 2014-07-20 by jcohenadad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# N.B. To avoid confusion for the user, I removed from the menu the other options for computing CSA (jcohenadad 2014-07-20)

# TODO: the import of scipy.misc imsave was moved to the specific cases (orth and ellipse) in order to avoid issue #62. This has to be cleaned in the future.

# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug              = 0
        self.verbose            = 0 # verbose
        self.step               = 1 # step of discretized plane in mm default is min(x_scale,y_scale)
        self.remove_temp_files  = 1
        self.volume_output      = 0
        self.spline_smoothing   = 1
        self.smoothing_param    = 700
        self.figure_fit         = 0
        self.fname_csa = 'csa.txt'  # output name for txt CSA
        self.name_output = 'csa_volume.nii.gz'  # output name for slice CSA
        self.name_method = 'counting_z_plane'  # for compute_CSA
        
        
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
import scipy
#from numpy.linalg import eig, inv
#import Image
#from scipy.interpolate import splev, splrep
import nibabel

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
    processes = ['extract_centerline', 'compute_csa']
    method_CSA = ['counting_ortho_plane', 'counting_z_plane', 'ellipse_ortho_plane', 'ellipse_z_plane']
    name_method = param.name_method
    volume_output = param.volume_output
    verbose = param.verbose
    start_time = time.time()
    remove_temp_files = param.remove_temp_files
    spline_smoothing = param.spline_smoothing
    step = param.step
    smoothing_param = param.smoothing_param
    figure_fit = param.figure_fit
    name_output = param.name_output
    
    # Parameters for debug mode
    if param.debug:
        fname_segmentation = path_sct+'/testing/data/errsm_23/t2/t2_segmentation_PropSeg.nii.gz'
        verbose = 1
        remove_temp_files = 0
        from matplotlib.pyplot import imshow, gray, show
        from mpl_toolkits.mplot3d import Axes3D
        
    # Check input parameters
    try:
         opts, args = getopt.getopt(sys.argv[1:],'hi:p:m:b:r:s:f:o:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts :
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            fname_segmentation = arg
        elif opt in ("-p"):
            name_process = arg
        elif opt in("-m"):
            name_method = arg
        elif opt in('-b'):
            volume_output = int(arg)
        elif opt in('-r'):
            remove_temp_files = int(arg)
        elif opt in ('-s'):
            spline_smoothing = int(arg)
        elif opt in ('-f'):
            figure_fit = int(arg)
        elif opt in ('-o'):
            name_output = arg
        elif opt in ('-v'):
            verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_segmentation == '' or name_process == '':
        usage()

    # display usage if the requested process is not available
    if name_process not in processes:
        usage()

    # display usage if incorrect method
    if name_process == 'compute_csa' and (name_method not in method_CSA):
        usage()
    
    # display usage if no method provided
    if name_process=='compute_csa' and method_CSA == '':
        usage() 
        
    # check existence of input files
    sct.check_file_exist(fname_segmentation)
    
    # print arguments
    print '\nCheck parameters:'
    print '.. segmentation file:             '+fname_segmentation

    if name_process == 'extract_centerline':
        extract_centerline(fname_segmentation,remove_temp_files)

    if name_process == 'compute_csa' :
        compute_csa(fname_segmentation,name_method,volume_output,verbose,remove_temp_files,spline_smoothing,step,smoothing_param,figure_fit,name_output)
    

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s\n'

    # End of Main


# EXTRACT_CENTERLINE
# ==========================================================================================

def extract_centerline(fname_segmentation,remove_temp_files):
    # Extract path, file and extension
    path_data, file_data, ext_data = sct.extract_fname(fname_segmentation)

    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp)

    # copy files into tmp folder
    sct.run('cp '+fname_segmentation+' '+path_tmp)

    # go to tmp folder
    os.chdir(path_tmp)
            
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



# compute_csa
# ==========================================================================================
def compute_csa(fname_segmentation,name_method,volume_output,verbose,remove_temp_files,spline_smoothing,step,smoothing_param,figure_fit,name_output):

    # Extract path, file and extension
    path_data_seg, file_data_seg, ext_data_seg = sct.extract_fname(fname_segmentation)
    
    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp)
    
    # copy files into tmp folder
    sct.run('cp '+fname_segmentation+' '+path_tmp)
    
    # go to tmp folder
    os.chdir(path_tmp)
        
    # Change orientation of the input segmentation into RPI
    print '\nOrient segmentation image to RPI orientation...'
    fname_segmentation_orient = 'tmp.segmentation_rpi' + ext_data_seg
    sct.run('sct_orientation -i ' + file_data_seg + ext_data_seg + ' -o ' + fname_segmentation_orient + ' -orientation RPI')

    # Get size of data
    print '\nGet data dimensions...'
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_segmentation_orient)
    print '.. '+str(nx)+' x '+str(ny)+' x '+str(nz)+' x '+str(nt)

    print '\nOpen segmentation volume...'
    file_seg = nibabel.load(fname_segmentation_orient)
    data_seg = file_seg.get_data()
    hdr_seg = file_seg.get_header()
    
    # Get mm scales of the volume
    x_scale=hdr_seg['pixdim'][1]
    y_scale=hdr_seg['pixdim'][2]
    z_scale=hdr_seg['pixdim'][3]
     
    
    # Extract min and max index in Z direction
    X, Y, Z = (data_seg>0).nonzero()
    coords_seg = np.array([str([X[i],Y[i],Z[i]]) for i in xrange(0,len(Z))]) #don't know why but finding strings in array of array of strings is WAY faster than doing the same with integers        
    #coords_seg = [[X[i],Y[i],Z[i]] for i in range(0,len(Z))] #don't know why but finding strings in array of array of strings is WAY faster than doing the same with integers        
    
    min_z_index, max_z_index = min(Z), max(Z)
    Xp,Yp = (data_seg[:,:,0]>=0).nonzero() # X and Y range
   
    x_centerline = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    y_centerline = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    z_centerline = np.array([iz for iz in xrange(min_z_index, max_z_index+1)])
    
    # Extract segmentation points and average per slice
    for iz in xrange(min_z_index, max_z_index+1):
        x_seg, y_seg = (data_seg[:,:,iz]>0).nonzero()
        x_centerline[iz-min_z_index] = np.mean(x_seg)
        y_centerline[iz-min_z_index] = np.mean(y_seg)


 #    ### First Method  : counting voxel in orthogonal plane + fitting ellipse in orthogonal plane

    # Fit the centerline points with spline and return the new fitted coordinates
    x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)

   # # 3D plot of the fit
 #    fig=plt.figure()
 #    ax=Axes3D(fig)
 #    ax.plot(x_centerline,y_centerline,z_centerline,zdir='z')
 #    ax.plot(x_centerline_fit,y_centerline_fit,z_centerline,zdir='z')
 #    plt.show()

    # Defining cartesian basis vectors 
    x=np.array([1,0,0])
    y=np.array([0,1,0])
    z=np.array([0,0,1])
    
    # Creating folder in which JPG files will be stored
    sct.run('mkdir JPG_Results')

    # Computing CSA
    print('\nComputing CSA...')
    
    # Empty arrays in which CSA for each z slice will be stored
    csa = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    # sections_ortho_counting = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    # sections_ortho_ellipse = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    # sections_z_ellipse = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    # sections_z_counting = [0 for i in xrange(0,max_z_index-min_z_index+1)]
    
    for iz in xrange(0, len(z_centerline)):

            # Equation of the the plane which is orthogonal to the spline at z=iz
            a = x_centerline_deriv[iz]
            b = y_centerline_deriv[iz]
            c = z_centerline_deriv[iz]
            
            #vector normal to the plane
            normal = normalize(np.array([a,b,c]))
            
            # angle between normal vector and z
            angle = np.arccos(np.dot(normal,z))
            
            if name_method == 'counting_ortho_plane' or name_method == 'ellipse_ortho_plane':
                
                x_center = x_centerline_fit[iz]
                y_center = y_centerline_fit[iz]
                z_center = z_centerline[iz]
            
                # use of x in order to get orientation of each plane, basis_1 is in the plane ax+by+cz+d=0
                basis_1 = normalize(np.cross(normal,x))
                basis_2 = normalize(np.cross(normal,basis_1))
            
                # maximum dimension of the tilted plane. Try multiply numerator by sqrt(2) ?
                max_diameter = (max([(max(X)-min(X))*x_scale,(max(Y)-min(Y))*y_scale]))/(np.cos(angle)) 
                
                # Forcing the step to be the min of x and y scale (default value is 1 mm)
                step = min([x_scale,y_scale])
                
                # discretized plane which will be filled with 0/1
                plane_seg = np.zeros((int(max_diameter/step),int(max_diameter/step)))
            
                # how the plane will be skimmed through
                plane_grid = np.linspace(-int(max_diameter/2),int(max_diameter/2),int(max_diameter/step)) 
                
                # we go through the plane
                for i_b1 in plane_grid :
                    
                    for i_b2 in plane_grid : 
                        
                        point = np.array([x_center*x_scale,y_center*y_scale,z_center*z_scale]) + i_b1*basis_1 +i_b2*basis_2
                        
                        # to which voxel belongs each point of the plane
                        coord_voxel = str([ int(point[0]/x_scale), int(point[1]/y_scale), int(point[2]/z_scale)])
                        #coord_voxel = [ int(point[0]/x_scale), int(point[1]/y_scale), int(point[2]/z_scale)]
                        
                        if (coord_voxel in coords_seg) is True :  # if this voxel is 1
                        
                            plane_seg[int((plane_grid==i_b1).nonzero()[0])][int((plane_grid==i_b2).nonzero()[0])] = 1
                            
                            # number of voxels that are in the intersection of each plane and the nonzeros values of segmentation, times the area of one cell of the discretized plane
                            if name_method == 'counting_ortho_plane':
                                csa[iz] = len((plane_seg>0).nonzero()[0])*step*step
          
                if verbose ==1 and name_method == 'counting_ortho_plane' :
                    
                    print('Cross-Section Area : ' + str(csa[iz]) + ' mm^2')
            
                if name_method == 'ellipse_ortho_plane' : 
                    
                    # import scipy stuff
                    from scipy.misc import imsave
                    
                    os.chdir('JPG_Results')
                    imsave('plane_ortho_' + str(iz) + '.jpg', plane_seg)
                    
                    # Tresholded gradient image
                    mag = edge_detection('plane_ortho_' + str(iz) + '.jpg')
                    
                    #Coordinates of the contour
                    x_contour,y_contour = (mag>0).nonzero()
                    
                    x_contour = x_contour*step
                    y_contour = y_contour*step
                    
                    #Fitting an ellipse
                    fit = Ellipse_fit(x_contour,y_contour)
                    
                    # Semi-minor axis, semi-major axis
                    a_ellipse, b_ellipse = ellipse_dim(fit)
                    
                    #Section = pi*a*b
                    csa[iz] = a_ellipse*b_ellipse*np.pi
                    
                    if verbose == 1 and name_method == 'ellipse_ortho_plane':
                        print('Cross-Section Area : ' + str(csa[iz]) + ' mm^2')
                    os.chdir('..')
                    
            if name_method == 'counting_z_plane' or name_method == 'ellipse_z_plane':
                 
                 # getting the segmentation for each z plane
                 x_seg, y_seg = (data_seg[:,:,iz+min_z_index]>0).nonzero()
                 seg = [[x_seg[i],y_seg[i]] for i in range(0,len(x_seg))]
                 
                 plane = np.zeros((max(Xp),max(Yp)))
                 
                 for i in seg:
                     # filling the plane with 0 and 1 regarding to the segmentation
                     plane[i[0] - 1][i[1] - 1] = 1
                     
                 if name_method == 'counting_z_plane' :
                     csa[iz] = len((plane>0).nonzero()[0])*x_scale*y_scale*np.cos(angle)
                
                 if verbose == 1 and name_method == 'counting_z_plane':
                     print('Cross-Section Area : ' + str(csa[iz]) + ' mm^2')
                
                 if name_method == 'ellipse_z_plane':
                     
                     # import scipy stuff
                     from scipy.misc import imsave
                                          
                     os.chdir('JPG_Results')
                     imsave('plane_z_' + str(iz) + '.jpg', plane)     
                     
                     # Tresholded gradient image
                     mag = edge_detection('plane_z_' + str(iz) + '.jpg')
                     
                     x_contour,y_contour = (mag>0).nonzero()
                     
                     x_contour = x_contour*x_scale
                     y_contour = y_contour*y_scale
                     
                     # Fitting an ellipse
                     fit = Ellipse_fit(x_contour,y_contour)
                     a_ellipse, b_ellipse = ellipse_dim(fit)
                     csa[iz] = a_ellipse*b_ellipse*np.pi*np.cos(angle)
                     
                     if verbose == 1 and name_method == 'ellipse_z_plane':
                         print('Cross-Section Area : ' + str(csa[iz]) + ' mm^2')
                    
                     os.chdir('..')


    # come back to parent folder
    os.chdir('..')

    if spline_smoothing == 1 :
        print('\nSmoothing results with spline...')
        tck = scipy.interpolate.splrep((z_centerline*z_scale), csa, s=smoothing_param)
        csa_smooth = scipy.interpolate.splev((z_centerline*z_scale), tck)
        if figure_fit == 1:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot((z_centerline*z_scale),csa)
            plt.plot((z_centerline*z_scale),csa_smooth)
            plt.legend(['CSA values', 'Smoothed values'],2)
            plt.savefig('Spline_fit.png')
        csa = csa_smooth  # update variable

    # Create output text file
    print('\nGenerating output text file...')
    file_results = open(param.fname_csa,'w')
    for i in range(min_z_index, max_z_index+1):
        file_results.write(str(int(i*z_scale)) + ',' + str(csa[i-min_z_index])+'\n')
    file_results.close()
    print '.. File created: '+param.fname_csa

    # output volume of csa values
    if volume_output == 1:
        # Extract orientation of the input segmentation
        status,sct_orientation_output = sct.run('sct_orientation -i '+path_data_seg+file_data_seg+ext_data_seg + ' -get')
        orientation = sct_orientation_output[-3:]
        # loop across slices
        for iz in range(min_z_index,max_z_index+1):
            # retrieve seg pixels
            x_seg, y_seg = (data_seg[:,:,iz]>0).nonzero()
            seg = [[x_seg[i],y_seg[i]] for i in range(0,len(x_seg))]
            # loop across pixels in segmentation
            for i in seg :
                # replace value with csa value
                data_seg[i[0], i[1], iz] = csa[iz-min_z_index]
        # create header
        hdr_seg.set_data_dtype('uint8') # set imagetype to uint8
        # save volume
        print '\nWrite NIFTI volumes...'
        data_seg = data_seg.astype(np.float32, copy =False)
        img = nibabel.Nifti1Image(data_seg, None, hdr_seg)
        file_name = path_tmp+'/'+file_data_seg+'_CSA_slices_rpi'+ext_data_seg
        nibabel.save(img,file_name)
        print '.. File created:' + file_name
        # Change orientation of the output centerline into input orientation
        print '\nOrient  image to input orientation: '
        sct.run('sct_orientation -i '+path_tmp+'/'+file_data_seg+'_CSA_slices_rpi'+ext_data_seg + ' -o ' + name_output + ' -orientation ' + orientation)

    del data_seg

    # Remove temporary files
    if remove_temp_files == 1 : 
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp)
   
    # End of compute_csa


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
                          
#=======================================================================================================================
# Normalization
#=======================================================================================================================

def normalize(vect):
    norm=np.linalg.norm(vect)
    return vect/norm
    
#=======================================================================================================================
# Ellipse fitting for a set of data
#=======================================================================================================================
#http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
def Ellipse_fit(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2
    C[1,1] = -1
    E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

#=======================================================================================================================
# Getting a and b parameter for fitted ellipse
#=======================================================================================================================

def ellipse_dim(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])


#=======================================================================================================================
# Detect edges of an image
#=======================================================================================================================
def edge_detection(f):
    
    import Image
    
    #sigma = 1.0
    img = Image.open(f) #grayscale
    imgdata = np.array(img, dtype = float)
    G = imgdata
    #G = ndi.filters.gaussian_filter(imgdata, sigma)
    gradx = np.array(G, dtype = float)                        
    grady = np.array(G, dtype = float)
 
    mask_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
          
    mask_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
 
    width = img.size[1]
    height = img.size[0]
 
    for i in range(1, width-1):
        for j in range(1, height-1):
        
            px = np.sum(mask_x*G[(i-1):(i+1)+1,(j-1):(j+1)+1])
            py = np.sum(mask_y*G[(i-1):(i+1)+1,(j-1):(j+1)+1])
            gradx[i][j] = px
            grady[i][j] = py

    mag = scipy.hypot(gradx,grady)

    treshold = np.max(mag)*0.9

    for i in range(width):
        for j in range(height):
            if mag[i][j]>treshold:
                mag[i][j]=1
            else:
                mag[i][j] = 0
   
    return mag
    
    
# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  This function performs various types of processing from the spinal cord segmentation:

USAGE
  """+os.path.basename(__file__)+"""  -i <segmentation> -p <process>

MANDATORY ARGUMENTS
  -i <segmentation>         spinal cord segmentation (e.g., use sct_segmentation_propagation)
  -p <process>              type of process to be performed:
                            - extract_centerline: extract centerline as binay file from segmentation
                            - compute_csa: computes cross-sectional area by counting pixels in each
                              slice and then geometrically adjusting using centerline orientation.
                              Output is a text file with z (1st column) and CSA in mm^2 (2nd column)

OPTIONAL ARGUMENTS
  -s {0,1}                   smooth CSA values with spline. Default="""+str(param.spline_smoothing)+"""
  -b {0,1}                   outputs a volume in which each slice\'s value is equal to the CSA in
                             mm^2. Default="""+str(param.volume_output)+"""
  -o <output_name>           name of the output volume if -b 1. Default="""+str(param.name_output)+"""
  -r {0,1}                   remove temporary files. Default="""+str(param.remove_temp_files)+"""
  -v {0,1}                   verbose. Default="""+str(param.verbose)+"""
  -h                         help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i binary_segmentation.nii.gz -p compute_csa\n"""

    # exit program
    sys.exit(2)

# START PROGRAM
# =========================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()
