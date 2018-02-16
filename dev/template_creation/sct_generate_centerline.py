#!/usr/bin/env python
#########################################################################################
#
# 
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Touati
# Created: 2014-08-11
#
# About the license: see the file LICENSE.TXT
#########################################################################################


#DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.verbose = 1
        self.output_name = 'generated_centerline.nii.gz'
        self.smoothness = 1
        
# check if needed Python libraries are already installed or not
import sys, os
import getopt

from nibabel import load, Nifti1Image, save
from numpy import linspace
from scipy import interpolate

import sct_utils as sct
#from msct_smooth import b_spline_nurbs
#non_parametric, Univariate_Spline, polynomial_fit, opt_f

def main():
    
    #Initialization
    fname = ''
    verbose = param.verbose
    output_name = param.output_name
    smoothness = param.smoothness
         
    try:
         opts, args = getopt.getopt(sys.argv[1:],'hi:o:s:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts :
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            fname = arg         
        elif opt in ("-o"):
            output_name = arg    
        elif opt in ("-s"):
            smoothness = arg    
        elif opt in ('-v'):
            verbose = int(arg)
    
    # display usage if a mandatory argument is not provided
    if fname == '' :
        usage()
    # check existence of input files
    print'\nCheck if file exists ...'
    
    sct.check_file_exist(fname)
    
    # check if RPI
    sct.check_if_rpi(fname)

    # Display arguments
    print'\nCheck input arguments...'
    print'  Input volume ...................... '+fname
    print'  Verbose ........................... '+str(verbose)

    file = load(fname)
    data = file.get_data()
    hdr = file.get_header()
    curdir = os.getcwd()
    
    X,Y,Z = (data>0).nonzero()
    Z_new = linspace(min(Z),max(Z),(max(Z)-min(Z)+1))
    
    # tck1 = interpolate.splrep(Z, X, s=200)
  #   X_fit = interpolate.splev(Z_new, tck1)
  #
  #   tck2 = interpolate.splrep(Z, Y, s=200)
  #   Y_fit = interpolate.splev(Z_new, tck2)

    # f1 = interpolate.interp1d(Z, X, kind='cubic')
 #    f2 = interpolate.interp1d(Z,Y, kind='cubic')
 #

    # sort X and Y arrays using Z
    X = [X[i] for i in Z[:].argsort()]
    Y = [Y[i] for i in Z[:].argsort()]
    Z = [Z[i] for i in Z[:].argsort()]

    print X, Y, Z

    # NURBS!
    #X_fit, Y_fit, Z_fit, x_deriv, y_deriv, z_deriv = b_spline_nurbs(X, Y, Z, degree=3, point_number=3000, path_qc=curdir)

    #f_opt_x, f_opt_y = opt_f(X,Y,Z)
    #print "f_opt = "+str(f_opt_x)+" "+str(f_opt_y)
    #f1 = non_parametric(Z,X,f=0.8)
    #f2 = non_parametric(Z,Y,f=0.8)

    f1 = interpolate.UnivariateSpline(Z, X)
    f2 = interpolate.UnivariateSpline(Z, Y)

    #f1 = polynomial_fit(Z,X,smoothness)
    #f2 = polynomial_fit(Z,Y,smoothness)
    
    X_fit = f1(Z_new)
    Y_fit = f2(Z_new)

    print X_fit
    print Y_fit

    if verbose==2 :
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(Z_new,X_fit)
        plt.plot(Z,X,'o',linestyle = 'None')
        plt.show()

        plt.figure()
        plt.plot(Z_new,Y_fit)
        plt.plot(Z,Y,'o',linestyle = 'None')
        plt.show()
    
    data =data*0
    
    for i in xrange(len(X_fit)):
        data[X_fit[i],Y_fit[i],Z_new[i]] = 1
    
    
    print '\nSave volume ...'
    hdr.set_data_dtype('float32') # set imagetype to uint8
    # save volume
    #data = data.astype(float32, copy =False)
    img = Nifti1Image(data, None, hdr)
    file_name = output_name
    save(img,file_name)
    
    print '\nFile created : ' + output_name
    
    del data
    

    
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION

Generates 3D centerline from an input mask. The mask contains points along the spinal cord and this
function interpolates the points using 3D spline. Output is a binary volume. 

USAGE
  """+os.path.basename(__file__)+"""  -i <input_volume> 

MANDATORY ARGUMENTS
  -i <input_volume>         input volume. No Default value
 
OPTIONAL ARGUMENTS
  -o <output_name>          name of the output. Default="""+str(param.output_name)+""" 
  -s <smoothness>           smoothness of spline. Default="""+str(param.smoothness)+""" 
  -v {0,1,2}                verbose.Verbose 2 for plotting. Default="""+str(param.verbose)+"""
  -h                        help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i volume.nii.gz\n"""

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






