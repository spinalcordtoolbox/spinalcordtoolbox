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
        
# check if needed Python libraries are already installed or not
import sys
import getopt
import sct_utils as sct
import nibabel
from numpy import linspace
import os
from msct_smooth import non_parametric, Univariate_Spline, polynomial_fit
from scipy import interpolate

def main():
    
    #Initialization
    fname = ''
    verbose = param.verbose
    output_name = param.output_name
         
    try:
         opts, args = getopt.getopt(sys.argv[1:],'hi:o:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts :
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            fname = arg         
        elif opt in ("-o"):
            output_name = arg    
        elif opt in ('-v'):
            verbose = int(arg)
    
    # display usage if a mandatory argument is not provided
    if fname == '' :
        usage()
    # check existence of input files
    print'\nCheck if file exists ...'
    
    sct.check_file_exist(fname)
    
    # Display arguments
    print'\nCheck input arguments...'
    print'  Input volume ...................... '+fname
    print'  Verbose ........................... '+str(verbose)

    file = nibabel.load(fname)
    data = file.get_data()
    hdr = file.get_header()
    
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
    
    f1 = interpolate.UnivariateSpline(Z,X, s=100)
    f2 = interpolate.UnivariateSpline(Z,Y,s=100)
  
    #f1 = polynomial_fit(Z,X,8)
    #f2 = polynomial_fit(Z,Y,8)
    
    
    #tckp,u = interpolate.splprep([X,Y,Z],s=1000,k=3)
    #xnew,ynew,znew = interpolate.splev(u,tckp)
    
    X_fit = f1(Z_new)
    Y_fit = f2(Z_new)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # plt.figure()
 #    plt.plot(znew,xnew)
 #    plt.plot(Z,X,'o',linestyle = 'None')
 #    plt.show()
 #
 #    plt.figure()
 #    plt.plot(znew,ynew)
 #    plt.plot(Z,Y,'o',linestyle = 'None')
 #    plt.show()
 #
 #    fig1 = plt.figure()
 #    ax = Axes3D(fig1)
 #    ax.plot(X,Y,Z,'o',linestyle = 'None',zdir='z')
 #    ax.plot(xnew,ynew,znew,zdir='z')
 #    plt.show()
 #
    if verbose==2 : 
        plt.figure()
        plt.plot(Z_new,X_fit)
        plt.plot(Z,X,'o',linestyle = 'None')
        plt.show()

        plt.figure()
        plt.plot(Z_new,Y_fit)
        plt.plot(Z,Y,'o',linestyle = 'None')
        plt.show()

        fig1 = plt.figure()
        ax = Axes3D(fig1)
        ax.plot(X,Y,Z,'o',linestyle = 'None',zdir='z')
        ax.plot(X_fit,Y_fit,Z_new,zdir='z')
        plt.show()

    
    data =data*0
    
    for i in xrange(len(X_fit)):
        data[X_fit[i],Y_fit[i],Z_new[i]] = 1
    
    
    print '\nSave volume ...'
    hdr.set_data_dtype('float32') # set imagetype to uint8
    # save volume
    #data = data.astype(float32, copy =False)
    img = nibabel.Nifti1Image(data, None, hdr)
    file_name = output_name
    nibabel.save(img,file_name)
    
    print '\nFile created : ' + output_name
    
    del data
    
    



    
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION

Generates 3D centerline from an input mask. 

USAGE
  """+os.path.basename(__file__)+"""  -i <input_volume> 

MANDATORY ARGUMENTS
  -i <input_volume>         input volume. No Default value
 
OPTIONAL ARGUMENTS
  -o <output_name>          name of the output. Default="""+str(param.output_name)+""" 
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






