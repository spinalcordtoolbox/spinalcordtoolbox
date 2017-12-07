#!/home/django/jtouati/Library/Enthought/Canopy_64bit/User/bin/python
#Author : JULIEN TOUATI
#reworked and updated by Augustin Roux

import sct_utils
import sys
import os
import numpy
from math import sqrt
from scipy import ndimage

import nibabel
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sct_nurbs import *


def main(file_name, div, curve = 0):
    print file_name
    
    file = nibabel.load(file_name)
    data = file.get_data()
    
    nx, ny, nz = getDim(file_name)
    
    x = [0 for iz in range(0, nz, 1)]
    y = [0 for iz in range(0, nz, 1)]
    z = [iz for iz in range(0, nz, 1)]
    
    for iz in range(0, nz, 1):
            x[iz], y[iz] = ndimage.measurements.center_of_mass(numpy.array(data[:,:,iz]))
            
    p1, p2, p3 = getPxDimensions(file_name)

    size = getSize(x, y, z, p1, p2, p3)       
   
    points = [[x[n],y[n],z[n]] for n in range(len(x))]
    size = size
    nurbs = NURBS(int(size),int(div),3,3000,points)
    P = nurbs.getCourbe3D()
    if P==1:
            print "ERROR: instability in NURBS computation, div will be incremented. "
            return 1

    x_centerline_fit=P[0]
    y_centerline_fit=P[1]
    z_centerline_fit=P[2]
    
    plot(x_centerline_fit, y_centerline_fit, z_centerline_fit, x, y, z)

    #### 3D plot
    if curve:
        fig1 = plt.figure()
        ax = Axes3D(fig1)
        ax.plot(x,y,z,zdir='z')
        ax.plot(x_centerline_fit,y_centerline_fit,z_centerline_fit,zdir='z')
        plt.show()
    
    save(file_name, div, size)


def getDim(file_name):
    cmd = 'fslsize '+file_name
    status, output = sct.run(cmd)
    # split output according to \n field
    output_split = output.split()
    # extract dimensions as integer
    nx = int(output_split[1])
    ny = int(output_split[3])
    nz = int(output_split[5])

    return nx, ny, nz

def getPxDimensions(file_name):
    cmd1 = 'fslval '+file_name+' pixdim1'
    status, output = sct.run(cmd1)
    p1 = float(output)
    cmd2 = 'fslval '+file_name+' pixdim2'
    status, output = sct.run(cmd2)
    p2 = float(output)
    cmd3 = 'fslval '+file_name+' pixdim3'
    status, output = sct.run(cmd3)
    p3 = float(output)
    return p1, p2, p3

    
def plot(x_centerline_fit, y_centerline_fit, z_centerline_fit, x, y, z, fname, nbptctl):
    fig=plt.figure()
    plt.subplot(2,2,1)
    plt.plot(x_centerline_fit,y_centerline_fit,'r-')
    plt.plot(x,y,'b:')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(2,2,2)
    #plt.close()
    plt.plot(x_centerline_fit,z_centerline_fit,'r-')
    plt.plot(x,z,'b:')
    plt.axis([55.0,57.0,0.0,140.0])
    plt.xlabel('x')
    plt.ylabel('z')

    plt.subplot(2,2,3)
    plt.plot(y_centerline_fit,z_centerline_fit,'r-')
    plt.plot(y,z,'b:')
    plt.axis([221.0,225.5,0.0,140.0])
    plt.xlabel('y')
    plt.ylabel('z')

    #fig.close()

    path, file_name, ext_fname = sct_utils.extract_fname(fname)

    plt.savefig('./curve_'+file_name+'_'+str(nbptctl),dpi=267)


def save(file_name, div, size):   
    path, file_name, ext_fname = sct_utils.extract_fname(file_name)
    nbpt = int(size/int(div))
    plt.savefig('../'+file_name+'_'+str(div)+'_'+str(int(size))+'_'+str(nbpt),dpi=267)
    plt.close()
 
 
def getSize(x, y, z, p1, p2, p3):
    s = 0
    for i in xrange (len(x)-1):
        s += sqrt((p1*(x[i+1]-x[i]))**2+(p2*(y[i+1]-y[i]))**2+(p3*(z[i+1]-z[i])**2))
    print "centerline size: ", s
    return s

    
if __name__ == "__main__":
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    print 'Argument List:', str(sys.argv)
    file_name = sys.argv[1]
    div = sys.argv[2]
    print file_name
    main(file_name, div)