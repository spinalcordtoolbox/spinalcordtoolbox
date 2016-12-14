
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d
import os
import numpy as np
from scipy import ndimage
from numpy import linalg
import nibabel
import splines_approximation_v2 as splinap
from sklearn.metrics import mean_squared_error


def returnSquaredErrors(fname=None, div=None, size=None):
    if fname == None:
        #fname = 'centerlines/t250_half_sup_straight_seg_centerline.nii.gz'
        fname = 't234_crop_200_500_straight_centerline.nii.gz'
    file = nibabel.load(fname)
    data = file.get_data()
    p1, p2, p3 = splinap.getPxDimensions(fname)
    nx, ny, nz = splinap.getDim(fname)
    #os.remove(fname)



    print 'sizes: ', nx, ny, nx
    #x, y, x_fit, y_fit = fit(data, nz)

    '''
    print x, y, z
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    ax.plot(x,y,z,'b:',zdir='z')
    plt.show()
    '''

    X,Y,Z = (data>0).nonzero()

    zmin = min(Z) + 10
    zmax = max(Z) - 10

    print "min value element : ", zmin
    print "max value element : ", zmax

    nz = zmax - zmin

    x = [0 for iz in range(0, nz, 1)]
    y = [0 for iz in range(0, nz, 1)]
    z = [iz for iz in range(0, nz, 1)]


    for iz in range(0, nz, 1):
        x[iz], y[iz] = ndimage.measurements.center_of_mass(np.array(data[:,:,iz + zmin]))


    x_fit,y_fit = get_good_centerline(nx, ny, nz)


    msx = mean_squared_error(x, x_fit)
    msy = mean_squared_error(y, y_fit)
    mean_ms = (msx+msy)/2

    #print x, x_fit
    #print y, y_fit

    fig1 = plt.figure()
    ax = Axes3D(fig1)
    ax.plot(x,y,z,zdir='z')
    ax.plot(x_fit,y_fit,z,zdir='z')
    plt.show()


    print msx, msy, mean_ms
    size_crop = splinap.getSize(x, y, z, p1, p2, p3)


    txt = open('results.txt', 'a')
    txt.write(str(div)+' '+str(int(round(size))/div)+' '+str(size_crop)+' '+str(size)+' '+str(msx)+' '+str(msy)+' '+' '+str(mean_ms)+' '+fname+'\n')
    #txt.write(' plynomial_fitting '+str(size_crop)+' '+str(size)+' '+str(msx)+' '+str(msy)+' '+' '+str(mean_ms)+' '+fname+'\n')


    txt.close()

    #splinap.plot(x_fit,y_fit, z, x, y, z, fname, int(round(size))/div)







def get_good_centerline (nx, ny,nz):
    x = [nx/2 for iz in range(0, nz, 1)]
    y = [ny/2 for iz in range(0, nz, 1)]
    return x,y


def fit(data, nz):
    x = [0 for iz in range(0, nz, 1)]
    y = [0 for iz in range(0, nz, 1)]
    z = [iz for iz in range(0, nz, 1)]

    for iz in range(0, nz, 1):
        x[iz], y[iz] = ndimage.measurements.center_of_mass(np.array(data[:,:,iz]))

    #Fit centerline in the Z-X plane using polynomial function
    print '\nFit centerline in the Z-X plane using polynomial function...'
    coeffsx = np.polyfit(z, x, 1)
    polyx = np.poly1d(coeffsx)
    x_fit = np.polyval(polyx, z)
    print 'x_fit'
    print x_fit

    #Fit centerline in the Z-Y plane using polynomial function
    print '\nFit centerline in the Z-Y plane using polynomial function...'
    coeffsy = np.polyfit(z, y, 1)
    polyy = np.poly1d(coeffsy)
    y_fit = np.polyval(polyy, z)


    #### 3D plot
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    ax.plot(x,y,z,zdir='z')
    ax.plot(x_fit,y_fit,z,zdir='z')
    plt.show()
    return x, y, x_fit, y_fit

 	#return x_centerline_fit,y_centerline_fit,z_centerline_fit


if __name__ == "__main__":
    returnSquaredErrors()