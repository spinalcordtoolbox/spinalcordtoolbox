#!/usr/bin/env python
#########################################################################################
#
# Module containing fitting functions 
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Touati
# Created: 2014-10-08
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from math import sqrt
from scipy.interpolate import splrep,splev
import sys
try:
    import numpy as np
except ImportError:
    print '--- numpy not installed! ---'
    sys.exit(2)
#from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#from sct_nurbs import NURBS


#=======================================================================================================================
# Over pad the input file, smooth and return the centerline
#=======================================================================================================================
def smooth(fname, padding):
    sct.run('sct_c3d '+fname+' -pad '+str(padding)+'x'+str(padding)+'x'+str(padding)+'vox '+str(padding)+'x'+str(padding)+'x'+str(padding)+'vox 0 -o tmp.centerline_pad.nii.gz')



#=======================================================================================================================
# Spline 2D using splrep & splev
#=======================================================================================================================
def spline_2D(z_centerline, x_centerline):

    m = np.mean(x_centerline)
    sigma = np.std(x_centerline)
    print (m - np.sqrt(2*m))*(sigma**2), (m + np.sqrt(2*m))*(sigma**2)

    smoothing_param = (((m + np.sqrt(2*m))*(sigma**2))+((m - np.sqrt(2*m))*(sigma**2)))/2
    print('\nSmoothing results with spline...')
    tck = splrep(z_centerline, x_centerline, s=smoothing_param)
    x_centerline_fit = splev(z_centerline, tck)
    return x_centerline_fit

    # plt.figure()
    # plt.plot(z_centerline,means)
    # plt.plot(z_centerline,means_smooth)
    # plt.show()


#=======================================================================================================================
# Polynomial fit
#=======================================================================================================================
def polynomial_fit(x,y,degree):
    
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    y_fit = np.polyval(poly, x)
 
    return y_fit,poly


#=======================================================================================================================
# Polynomial derivative
#=======================================================================================================================   
def polynomial_deriv(x,poly):
    
    poly_deriv = np.polyder(poly, m = 1)
    y_fit_deriv = np.polyval(poly_deriv, x)
    
    return y_fit_deriv,poly_deriv


def norm(x, y, z, p1, p2, p3):
    s = 0
    for i in xrange (len(x)-1):
        s += sqrt((p1*(x[i+1]-x[i]))**2+(p2*(y[i+1]-y[i]))**2+(p3*(z[i+1]-z[i])**2))
    print "centerline size: ", s
    return s


#=======================================================================================================================
# Evaluate derivative of data points
#=======================================================================================================================
def evaluate_derivative_2D(x,y):
        
    y_deriv = np.array([(y[i+1]-y[i])/(x[i+1]-x[i]) for i in range (0,len(x)-1)])
    y_deriv = np.append(y_deriv,(y[-1] - y[-2])/(x[-1] - x[-2]))
    
    return y_deriv


def evaluate_derivative_3D(x, y, z, px, py, pz):

    x = [x_elem*px for x_elem in x]
    y = [y_elem*py for y_elem in y]
    z = [z_elem*pz for z_elem in z]

    x_deriv = np.array([(x[i+1]-x[i-1])/sqrt((x[i+1]-x[i-1])**2+(y[i+1]-y[i-1])**2+(z[i+1]-z[i-1])**2) for i in range (1,len(x)-1)])
    y_deriv = np.array([(y[i+1]-y[i-1])/sqrt((x[i+1]-x[i-1])**2+(y[i+1]-y[i-1])**2+(z[i+1]-z[i-1])**2) for i in range (1,len(y)-1)])
    z_deriv = np.array([(z[i+1]-z[i-1])/sqrt((x[i+1]-x[i-1])**2+(y[i+1]-y[i-1])**2+(z[i+1]-z[i-1])**2) for i in range (1,len(z)-1)])

    x_deriv = np.insert(x_deriv, 0, (x[1]-x[0])/sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2))
    x_deriv = np.append(x_deriv, (x[-1]-x[-2])/sqrt((x[-1]-x[-2])**2+(y[-1]-y[-2])**2+(z[-1]-z[-2])**2))

    #print len(x_deriv)

    y_deriv = np.insert(y_deriv, 0, (y[1]-y[0])/sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2))
    y_deriv = np.append(y_deriv, (y[-1]-y[-2])/sqrt((x[-1]-x[-2])**2+(y[-1]-y[-2])**2+(z[-1]-z[-2])**2))

    z_deriv = np.insert(z_deriv, 0, (z[1]-z[0])/sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2))
    z_deriv = np.append(z_deriv, (z[-1]-z[-2])/sqrt((x[-1]-x[-2])**2+(y[-1]-y[-2])**2+(z[-1]-z[-2])**2))

    return x_deriv, y_deriv, z_deriv


#=======================================================================================================================
# Non parametric regression
#=======================================================================================================================
def non_parametric(x,y,f = 0.25,iter = 3):
    """lowess(x, y, f=2./3., iter=3) -> yest

    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.

    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    https://gist.github.com/agramfort/850437 """
    from math import ceil
    from scipy import linalg
    
    n = len(x)
    r = int(ceil(f*n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:,None] - x[None,:]) / h), 0.0, 1.0)
    w = (1 - w**3)**3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:,i]
            b = np.array([np.sum(weights*y), np.sum(weights*y*x)])
            A = np.array([[np.sum(weights), np.sum(weights*x)],
                   [np.sum(weights*x), np.sum(weights*x*x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1]*x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta**2)**2

    return yest


def opt_f(x, y, z):
    print 'optimizing f parameter in non-parametric...'
    f_list = [0.1, 0.15, 0.20, 0.22, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5]
    msx_min = 2
    msy_min = 2
    f_opt_y = 5
    f_opt_x = 5
    for f in f_list:
        try:
            x_fit = non_parametric(z, x, f)
            y_fit = non_parametric(z, y, f)

            #msx = mean_squared_error(x, x_fit)
            #msy = mean_squared_error(y, y_fit)

            msex = mean_squared_error(x, x_fit)
            msey = mean_squared_error(y, y_fit)

            #print msx, msex, f
            #print msy, msey, f

            if msx < msx_min:
                msx_min = msx
                f_opt_x = f
            if msy < msy_min:
                msy_min = msy
                f_opt_y = f

            x_fit_d, y_fit_d, z_d = evaluate_derivative_3D(x_fit, y_fit, z)
            x_fit_dd, y_fit_dd, z_dd = evaluate_derivative_3D(x_fit_d, y_fit_d, z_d)
            amp_xd = np.max(abs(x_fit_dd))
            amp_yd = np.max(abs(y_fit_dd))
            mean_xd = np.mean(x_fit_dd)
            mean_yd = np.mean(y_fit_dd)
            mean = mean_xd + mean_yd

            # ax = plt.subplot(1,2,1)
            # plt.plot(z, x_fit, 'b-', label='centerline')
            # plt.plot(z, x_fit_d, 'r-', label='deriv')
            # plt.plot(z, x_fit_dd, 'y-', label='derivsec')
            # plt.xlabel('x')
            # plt.ylabel('z')
            # ax = plt.subplot(1,2,2)
            # plt.plot(z, y_fit, 'b-', label='centerline')
            # plt.plot(z, y_fit_d, 'r-', label='deriv')
            # plt.plot(z, y_fit_dd, 'r-', label='fit')
            # plt.xlabel('y')
            # plt.ylabel('z')
            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles, labels)
            # plt.show()

            print 'AMP', amp_xd, amp_yd
            print 'MEAN', mean_xd, mean_yd, mean

        except np.linalg.linalg.LinAlgError:
            print 'LinAlgError raised'

    print msx_min, f_opt_x
    print msy_min, f_opt_y
    return f_opt_x, f_opt_y


#=======================================================================================================================
# Univariate Spline fitting
#=======================================================================================================================
def Univariate_Spline(x, y, w=None, bbox=[None, None], k=3, s=None) :
    
    from scipy.interpolate import UnivariateSpline
    s = UnivariateSpline(x, y, w, bbox, k, s)
    ys = s(x)
    
    return ys    
    

#=======================================================================================================================
# 3D B-Spline function, sct_nurbs
#=======================================================================================================================
#def b_spline_nurbs(x, y, z, control_points=0, degree=3,point_number=3000):

def b_spline_nurbs(x, y, z, fname_centerline, degree=3,point_number=3000):

    #from sct_nurbs_v2 import NURBS
    from sct_nurbs import NURBS
          
    print '\nFitting centerline using B-spline approximation...'
    data = [[x[n], y[n], z[n]] for n in range(len(x))]

    # if control_points == 0:
    #     nurbs = NURBS(degree, point_number, data) # BE very careful with the spline order that you choose : if order is too high ( > 4 or 5) you need to set a higher number of Control Points (cf sct_nurbs ). For the third argument (number of points), give at least len(z_centerline)+500 or higher
    # else:
    #     print 'In b_spline_nurbs we get control_point = ', control_points
    #     nurbs = NURBS(degree, point_number, data, False, control_points)

    import math
    centerlineSize = getSize(x, y, z, fname_centerline)
    nbControl = 30*math.log(centerlineSize, 10) - 42
    nbControl = round(nbControl)
    nurbs = NURBS(degree, point_number, data, False, nbControl)

    P = nurbs.getCourbe3D()
    x_fit=P[0]
    y_fit=P[1]
    z_fit=P[2]
    Q = nurbs.getCourbe3D_deriv()
    x_deriv=Q[0]
    y_deriv=Q[1]
    z_deriv=Q[2]
  
    return x_fit, y_fit,z_fit,x_deriv,y_deriv,z_deriv


def getSize(x, y, z, file_name):
    # get pixdim
    import commands
    cmd1 = 'fslval '+file_name+' pixdim1'
    status, output = commands.getstatusoutput(cmd1)
    p1 = float(output)
    cmd2 = 'fslval '+file_name+' pixdim2'
    status, output = commands.getstatusoutput(cmd2)
    p2 = float(output)
    cmd3 = 'fslval '+file_name+' pixdim3'
    status, output = commands.getstatusoutput(cmd3)
    p3 = float(output)

    # Centerline size
    s = 0
    for i in xrange (len(x)-1):
        s += sqrt((p1*(x[i+1]-x[i]))**2+(p2*(y[i+1]-y[i]))**2+(p3*(z[i+1]-z[i])**2))
    #print "centerline size: ", s
    return s


#=======================================================================================================================
# functions to get ceterline size
#=======================================================================================================================
def getPxDimensions(file_name):
    import commands
    cmd1 = 'fslval '+file_name+' pixdim1'
    status, output = commands.getstatusoutput(cmd1)
    p1 = float(output)
    cmd2 = 'fslval '+file_name+' pixdim2'
    status, output = commands.getstatusoutput(cmd2)
    p2 = float(output)
    cmd3 = 'fslval '+file_name+' pixdim3'
    status, output = commands.getstatusoutput(cmd3)
    p3 = float(output)
    return p1, p2, p3


#def getSize(x, y, z, p1, p2, p3):
#    s = 0
#    for i in xrange (len(x)-1):
#        s += sqrt((p1*(x[i+1]-x[i]))**2+(p2*(y[i+1]-y[i]))**2+(p3*(z[i+1]-z[i])**2))
#    print "centerline size: ", s
#    return s



#=======================================================================================================================
# 3D B-Spline function, python function
#=======================================================================================================================
def b_spline_python(x, y, z, s = 0, k = 3, nest = -1):
    """see http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html for full input information"""
    from scipy.interpolate import splprep,splev
    
    tckp,u = splprep([x,y,z], s = s, k = k, nest = nest)

    xnew,ynew,znew = splev(u,tckp)
    
    return xnew, ynew, znew 


#=======================================================================================================================
# lowpass filter  
#=======================================================================================================================
def lowpass (y) :
    from scipy.fftpack import fftfreq, fft
    from scipy.signal import filtfilt, iirfilter
  
    frequency = fftfreq(len(y))
    spectrum = np.abs(fft(y, n=None, axis=-1, overwrite_x=False))
    Wn = np.amax(frequency)/10
    N = 5              #Order of the filter
    b, a = iirfilter(N, Wn, rp=None, rs=None, btype='low', analog=False, ftype='butter', output='ba')
    y_smooth = filtfilt(b, a, y, axis=-1, padtype=None)


    return y_smooth
    
#=======================================================================================================================
# moving_average
#=======================================================================================================================   
def moving_average(y, n=3) :
    y_smooth = np.cumsum(y, dtype=float)
    y_smooth[n:] = y_smooth[n:] - y_smooth[:-n]
    
    return y_smooth[n - 1:] / n


#=======================================================================================================================
# moving_average
#=======================================================================================================================
def mean_squared_error(x, x_fit):
    mse = 0
    if len(x_fit) == len(x) and len(x) is not 0:
        n = len(x)
        for i in range(0,len(x)):
            mse += (x[i]-x_fit[i])*(x[i]-x_fit[i])
        mse = float(mse)
        mse *= (1/float(n))
        return mse
    else:
        print "cannot calculate the mean squared error, check if the argument have the same length. \n"