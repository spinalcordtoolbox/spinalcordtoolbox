#!/usr/bin/env python
#########################################################################################
#
# Module containing fitting functions 
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Touati
# Created: 2014-07-08
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from math import sqrt
import sys
try:
    import numpy as np
except ImportError:
    print '--- numpy not installed! ---'
    sys.exit(2)
#from sct_nurbs import NURBS


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

def evaluate_derivative_3D(x, y, z):

    x_deriv = np.array([(x[i+1]-x[i-1])/sqrt((x[i+1]-x[i-1])**2+(y[i+1]-y[i-1])**2+(z[i+1]-z[i-1])**2) for i in range (1,len(x)-1)])
    y_deriv = np.array([(y[i+1]-y[i-1])/sqrt((x[i+1]-x[i-1])**2+(y[i+1]-y[i-1])**2+(z[i+1]-z[i-1])**2) for i in range (1,len(y)-1)])
    z_deriv = np.array([(z[i+1]-z[i-1])/sqrt((x[i+1]-x[i-1])**2+(y[i+1]-y[i-1])**2+(z[i+1]-z[i-1])**2) for i in range (1,len(z)-1)])

    np.insert(x_deriv, 0, (x[1]-x[0])/sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2))
    np.append(x_deriv, (x[-1]-x[-2])/sqrt((x[-1]-x[-2])**2+(y[-1]-y[-2])**2+(z[-1]-z[-2])**2))

    np.insert(y_deriv, 0, (y[1]-y[0])/sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2))
    np.append(y_deriv, (y[-1]-y[-2])/sqrt((x[-1]-x[-2])**2+(y[-1]-y[-2])**2+(z[-1]-z[-2])**2))

    np.insert(z_deriv, 0, (z[1]-z[0])/sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2))
    np.append(z_deriv, (z[-1]-z[-2])/sqrt((x[-1]-x[-2])**2+(y[-1]-y[-2])**2+(z[-1]-z[-2])**2))

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

def b_spline_nurbs(x,y,z, control_points = 0, degree = 3,point_number = 3000):


    from sct_nurbs_v2 import NURBS
    #from sct_nurbs import NURBS
          
    print '\nFitting centerline using B-spline approximation...'
    data = [[x[n],y[n],z[n]] for n in range(len(x))]
    if control_points == 0:
        nurbs = NURBS(degree,point_number,data) # BE very careful with the spline order that you choose : if order is too high ( > 4 or 5) you need to set a higher number of Control Points (cf sct_nurbs ). For the third argument (number of points), give at least len(z_centerline)+500 or higher
    else:
        print 'In b_spline_nurbs we get control_point = ', control_points
        nurbs = NURBS(degree, point_number, data, False, control_points)



    P = nurbs.getCourbe3D()
    x_fit=P[0]
    y_fit=P[1]
    z_fit=P[2]
    Q = nurbs.getCourbe3D_deriv()
    x_deriv=Q[0]
    y_deriv=Q[1]
    z_deriv=Q[2]
  
    return x_fit, y_fit,z_fit,x_deriv,y_deriv,z_deriv



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






