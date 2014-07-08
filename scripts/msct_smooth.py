#!/usr/bin/env python
#########################################################################################
#
# Module containing 2D fitting functions 
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Touati
# Created: 2014-07-08
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
try:
    import numpy as np
except ImportError:
    print '--- numpy not installed! ---'
    sys.exit(2)
    

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
    
#=======================================================================================================================
# Evaluate derivative of data points
#=======================================================================================================================
def evaluate_derivative(x,y):
        
    y_deriv = np.array([(y[i+1]-y[i])/(x[i+1]-x[i]) for i in range (0,len(x)-1)])
    y_deriv = np.insert(y_deriv,-1,(y[-1] - y[-2])/(x[-1] - x[-2]))
    
    return y_deriv 

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
    
    
    
    