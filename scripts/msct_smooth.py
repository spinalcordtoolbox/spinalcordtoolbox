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

from scipy.interpolate import splrep, splev


#=======================================================================================================================
# Over pad the input file, smooth and return the centerline
#=======================================================================================================================
def smooth(fname, padding):
    sct.run('isct_c3d '+fname+' -pad '+str(padding)+'x'+str(padding)+'x'+str(padding)+'vox '+str(padding)+'x'+str(padding)+'x'+str(padding)+'vox 0 -o tmp.centerline_pad.nii.gz')



#=======================================================================================================================
# Spline 2D using splrep & splev
#=======================================================================================================================
def spline_2D(z_centerline, x_centerline):

    from numpy import mean, std, sqrt

    m = mean(x_centerline)
    sigma = std(x_centerline)
    print (m - sqrt(2*m))*(sigma**2), (m + sqrt(2*m))*(sigma**2)

    smoothing_param = (((m + sqrt(2*m))*(sigma**2))+((m - sqrt(2*m))*(sigma**2)))/2
    print('\nSmoothing results with spline...')
    tck = splrep(z_centerline, x_centerline, s = smoothing_param)
    x_centerline_fit = splev(z_centerline, tck)
    return x_centerline_fit



#=======================================================================================================================
# Polynomial fit
#=======================================================================================================================
def polynomial_fit(x,y,degree):

    import numpy as np

    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    y_fit = np.polyval(poly, x)
 
    return y_fit, poly


#=======================================================================================================================
# Polynomial derivative
#=======================================================================================================================   
def polynomial_deriv(x,poly):

    from numpy import polyder, polyval

    poly_deriv = polyder(poly, m = 1)
    y_fit_deriv = polyval(poly_deriv, x)
    
    return y_fit_deriv, poly_deriv



#=======================================================================================================================
# Get norm
#=======================================================================================================================
def norm(x, y, z, p1, p2, p3):
    from math import sqrt
    s = 0
    for i in xrange (len(x)-1):
        s += sqrt((p1*(x[i+1]-x[i]))**2+(p2*(y[i+1]-y[i]))**2+(p3*(z[i+1]-z[i])**2))
    print "centerline size: ", s
    return s



#=======================================================================================================================
# Evaluate derivative of data points
#=======================================================================================================================
def evaluate_derivative_2D(x,y):
    from numpy import array, append
    y_deriv = array([(y[i+1]-y[i])/(x[i+1]-x[i]) for i in range(0, len(x)-1)])
    y_deriv = append(y_deriv,(y[-1] - y[-2])/(x[-1] - x[-2]))
    return y_deriv



#=======================================================================================================================
# Evaluate derivative of data points in 3D
#=======================================================================================================================
def evaluate_derivative_3D(x, y, z, px, py, pz):
    from numpy import array, sqrt, insert, append

    x = [x_elem*px for x_elem in x]
    y = [y_elem*py for y_elem in y]
    z = [z_elem*pz for z_elem in z]

    x_deriv = array([(x[i+1]-x[i-1])/sqrt((x[i+1]-x[i-1])**2+(y[i+1]-y[i-1])**2+(z[i+1]-z[i-1])**2) for i in range(1,len(x)-1)])
    y_deriv = array([(y[i+1]-y[i-1])/sqrt((x[i+1]-x[i-1])**2+(y[i+1]-y[i-1])**2+(z[i+1]-z[i-1])**2) for i in range(1,len(y)-1)])
    z_deriv = array([(z[i+1]-z[i-1])/sqrt((x[i+1]-x[i-1])**2+(y[i+1]-y[i-1])**2+(z[i+1]-z[i-1])**2) for i in range(1,len(z)-1)])

    x_deriv = insert(x_deriv, 0, (x[1]-x[0])/sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2))
    x_deriv = append(x_deriv, (x[-1]-x[-2])/sqrt((x[-1]-x[-2])**2+(y[-1]-y[-2])**2+(z[-1]-z[-2])**2))

    #print len(x_deriv)

    y_deriv = insert(y_deriv, 0, (y[1]-y[0])/sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2))
    y_deriv = append(y_deriv, (y[-1]-y[-2])/sqrt((x[-1]-x[-2])**2+(y[-1]-y[-2])**2+(z[-1]-z[-2])**2))

    z_deriv = insert(z_deriv, 0, (z[1]-z[0])/sqrt((x[1]-x[0])**2+(y[1]-y[0])**2+(z[1]-z[0])**2))
    z_deriv = append(z_deriv, (z[-1]-z[-2])/sqrt((x[-1]-x[-2])**2+(y[-1]-y[-2])**2+(z[-1]-z[-2])**2))

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
    from numpy import sort, abs, zeros, ones, array, sum, median, clip
    
    n = len(x)
    r = int(ceil(f*n))
    h = [sort(abs(x - x[i]))[r] for i in range(n)]
    w = clip(abs((x[:,None] - x[None,:]) / h), 0.0, 1.0)
    w = (1 - w**3)**3
    yest = zeros(n)
    delta = ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:,i]
            b = array([sum(weights*y), sum(weights*y*x)])
            A = array([[sum(weights), sum(weights*x)],
                   [sum(weights*x), sum(weights*x*x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1]*x[i]

        residuals = y - yest
        s = median(abs(residuals))
        delta = clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta**2)**2

    return yest



#=======================================================================================================================
# TODO: ADD DESCRIPTION
#=======================================================================================================================
def opt_f(x, y, z):
    from numpy import max, mean, linalg
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

            msex = mean_squared_error(x, x_fit)
            msey = mean_squared_error(y, y_fit)

            if msx < msx_min:
                msx_min = msx
                f_opt_x = f
            if msy < msy_min:
                msy_min = msy
                f_opt_y = f

            x_fit_d, y_fit_d, z_d = evaluate_derivative_3D(x_fit, y_fit, z)
            x_fit_dd, y_fit_dd, z_dd = evaluate_derivative_3D(x_fit_d, y_fit_d, z_d)
            amp_xd = max(abs(x_fit_dd))
            amp_yd = max(abs(y_fit_dd))
            mean_xd = mean(x_fit_dd)
            mean_yd = mean(y_fit_dd)
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

        except linalg.linalg.LinAlgError:
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

def b_spline_nurbs(x, y, z, fname_centerline=None, degree=3, point_number=3000, nbControl=-1, verbose=1):

    from math import log
    from msct_nurbs import NURBS

    """x.reverse()
    y.reverse()
    z.reverse()"""
          
    print '\nFitting centerline using B-spline approximation...'
    data = [[x[n], y[n], z[n]] for n in range(len(x))]

    # if control_points == 0:
    #     nurbs = NURBS(degree, point_number, data) # BE very careful with the spline order that you choose : if order is too high ( > 4 or 5) you need to set a higher number of Control Points (cf sct_nurbs ). For the third argument (number of points), give at least len(z_centerline)+500 or higher
    # else:
    #     print 'In b_spline_nurbs we get control_point = ', control_points
    #     nurbs = NURBS(degree, point_number, data, False, control_points)

    if nbControl == -1:
        centerlineSize = getSize(x, y, z, fname_centerline)
        nbControl = 30*log(centerlineSize, 10) - 42
        nbControl = round(nbControl)

    nurbs = NURBS(degree, point_number, data, False, nbControl, verbose)

    P = nurbs.getCourbe3D()
    x_fit=P[0]
    y_fit=P[1]
    z_fit=P[2]
    Q = nurbs.getCourbe3D_deriv()
    x_deriv=Q[0]
    y_deriv=Q[1]
    z_deriv=Q[2]

    """x_fit = x_fit[::-1]
    y_fit = x_fit[::-1]
    z_fit = x_fit[::-1]
    x_deriv = x_fit[::-1]
    y_deriv = x_fit[::-1]
    z_deriv = x_fit[::-1]"""

    PC = nurbs.getControle()
    PC_x = [p[0] for p in PC]
    PC_y = [p[1] for p in PC]
    PC_z = [p[2] for p in PC]

    if verbose == 2:
        import matplotlib.pyplot as plt
        plt.figure(1)
        #ax = plt.subplot(211)
        plt.subplot(211)
        plt.plot(z, x, 'r.')
        plt.plot(z_fit, x_fit)
        plt.plot(PC_z,PC_x,'go')
        plt.title("X")
        #ax.set_aspect('equal')
        plt.xlabel('z')
        plt.ylabel('x')
        #ay = plt.subplot(212)
        plt.subplot(212)
        plt.plot(z, y, 'r.')
        plt.plot(z_fit, y_fit)
        plt.plot(PC_z,PC_y,'go')
        plt.title("Y")
        #ay.set_aspect('equal')
        plt.xlabel('z')
        plt.ylabel('y')
        plt.show()
  
    return x_fit, y_fit, z_fit, x_deriv, y_deriv, z_deriv



#=======================================================================================================================
# 3D B-Spline function using ITK
#=======================================================================================================================
def b_spline_nurbs_itk(fname_centerline, numberOfLevels=10):

    print '\nFitting centerline using B-spline approximation (using ITK)...'
    import sct_utils as sct
    status, output = sct.run("isct_bsplineapproximator -i "+fname_centerline+" -o tmp.centerline.txt -l "+str(numberOfLevels))
    if (status != 0):
        print "WARNING: \n"+output

    f = open('tmp.centerline.txt', 'r')
    x_fit = []
    y_fit = []
    z_fit = []
    x_deriv = []
    y_deriv = []
    z_deriv = []
    for line in f:
        center = line.split(' ')
        x_fit.append(float(center[0]))
        y_fit.append(float(center[1]))
        z_fit.append(float(center[2]))
        x_deriv.append(float(center[3]))
        y_deriv.append(float(center[4]))
        z_deriv.append(float(center[5]))

    return x_fit, y_fit, z_fit, x_deriv, y_deriv, z_deriv



#=======================================================================================================================
# get size
#=======================================================================================================================
def getSize(x, y, z, file_name=None):
    from commands import getstatusoutput
    from math import sqrt
    # get pixdim
    if file_name is not None:
        cmd1 = 'fslval '+file_name+' pixdim1'
        status, output = getstatusoutput(cmd1)
        p1 = float(output)
        cmd2 = 'fslval '+file_name+' pixdim2'
        status, output = getstatusoutput(cmd2)
        p2 = float(output)
        cmd3 = 'fslval '+file_name+' pixdim3'
        status, output = getstatusoutput(cmd3)
        p3 = float(output)
    else:
        p1, p2, p3 = 1.0, 1.0, 1.0

    # Centerline size
    s = 0
    for i in xrange (len(x)-1):
        s += sqrt((p1*(x[i+1]-x[i]))**2+(p2*(y[i+1]-y[i]))**2+(p3*(z[i+1]-z[i])**2))
    #print "centerline size: ", s
    return s



#=======================================================================================================================
# functions to get centerline size
#=======================================================================================================================
def getPxDimensions(file_name):
    from commands import getstatusoutput
    cmd1 = 'fslval '+file_name+' pixdim1'
    status, output = getstatusoutput(cmd1)
    p1 = float(output)
    cmd2 = 'fslval '+file_name+' pixdim2'
    status, output = getstatusoutput(cmd2)
    p2 = float(output)
    cmd3 = 'fslval '+file_name+' pixdim3'
    status, output = getstatusoutput(cmd3)
    p3 = float(output)
    return p1, p2, p3



#=======================================================================================================================
# 3D B-Spline function, python function
#=======================================================================================================================
def b_spline_python(x, y, z, s = 0, k = 3, nest = -1):
    """see http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html for full input information"""
    from scipy.interpolate import splprep, splev
    tckp, u = splprep([x,y,z], s = s, k = k, nest = nest)
    xnew, ynew, znew = splev(u, tckp)
    return xnew, ynew, znew



#=======================================================================================================================
# lowpass filter  
#=======================================================================================================================
def lowpass(y):
    from scipy.fftpack import fftfreq, fft
    from scipy.signal import filtfilt, iirfilter
    from numpy import abs, amax
    frequency = fftfreq(len(y))
    spectrum = abs(fft(y, n=None, axis=-1, overwrite_x=False))
    Wn = amax(frequency)/10
    N = 5  # Order of the filter
    b, a = iirfilter(N, Wn, rp=None, rs=None, btype='low', analog=False, ftype='butter', output='ba')
    y_smooth = filtfilt(b, a, y, axis=-1, padtype=None)
    return y_smooth



#=======================================================================================================================
# moving_average
#=======================================================================================================================   
def moving_average(y, n=3):
    from numpy import cumsum
    y_smooth = cumsum(y, dtype=float)
    y_smooth[n:] = y_smooth[n:] - y_smooth[:-n]
    return y_smooth[n - 1:] / n



#=======================================================================================================================
# moving_average
#=======================================================================================================================
def mean_squared_error(x, x_fit):
    mse = 0
    if len(x_fit) == len(x) and len(x) is not 0:
        n = len(x)
        for i in range(0, len(x)):
            mse += (x[i]-x_fit[i])*(x[i]-x_fit[i])
        mse = float(mse)
        mse *= (1/float(n))
        return mse
    else:
        print "cannot calculate the mean squared error, check if the argument have the same length. \n"



#=======================================================================================================================
# windowing
#=======================================================================================================================
def smoothing_window(x, window_len=11, window='hanning', verbose = 0):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window (in number of points); should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smoothing_window(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string

    NOTE: If in 'convolve', the mode is not 'same' then: length(output) != length(input), to correct this:
    return y[(window_len/2-1):-(window_len/2)] instead of just y if window_len is even
    return y[(window_len/2-1):-(window_len/2)+1] instead of just y if window_len is odd.
    """
    from numpy import append, insert, ones, convolve, hanning  # IMPORTANT: here, we only import hanning. For more windows, add here.
    from math import ceil
    import sct_utils as sct

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        sct.printv('Window size is too small. No smoothing was applied.', 1, 'warning')
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window can only be the following: 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    ## Checking the window's size
    nb_points = x.shape[0]
    #The number of points of the curve must be superior to int(window_length/(2.0*pz))
    if window_len > int(2*nb_points):
        window_len = int(2*nb_points)
        sct.printv("WARNING: The ponderation window's length was too high compared to the number of points. The value is now of: "+str(window_len) +'warning')

    # make window_len as odd integer (x = x+1 if x is even)
    window_len_int = ceil((window_len + 1)/2)*2 - 1

    # s = r_[x[window_len_int-1:0:-1], x, x[-1:-window_len_int:-1]]

    # Creation of the window
    if window == 'flat': #moving average
        w = ones(window_len, 'd')
    else:
        w = eval(window+'(window_len_int)')

    ##Implementation of an extended curve to apply the smoothing on in order to avoid edge effects
    # Extend the curve before smoothing
    x_extended = x
    size_curve = x.shape[0]
    size_padding = int(round(window_len/2.0))

    for i in range(size_padding):
        x_extended = append(x_extended, 2*x[-1] - x[-2-i])
        x_extended = insert(x_extended, 0, 2*x[0] - x[i+1])

    # Convolution of the window with the extended signal
    y = convolve(x_extended, w/w.sum(), mode='valid')

    # Display smoothing
    if verbose == 2:
        import matplotlib.pyplot as plt
        z = [i for i in range(y.shape[0])]

        plt.figure()
        pltx, = plt.plot(z, x, 'ro')
        pltx_fit, = plt.plot(z, y)

        plt.title("Type of window: %s     Window_length= %d mm" % (window, window_len))
        #ax.set_aspect('equal')
        plt.xlabel('z')
        plt.ylabel('x')
        plt.legend([pltx, pltx_fit], ['Normal', 'Smoothed'])

        plt.show()

    return y
    # if window_len_int%2 == 0:
    #     return y[(window_len_int/2-1):-(window_len_int/2)]
    # if window_len_int%2 != 0:
    #     return y[(window_len_int/2-1):-(window_len_int/2+1)]
