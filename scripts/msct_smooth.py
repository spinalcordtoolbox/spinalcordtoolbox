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
import sct_utils as sct


#=======================================================================================================================
# Over pad the input file, smooth and return the centerline
#=======================================================================================================================
def smooth(fname, padding):
    sct.run('sct_image -i ' + fname + ' -o tmp.centerline_pad.nii.gz -pad ' + str(padding) + ',' + str(padding) + ',' + str(padding))


#=======================================================================================================================
# Spline 2D using splrep & splev
#=======================================================================================================================
def spline_2D(z_centerline, x_centerline):

    from numpy import mean, std, sqrt

    m = mean(x_centerline)
    sigma = std(x_centerline)
    sct.printv((m - sqrt(2 * m)) * (sigma**2), (m + sqrt(2 * m)) * (sigma**2))

    smoothing_param = (((m + sqrt(2 * m)) * (sigma**2)) + ((m - sqrt(2 * m)) * (sigma**2))) / 2
    sct.printv('\nSmoothing results with spline...')
    tck = splrep(z_centerline, x_centerline, s = smoothing_param)
    x_centerline_fit = splev(z_centerline, tck)
    return x_centerline_fit


#=======================================================================================================================
# Polynomial fit
#=======================================================================================================================
def polynomial_fit(x, y, degree):

    import numpy as np

    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    y_fit = np.polyval(poly, x)

    return y_fit, poly


#=======================================================================================================================
# Polynomial derivative
#=======================================================================================================================
def polynomial_deriv(x, poly):

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
    for i in xrange(len(x) - 1):
        s += sqrt((p1 * (x[i + 1] - x[i]))**2 + (p2 * (y[i + 1] - y[i]))**2 + (p3 * (z[i + 1] - z[i])**2))
    sct.printv("centerline size: ", s)
    return s


#=======================================================================================================================
# Evaluate derivative of data points
#=======================================================================================================================
def evaluate_derivative_2D(x, y, px, py):
    """
    Compute derivative in 2D, accounting for pixel size in each dimension
    :param x:
    :param y:
    :param px:
    :param py:
    :return:
    """

    from numpy import array, sqrt, insert, append

    x = [x_elem * px for x_elem in x]
    y = [y_elem * py for y_elem in y]

    # compute derivative for points 2 --> n-1
    x_deriv = array([(x[i + 1] - x[i - 1]) / (y[i + 1] - y[i - 1]) for i in range(1, len(x) - 1)])
    y_deriv = array([(y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]) for i in range(1, len(y) - 1)])

    # compute derivatives for points 1 and n.
    x_deriv = insert(x_deriv, 0, (x[1] - x[0]) / (y[1] - y[0]))
    x_deriv = append(x_deriv, (x[-1] - x[-2]) / (y[-1] - y[-2]))
    y_deriv = insert(y_deriv, 0, (y[1] - y[0]) / (x[1] - x[0]))
    y_deriv = append(y_deriv, (y[-1] - y[-2]) / (x[-1] - x[-2]))

    return x_deriv, y_deriv


#=======================================================================================================================
# Evaluate derivative of data points in 3D
#=======================================================================================================================
def evaluate_derivative_3D(x, y, z, px, py, pz):
    """
    Compute derivative in 3D, accounting for pixel size in each dimension
    :param x:
    :param y:
    :param z:
    :param px:
    :param py:
    :param pz:
    :return:
    """
    from numpy import array, sqrt, insert, append

    x = [x_elem * px for x_elem in x]
    y = [y_elem * py for y_elem in y]
    z = [z_elem * pz for z_elem in z]

    # compute derivative for points 2 --> n-1
    x_deriv = array([(x[i + 1] - x[i - 1]) / sqrt((x[i + 1] - x[i - 1])**2 + (y[i + 1] - y[i - 1])**2 + (z[i + 1] - z[i - 1])**2) for i in range(1, len(x) - 1)])
    y_deriv = array([(y[i + 1] - y[i - 1]) / sqrt((x[i + 1] - x[i - 1])**2 + (y[i + 1] - y[i - 1])**2 + (z[i + 1] - z[i - 1])**2) for i in range(1, len(y) - 1)])
    z_deriv = array([(z[i + 1] - z[i - 1]) / sqrt((x[i + 1] - x[i - 1])**2 + (y[i + 1] - y[i - 1])**2 + (z[i + 1] - z[i - 1])**2) for i in range(1, len(z) - 1)])

    # compute derivatives for points 1 and n.
    x_deriv = insert(x_deriv, 0, (x[1] - x[0]) / sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2 + (z[1] - z[0])**2))
    x_deriv = append(x_deriv, (x[-1] - x[-2]) / sqrt((x[-1] - x[-2])**2 + (y[-1] - y[-2])**2 + (z[-1] - z[-2])**2))
    y_deriv = insert(y_deriv, 0, (y[1] - y[0]) / sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2 + (z[1] - z[0])**2))
    y_deriv = append(y_deriv, (y[-1] - y[-2]) / sqrt((x[-1] - x[-2])**2 + (y[-1] - y[-2])**2 + (z[-1] - z[-2])**2))
    z_deriv = insert(z_deriv, 0, (z[1] - z[0]) / sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2 + (z[1] - z[0])**2))
    z_deriv = append(z_deriv, (z[-1] - z[-2]) / sqrt((x[-1] - x[-2])**2 + (y[-1] - y[-2])**2 + (z[-1] - z[-2])**2))

    return x_deriv, y_deriv, z_deriv


#=======================================================================================================================
# Non parametric regression
#=======================================================================================================================
def non_parametric(x, y, f = 0.25, iter = 3):
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
    r = int(ceil(f * n))
    h = [sort(abs(x - x[i]))[r] for i in range(n)]
    w = clip(abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w**3)**3
    yest = zeros(n)
    delta = ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = array([sum(weights * y), sum(weights * y * x)])
            A = array([[sum(weights), sum(weights * x)],
                   [sum(weights * x), sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

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
    sct.printv('optimizing f parameter in non-parametric...')
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

            if msex < msx_min:
                msx_min = msex
                f_opt_x = f
            if msey < msy_min:
                msy_min = msey
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

            sct.printv('AMP', amp_xd, amp_yd)
            sct.printv('MEAN', mean_xd, mean_yd, mean)

        except linalg.linalg.LinAlgError:
            sct.printv('LinAlgError raised')
    sct.printv(msx_min, f_opt_x)
    sct.printv(msy_min, f_opt_y)
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
def b_spline_nurbs(x, y, z, fname_centerline=None, degree=3, point_number=3000, nbControl=-1, verbose=1, all_slices=True, path_qc='../'):

    from math import log
    from msct_nurbs import NURBS

    twodim = False
    if z is None:
        twodim = True

    """x.reverse()
    y.reverse()
    z.reverse()"""

    sct.printv('\nFitting centerline using B-spline approximation...', verbose)
    if not twodim:
        data = [[x[n], y[n], z[n]] for n in range(len(x))]
    else:
        data = [[x[n], y[n]] for n in range(len(x))]

    # if control_points == 0:
    #     nurbs = NURBS(degree, point_number, data) # BE very careful with the spline order that you choose : if order is too high ( > 4 or 5) you need to set a higher number of Control Points (cf sct_nurbs ). For the third argument (number of points), give at least len(z_centerline)+500 or higher
    # else:
    #     sct.printv('In b_spline_nurbs we get control_point = ', control_points)
    #     nurbs = NURBS(degree, point_number, data, False, control_points)

    if nbControl == -1:
        centerlineSize = getSize(x, y, z, fname_centerline)
        nbControl = 30 * log(centerlineSize, 10) - 42
        nbControl = round(nbControl)

    nurbs = NURBS(degree, point_number, data, False, nbControl, verbose, all_slices=all_slices, twodim=twodim)

    if not twodim:
        P = nurbs.getCourbe3D()
        x_fit = P[0]
        y_fit = P[1]
        z_fit = P[2]
        Q = nurbs.getCourbe3D_deriv()
        x_deriv = Q[0]
        y_deriv = Q[1]
        z_deriv = Q[2]
    else:
        P = nurbs.getCourbe2D()
        x_fit = P[0]
        y_fit = P[1]
        Q = nurbs.getCourbe2D_deriv()
        x_deriv = Q[0]
        y_deriv = Q[1]

    """x_fit = x_fit[::-1]
    y_fit = x_fit[::-1]
    z_fit = x_fit[::-1]
    x_deriv = x_fit[::-1]
    y_deriv = x_fit[::-1]
    z_deriv = x_fit[::-1]"""

    if verbose == 2:
        PC = nurbs.getControle()
        PC_x = [p[0] for p in PC]
        PC_y = [p[1] for p in PC]
        if not twodim:
            PC_z = [p[2] for p in PC]

        import matplotlib
        matplotlib.use('Agg')  # prevent display figure
        import matplotlib.pyplot as plt
        if not twodim:
            plt.figure(1)
            #ax = plt.subplot(211)
            plt.subplot(211)
            plt.plot(z, x, 'r.')
            plt.plot(z_fit, x_fit)
            plt.plot(PC_z, PC_x, 'go')
            plt.title("X")
            # ax.set_aspect('equal')
            plt.xlabel('z')
            plt.ylabel('x')
            #ay = plt.subplot(212)
            plt.subplot(212)
            plt.plot(z, y, 'r.')
            plt.plot(z_fit, y_fit)
            plt.plot(PC_z, PC_y, 'go')
            plt.title("Y")
            # ay.set_aspect('equal')
            plt.xlabel('z')
            plt.ylabel('y')
            plt.show()
        else:
            plt.figure(1)
            plt.plot(y, x, 'r.')
            plt.plot(y_fit, x_fit)
            plt.plot(PC_y, PC_x, 'go')
            # ax.set_aspect('equal')
            plt.xlabel('y')
            plt.ylabel('x')
            plt.show()
        plt.savefig(path_qc+'b_spline_nurbs.png')
        plt.close()

    if not twodim:
        return x_fit, y_fit, z_fit, x_deriv, y_deriv, z_deriv, nurbs.error_curve_that_last_worked
    else:
        return x_fit, y_fit, x_deriv, y_deriv, nurbs.error_curve_that_last_worked


#=======================================================================================================================
# 3D B-Spline function using ITK
#=======================================================================================================================
def b_spline_nurbs_itk(fname_centerline, numberOfLevels=10):

    sct.printv('\nFitting centerline using B-spline approximation (using ITK)...')
    import sct_utils as sct
    status, output = sct.run("isct_bsplineapproximator -i " + fname_centerline + " -o tmp.centerline.txt -l " + str(numberOfLevels))
    if (status != 0):
        sct.printv("WARNING: \n" + output)

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
        cmd1 = 'fslval ' + file_name + ' pixdim1'
        status, output = getstatusoutput(cmd1)
        p1 = float(output)
        cmd2 = 'fslval ' + file_name + ' pixdim2'
        status, output = getstatusoutput(cmd2)
        p2 = float(output)
        cmd3 = 'fslval ' + file_name + ' pixdim3'
        status, output = getstatusoutput(cmd3)
        p3 = float(output)
    else:
        p1, p2, p3 = 1.0, 1.0, 1.0

    # Centerline size
    s = 0
    for i in xrange(len(x) - 1):
        s += sqrt((p1 * (x[i + 1] - x[i]))**2 + (p2 * (y[i + 1] - y[i]))**2 + (p3 * (z[i + 1] - z[i])**2))
    # sct.printv("centerline size: ", s)
    return s


#=======================================================================================================================
# functions to get centerline size
#=======================================================================================================================
def getPxDimensions(file_name):
    from commands import getstatusoutput
    cmd1 = 'fslval ' + file_name + ' pixdim1'
    status, output = getstatusoutput(cmd1)
    p1 = float(output)
    cmd2 = 'fslval ' + file_name + ' pixdim2'
    status, output = getstatusoutput(cmd2)
    p2 = float(output)
    cmd3 = 'fslval ' + file_name + ' pixdim3'
    status, output = getstatusoutput(cmd3)
    p3 = float(output)
    return p1, p2, p3


#=======================================================================================================================
# 3D B-Spline function, python function
#=======================================================================================================================
def b_spline_python(x, y, z, s = 0, k = 3, nest = -1):
    """see http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.splprep.html for full input information"""
    from scipy.interpolate import splprep, splev
    tckp, u = splprep([x, y, z], s = s, k = k, nest = nest)
    xnew, ynew, znew = splev(u, tckp)
    return xnew, ynew, znew


#=======================================================================================================================
# lowpass filter
#=======================================================================================================================
def lowpass(y):
    """Signal smoothing by low pass filtering.

    This method is based on the application of a butterworth low pas filter of order 5 to the signal. It skims out the
    higher frequencies that are responsible for abrupt changes thus smoothing the signal. Output edges are different
    from input edges.

    input:
        y: input signal (type: list)

    output:
        y_smooth : filtered signal (type: ndarray)

    """
    from scipy.fftpack import fftfreq, fft
    from scipy.signal import filtfilt, iirfilter
    from numpy import abs, amax
    frequency = fftfreq(len(y))
    spectrum = abs(fft(y, n=None, axis=-1, overwrite_x=False))
    Wn = amax(frequency) / 10
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
            mse += (x[i] - x_fit[i]) * (x[i] - x_fit[i])
        mse = float(mse)
        mse *= (1 / float(n))
        return mse
    else:
        sct.printv("cannot calculate the mean squared error, check if the argument have the same length. \n")


#=======================================================================================================================
# windowing
#=======================================================================================================================
def smoothing_window(x, window_len=11, window='hanning', verbose = 0, robust=0, remove_edge_points=2):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal (type: array)
        window_len: the dimension of the smoothing window (in number of points); should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        y: the smoothed signal (type: array). Same size as x.

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smoothing_window(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """
    from numpy import append, insert, ones, convolve, hanning  # IMPORTANT: here, we only import hanning. For more windows, add here.
    from math import ceil, floor
    import sct_utils as sct

    # outlier detection
    if robust:
        mask = outliers_detection(x, type='median', factor=2, return_filtered_signal='no', verbose=verbose)
        x = outliers_completion(mask, verbose=0)

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        sct.printv('Window size is too small. No smoothing was applied.', verbose=verbose, type='warning')
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window can only be the following: 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    # make sure there are enough points before removing those at the edge
    size_curve = x.size
    if size_curve < 10:
        remove_edge_points = 0

    # remove edge points (in case segmentation is bad at the edge)
    if not remove_edge_points == 0:
        x_extended = x[remove_edge_points:-remove_edge_points]  # remove points at the edge (jcohenadad, issue #513)
    else:
        x_extended = x

    # Checking the window's size
    nb_points = x_extended.shape[0]
    # The number of points of the curve must be superior to int(window_length/(2.0*pz))
    if window_len > int(nb_points):
        window_len = int(nb_points)
        sct.printv("WARNING: The smoothing window is larger than the number of points. New value: " + str(window_len), verbose=verbose, type='warning')

    # Make window_len as odd integer (x = x+1 if x is even)
    window_len_int = ceil((floor(window_len) + 1) / 2) * 2 - 1

    # Add padding
    size_padding = int(round((window_len_int - 1) / 2.0) + remove_edge_points)
    for i in range(size_padding):
        x_extended = append(x_extended, 2 * x_extended[-1 - i] - x_extended[-1 - 2 * i - 1])
        x_extended = insert(x_extended, 0, 2 * x_extended[i] - x_extended[2 * i + 1])

    # Creation of the window
    if window == 'flat':  # moving average
        w = ones(window_len, 'd')
    else:
        w = eval(window + '(window_len_int)')

    # Convolution of the window with the extended signal
    # len(y) = (len(x_extended) + len(w)) / 2
    y = convolve(x_extended, w / w.sum(), mode='valid')

    # Display smoothing
    if verbose == 2:
        import matplotlib.pyplot as plt
        from copy import copy
        z = [i + size_padding - remove_edge_points for i in range(x.shape[0])]
        z_extended = [i for i in range(x_extended.shape[0])]
        # Create x_display to visualize concording results
        plt.figure()
        pltx_ext, = plt.plot(z_extended, x_extended, 'go')
        pltx, = plt.plot(z, x, 'bx')
        #pltx, = plt.plot(z_extended[size_padding:size_padding + size_curve], x_display[size_padding:size_padding + size_curve], 'bo')
        pltx_fit, = plt.plot(z, y, 'r', linewidth=2)
        plt.title("Type of window: %s     Window_length= %d mm" % (window, window_len))
        plt.xlabel('z')
        plt.ylabel('x')
        plt.legend([pltx_ext, pltx, pltx_fit], ['Extended', 'Normal', 'Smoothed'])
        plt.show()

    return y


def outliers_detection(data, type='median', factor=2, return_filtered_signal='no', verbose=0):
    """Detect outliers within a signal.

    This method is based on the comparison of the distance between points of the signal and the mean of the signal.
    There are two types of detection process.
        -'std' process compares the distance between the mean of the signal
    and the points of the signal with the std. If the distance is superior to factor * std than the point is considered
    as an outlier
        -'median' process first detect extreme outliers using the Median Absolute Deviation (MAD) and then calculate the
        std with the filtered signal (i.e. the signal without the extreme outliers). It then uses the same process as the
        'std' process comparing the distance between the mean and the points to the std. The idea beneath is that the
        calculation of the std is biased by the presence of the outliers; retrieving extreme ones before hand improves
        the accuracy of the algorithm. (http://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers)

    input:
        data: the input signal (type: array)
        type: the type of algorithm process ('median' or 'std') (type: string)
        factor: the sensibility of outlier detection (if infinite no outlier will be find) (type: int or float)
        return_filtered_signal: option to ask for the 'filtered signal' (i.e. the signal of smaller shape that present
            no outliers) ('yes' or 'no')
        verbose: display parameter; specify 'verbose = 2' if display is desired (type: int)

    output:
        mask: a mask of same shape as the input signal that takes same values for non outlier points and 'nan' values for
        outliers (type: array)
        filtered [optional]: signal of smaller shape that present no outliers (type: array)


    TODO: other outlier detection algorithms could be implemented
    """

    from numpy import mean, median, std, isnan, asarray
    from copy import copy

    # if nan are detected in data, replace by extreme values so that they are detected by outliers detection algo
    for i in xrange(len(data)):
        if isnan(data[i]):
            data[i] = 9999999

    if type == 'std':
        u = mean(data)
        s = std(data)
        index_1 = data > (u + factor * s)
        index_2 = (u - factor * s) > data
        filtered = [e for e in data if (u - factor * s < e < u + factor * s)]
        mask = copy(data)
        mask[index_1] = None
        mask[index_2] = None

    if type == 'median':
        # Detect extrem outliers using median
        d = abs(data - median(data))
        mdev = 1.4826 * median(d)
        s = d / mdev if mdev else 0.
        mean_s = mean(s)
        index_1 = s > 5 * mean_s
        mask_1 = copy(data)
        mask_1[index_1] = None
        filtered_1 = [e for i, e in enumerate(data.tolist()) if not isnan(mask_1[i])]
        # Recalculate std using filtered variable and detect outliers with threshold factor * std
        u = mean(filtered_1)
        std_1 = std(filtered_1)
        filtered = [e for e in data if (u - factor * std_1 < e < u + factor * std_1)]
        index_1_2 = data > (u + factor * std_1)
        index_2_2 = (u - factor * std_1) > data
        mask = copy(data)
        mask[index_1_2] = None
        mask[index_2_2] = None

    if verbose == 2:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.subplot(211)
        plt.plot(data, 'bo')
        axes = plt.gca()
        y_lim = axes.get_ylim()
        plt.title("Before outliers deletion")
        plt.subplot(212)
        plt.plot(mask, 'bo')
        plt.ylim(y_lim)
        plt.title("After outliers deletion")
        plt.show()
    if return_filtered_signal == 'yes:':
        filtered = asarray(filtered)
        return filtered, mask
    else:
        return mask


def outliers_completion(mask, verbose=0):
    """Replace outliers within a signal.

    This method is based on the replacement of outlier using linear interpolation with closest non outlier points by
    recurrence. We browse through the signal from 'left to right' replacing each outlier by the average of the two
    closest non outlier points. Once one outlier is replaced, it is no longer consider as an outlier and may be used for
    the calculation of the replacement value of the next outlier (recurrence process).
    To be used after outlier_detection.

    input:
        mask: the input signal (type: array) that takes 'nan' values at the position of the outlier to be retrieved
        verbose: display parameters; specify 'verbose = 2' if display is desired (type: int)

    output:
        signal_completed: the signal of input that has been completed (type: array)

    example:

    mask_x = outliers_detection(x, type='median', factor=factor, return_filtered_signal='no', verbose=0)
    x_no_outliers = outliers_completion(mask_x, verbose=0)

    N.B.: this outlier replacement technique is not a good statistical solution. Our desire of replacing outliers comes
    from the fact that we need to treat data of same shape but by doing so we are also flawing the signal.
    """
    from numpy import nan_to_num, nonzero, transpose, append, insert, isnan
    # Complete mask that as nan values by linear interpolation of the closest points
    # Define signal completed
    signal_completed = nan_to_num(mask)
    # take index of all non nan points
    X_signal_completed = nonzero(signal_completed)
    X_signal_completed = transpose(X_signal_completed)
    # initialization: we set the extrem values to avoid edge effects
    if len(X_signal_completed) != 0:
        signal_completed[0] = signal_completed[X_signal_completed[0]]
        signal_completed[-1] = signal_completed[X_signal_completed[-1]]
        # Add two rows to the vector X_signal_completed:
        # one before as signal_completed[0] is now diff from 0
        # one after as signal_completed[-1] is now diff from 0
        X_signal_completed = append(X_signal_completed, len(signal_completed) - 1)
        X_signal_completed = insert(X_signal_completed, 0, 0)
        # linear interpolation
        # count_zeros=0
        for i in range(1, len(signal_completed) - 1):
            if signal_completed[i] == 0:
            # signal_completed[i] = ((X_signal_completed[i-count_zeros]-i) * signal_completed[X_signal_completed[i-1-count_zeros]] + (i-X_signal_completed[i-1-count_zeros]) * signal_completed[X_signal_completed[i-count_zeros]])/float(X_signal_completed[i-count_zeros]-X_signal_completed[i-1-count_zeros]) # linear interpolation ponderate by distance with closest non zero points
            # signal_completed[i] = 0.25 * (signal_completed[X_signal_completed[i-1-count_zeros]] + signal_completed[X_signal_completed[i-count_zeros]] + signal_completed[X_signal_completed[i-2-count_zeros]] + signal_completed[X_signal_completed[i-count_zeros+1]]) # linear interpolation with closest non zero points (2 points on each side)
                signal_completed[i] = 0.5 * (signal_completed[X_signal_completed[i - 1]] + signal_completed[X_signal_completed[i]])  # linear interpolation with closest non zero points
                # redefine X_signal_completed
                X_signal_completed = nonzero(signal_completed)
                X_signal_completed = transpose(X_signal_completed)
                #count_zeros += 1
    if verbose == 2:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(mask, 'bo')
        plt.title("Before outliers completion")
        axes = plt.gca()
        y_lim = axes.get_ylim()
        plt.subplot(2, 1, 2)
        plt.plot(signal_completed, 'bo')
        plt.title("After outliers completion")
        plt.ylim(y_lim)
        plt.show()
    return signal_completed
