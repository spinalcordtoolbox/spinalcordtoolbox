#!/usr/bin/env python
#########################################################################################
#
# Vertebral Disks Detection
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Julien Cohen-Adad
# Modified: 2014-06-14
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# check if needed Python libraries are already installed or not
import os
import sys
import time
import getopt
import commands
import math
import scipy
import scipy.signal
import scipy.fftpack

try:
    import nibabel
except ImportError:
    print '--- nibabel not installed! Exit program. ---'
    sys.exit(2)
try:
    import numpy as np
except ImportError:
    print '--- numpy not installed! Exit program. ---'
    sys.exit(2)

# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
import sct_utils as sct

fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI

#=======================================================================================================================
# class definition
#=======================================================================================================================

class param_class:
    def __init__(self):
    
        # PATH AND FILE NAME FOR ANATOMICAL IMAGE
        self.debug                     = 0
        self.input_anat                = ''
        self.contrast                  = ''
        self.mean_distance_mat         = ''
        self.output_path               = ''
        
        # Spinal Cord labeling Parameters
        self.input_centerline          = ''                                        # optional
        self.shift_AP                  = 17                                        # shift the centerline on the spine in mm default : 17 mm
        self.size_AP                   = 6                                         # mean around the centerline in the anterior-posterior direction in mm
        self.size_RL                   = 5                                         # mean around the centerline in the right-left direction in mm
        
        # =======================================================
        # OTHER PARAMETERS
        self.verbose                   = 0                                         # display text
        self.plot_graph                = 0
            
#=======================================================================================================================
# main
#=======================================================================================================================

def main():
    
    print '\n\n\n==================================================='
    print '               Running: sct_labeling'
    print '===================================================\n'
    
    # Initialization
    start_time = time.time()
    param = param_class()
    
    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:c:l:m:a:s:r:o:g:v:')
    except getopt.GetoptError as err:
        print str(err)
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            param.input_anat = arg
        elif opt in ('-c'):
            param.contrast = arg
        elif opt in ('-l'):
            param.input_centerline = arg
        elif opt in ('-m'):
            param.mean_distance_mat = arg
        elif opt in ('-a'):
            param.shift_AP = int(arg)
        elif opt in ('-s'):
            param.size_AP = int(arg)
        elif opt in ('-r'):
            param.size_RL = int(arg)
        elif opt in ('-o'):
            param.output_path = arg
        elif opt in ('-g'):
            param.plot_graph = int(arg)
        elif opt in ('-v'):
            param.verbose = int(arg)

    # Display usage if a mandatory argument is not provided
    if param.input_anat == '' or param.contrast=='' or param.input_centerline=='' or param.mean_distance_mat=='':
        print '\n \n All mandatory arguments are not provided \n \n'
        usage()


    # Extract path, file and extension
    input_path, file_data, ext_data = sct.extract_fname(param.input_anat)

    if param.output_path=='': param.output_path = os.getcwd() + '/'

    print 'Input File:',param.input_anat
    print 'Center_line file:',param.input_centerline
    print 'Contrast:',param.contrast
    print 'Mat File:',param.mean_distance_mat,'\n'

    # check existence of input files
    sct.check_file_exist(param.input_anat)
    sct.check_file_exist(param.input_centerline)
    sct.check_file_exist(param.mean_distance_mat)
    
    verbose = param.verbose

    #==================================================
    # Reorientation of the data if needed
    #==================================================
    command = 'fslhd ' + param.input_anat
    result = commands.getoutput(command)
    orientation = result[result.find('qform_xorient')+15] + result[result.find('qform_yorient')+15] + result[result.find('qform_zorient')+15]

    if orientation!='ASR':
        sct.printv('\nReorient input volume to AP SI RL orientation...',param.verbose)
        sct.run(sct.fsloutput + 'fslswapdim ' + param.input_anat + ' AP SI RL ' + input_path + 'tmp.anat_orient')
        sct.run(sct.fsloutput + 'fslswapdim ' + param.input_centerline + ' AP SI RL ' + input_path + 'tmp.centerline_orient')
        
        param.input_anat = input_path + 'tmp.anat_orient.nii'
        param.input_centerline = input_path + 'tmp.centerline_orient.nii'

    if param.plot_graph:
        import pylab as pl

    #==================================================
    # Loading Images
    #==================================================
    sct.printv('\nLoading Images...',verbose)
    anat_file = nibabel.load(param.input_anat)
    anat = anat_file.get_data()
    hdr = anat_file.get_header()
    dims = hdr['dim']
    scales = hdr['pixdim']
    
    centerline_file = nibabel.load(param.input_centerline)
    centerline = centerline_file.get_data()

    shift_AP = param.shift_AP*scales[1]
    size_AP = param.size_AP*scales[1]
    size_RL = param.size_RL*scales[3]

    np.uint16(anat)
    
    #==================================================
    # Calculation of the profile intensity
    #==================================================
    sct.printv('\nCalculation of the profile intensity...',verbose)

    #find coordinates of the centerline
    X,Y,Z = np.where(centerline>0)

    #reordering x,y,z with y in the growing sense
    j = np.argsort(Y)
    y = Y[j]
    x = X[j]
    z = Z[j]

    #eliminating repeatitions in y
    index=0
    for i in range(len(y)-1):
        if y[i]==y[i+1]:
            if index==0:
                index_double = i
            else:
                index_double = np.resize(index_double,index+1)
                index_double[index] = i
            index = index + 1

    mask = np.ones(len(y), dtype=bool)
    mask[index_double] = False

    y = y[mask]
    x = x[mask]
    z = z[mask]
    
    #shift the centerline to the spine of shift_AP
    x1 = np.round(x-shift_AP/scales[1])

    #build intensity profile along the centerline
    I = np.zeros((len(y),1))

    for index in range(len(y)):
        lim_plus = index + 5
        lim_minus = index - 5
        
        if lim_minus<0: lim_minus = 0
        if lim_plus>=len(x1): lim_plus = len(x1) - 1

        # normal vector of the orthogonal plane to the centerline i.e tangent vector to the centerline
        Vx = x1[lim_plus] - x1[lim_minus]
        Vz = z[lim_plus] - z[lim_minus]
        Vy = y[lim_plus] - y[lim_minus]

        d = Vx*x1[index] + Vy*y[index] + Vz*z[index]

        #Averaging
        for i_slice_RL in range(2*np.int(round(size_RL/scales[3]))):
            for i_slice_AP in range(2*np.int(round(size_AP/scales[1]))):
                result = (d - Vx*(x1[index] + i_slice_AP - size_AP - 1) - Vz*z[index])/Vy
                
                if result > anat.shape[1]: result = anat.shape[1]
                
                I[index] = I[index] + anat[np.int(round(x1[index]+i_slice_AP - size_AP - 1)),np.int(np.floor(result)),np.int(round(z[index] + i_slice_RL - size_RL - 1))]

    start_centerline_y = y[0]
    X = np.where(I==0)
    mask2 = np.ones((len(y),1), dtype=bool)
    mask2[X,0] = False
#    I = I[mask2]

    if param.plot_graph:
        pl.plot(I)
        pl.xlabel('direction superior-inferior')
        pl.ylabel('intensity')
        pl.title('Intensity profile along the shifted spinal cord centerline')
        pl.show()

    #==================================================
    # Detrending Intensity
    #==================================================
    sct.printv('\nDetrending Intensity...',verbose)
    frequency = scipy.fftpack.fftfreq(len(I[:,0]), d=1)
    spectrum = np.abs(scipy.fftpack.fft(I[:,0], n=None, axis=-1, overwrite_x=False))

    #Using iir filter for detrending
    Wn = np.amax(frequency)/10
    N = 5              #Order of the filter
    b, a = scipy.signal.iirfilter(N, Wn, rp=None, rs=None, btype='low', analog=False, ftype='bessel', output='ba')
    I_fit = scipy.signal.filtfilt(b, a, I[:,0], axis=-1, padtype='constant', padlen=None)
    if param.plot_graph:
        pl.plot(I[:,0])
        pl.plot(I_fit)
        pl.show()

    I_detrend = np.zeros((len(I[:,0]),1))
    I_detrend[:,0] = I[:,0] - I_fit
    if param.contrast == 'T1':
        I_detrend = I_detrend/(np.amax(I_detrend))
    else:
        I_detrend = I_detrend/abs((np.amin(I_detrend)))

    if param.plot_graph:
        pl.plot(I_detrend[:,0])
        pl.xlabel('direction superior-inferior')
        pl.ylabel('intensity')
        pl.title('Intensity profile along the shifted spinal cord centerline after detrending and basic normalization')
        pl.show()

#    info_1 = input('\nIs the more rostral vertebrae the C1 or C2 one? if yes, enter 1 otherwise 0:')
#    if info_1==0:
#        level_start = input('enter the level of the more rostral vertebra - choice of the more rostral vertebral level of the field of view:')
#    else:
#        level_start = 2
#
#    mean_distance_dict = scipy.io.loadmat(param.mean_distance_mat)
#    mean_distance = (mean_distance_dict.values()[2]).T
#    C1C2_distance = mean_distance[0:2]
#
#    space = np.linspace(-5/scales[2], 5/scales[2], round(11/scales[2]), endpoint=True)
#    pattern = (np.sinc((space*scales[2])/15))**(20)
#
#    if param.contrast == 'T1':
#        mean_distance = mean_distance[level_start-1:len(mean_distance)]
#        xmax_pattern = np.argmax(pattern)
#    else:
#        mean_distance = mean_distance[level_start+1:len(mean_distance)]
#        xmax_pattern = np.argmin(pattern)
#        pixend = len(pattern) - xmax_pattern
#
#    #==================================================
#    # step 1 : Find the First Peak
#    #==================================================
#    sct.printv('\nFinding the First Peak...',verbose)
#    pattern1 =  np.concatenate((pattern,np.zeros(len(I_detrend[:,0])-len(pattern))))
#
#    #correlation between the pattern and the intensity profile
#    corr_all = scipy.signal.correlate(I_detrend[:,0],pattern1)
#
#    #finding the maxima of the correlation
#    loc_corr = np.arange(-np.round((len(corr_all)/2)),np.round(len(corr_all)/2)+2)
#    index_fp = 0
#    count = 0
#    for i in range(len(corr_all)):
#        if corr_all[i]>0.1:
#            if i==0:
#                if corr_all[i]<corr_all[i+1]:
#                    index_fp = i
#                    count = count + 1
#            elif i==(len(corr_all)-1):
#                if corr_all[i]<corr_all[i-1]:
#                    index_fp = np.resize(index_fp,count+1)
#                    index_fp[len(index_fp)-1] = i
#            else:
#                if corr_all[i]<corr_all[i+1]:
#                    index_fp = np.resize(index_fp,count+1)
#                    index_fp[len(index_fp)-1] = i
#                    count = count + 1
#                elif corr_all[i]<corr_all[i-1]:
#                    index_fp = np.resize(index_fp,count+1)
#                    index_fp[len(index_fp)-1] = i
#                    count = count + 1
#        else:
#            if i==0:
#                index_fp = i
#                count = count + 1
#            else:
#                index_fp = np.resize(index_fp,count+1)
#                index_fp[len(index_fp)-1] = i
#                count = count + 1                
#
#
#    mask_fp = np.ones(len(corr_all), dtype=bool)
#    mask_fp[index_fp] = False
#    value = corr_all[mask_fp]
#    loc_corr = loc_corr[mask_fp]    
#
#    loc_corr = loc_corr - I_detrend.shape[0]
#
#    if param.contrast == 'T1':
#        loc_first_peak = xmax_pattern - loc_corr[np.amax(np.where(value>1))]
#        Mcorr1 = value[np.amax(np.where(value>1))]
#
#        #building the pattern that has to be added at each iteration in step 2
#        if xmax_pattern<loc_first_peak:
#            template_truncated = np.concatenate((np.zeros((loc_first_peak-xmax_pattern)),pattern))
#        else:
#            template_truncated = pattern[(xmax_pattern-loc_first_peak-1):]
#
#        xend = np.amax(np.where(template_truncated>0.02))
#        pixend = xend - loc_first_peak
#        parameter = 0.15
#    else:
#        loc_first_peak = xmax_pattern - loc_corr[np.amax(np.where(value>0.6))]
#        Mcorr1 = value[np.amax(np.where(value>0.6))]
#
#        #building the pattern that has to be added at each iteration in step 2
#        if loc_first_peak>=0:
#            template_truncated = pattern[(loc_first_peak+1):]
#        else:
#            template_truncated = np.concatenate((np.zeros(abs(loc_first_peak)),pattern))
#
#        xend = len(template_truncated)
#
#        # smoothing the intensity curve----
#        I_detrend[:,0] = scipy.ndimage.filters.gaussian_filter1d(I_detrend[:,0],10)
#        parameter = 0.05
#        
#    if param.plot_graph:
#        pl.plot(template_truncated)
#        pl.plot(I_detrend)
#        pl.title('Detection of First Peak')
#        pl.xlabel('direction anterior-posterior (mm)')
#        pl.ylabel('intensity')
#        pl.show()
#        
#    loc_peak_I = np.arange(len(I_detrend[:,0]))
#    count = 0
#    index_p = 0
#    for i in range(len(I_detrend[:,0])):
#        if I_detrend[i]>parameter:
#            if i==0:
#                if I_detrend[i,0]<I_detrend[i+1,0]:
#                    index_p = i
#                    count  =  count + 1
#            elif i==(len(I_detrend[:,0])-1):
#                if I_detrend[i,0]<I_detrend[i-1,0]:
#                    index_p = np.resize(index_p,count+1)
#                    index_p[len(index_p)-1] = i                
#            else:
#                if I_detrend[i,0]<I_detrend[i+1,0]:
#                    index_p = np.resize(index_p,count+1)
#                    index_p[len(index_p)-1] = i
#                    count = count+1
#                elif I_detrend[i,0]<I_detrend[i-1,0]:
#                    index_p = np.resize(index_p,count+1)
#                    index_p[len(index_p)-1] = i
#                    count = count+1
#        else:
#            if i==0:
#                index_p = i
#                count  =  count + 1
#            else:
#                index_p = np.resize(index_p,count+1)
#                index_p[len(index_p)-1] = i
#                count = count+1
#
#    mask_p = np.ones(len(I_detrend[:,0]), dtype=bool)
#    mask_p[index_p] = False
#    value_I = I_detrend[mask_p]
#    loc_peak_I = loc_peak_I[mask_p]
#    
#    count = 0
#    index = []
#    for i in range(len(loc_peak_I)-1):
#        if i==0:
#            if loc_peak_I[i+1]-loc_peak_I[i]<round(10/scales[1]):
#                index = i
#                count = count + 1
#        else:
#            if (loc_peak_I[i+1]-loc_peak_I[i])<round(10/scales[1]):
#                index =  np.resize(index,count+1)
#                index[len(index)-1] = i
#                count = count + 1
#            elif (loc_peak_I[i]-loc_peak_I[i-1])<round(10/scales[1]):
#                index =  np.resize(index,count+1)
#                index[len(index)-1] = i
#                count = count + 1
#
#    mask_I = np.ones(len(value_I), dtype=bool)
#    mask_I[index] = False
#
#    if param.contrast == 'T1':
#        value_I = value_I[mask_I]
#    else:
#        value_I = -value_I[mask_I]
#
#
#
#    loc_peak_I = loc_peak_I[mask_I]
#
#    #fitting the roughly found maxima with a smoothing spline
#    from scipy.interpolate import UnivariateSpline
#    fit = UnivariateSpline(loc_peak_I,value_I)
#    P = fit(np.arange(len(I_detrend)))
#
#    for i in range(len(I_detrend)):
#        if P[i]>0.1:
#            I_detrend[i,0] = I_detrend[i,0]/P[i]
#
#    if param.plot_graph:
#        pl.xlim(0,len(I_detrend)-1)
#        pl.plot(loc_peak_I,value_I)
#        pl.plot(I_detrend)
#        pl.plot(P,color='y')
#        pl.title('Setting values of peaks at one by fitting a smoothing spline')
#        pl.xlabel('direction superior-inferior (mm)')
#        pl.ylabel('normalized intensity')
#        pl.show(block=False)
#
#    #=====================================================================================
#    # step 2 : Cross correlation between the adjusted template and the intensity profile.
#    #          Local moving of template's peak from the first peak already found
#    #=====================================================================================
#    
#    #For each loop, a peak is located at the most likely postion and then local adjustement is done.
#    #The position of the next peak is calculated from previous positions
#
#    sct.printv('\nFinding Cross correlation between the adjusted template and the intensity profile...',verbose)
#    mean_distance_new = mean_distance
#    mean_ratio = np.zeros(len(mean_distance))
#    L = np.round(1.2*max(mean_distance)) - np.round(0.8*min(mean_distance))
#    corr_peak  = np.zeros((L,len(mean_distance)))          # corr_peak  = np.nan #for T2
#
#    #loop on each peak
#    for i_peak in range(len(mean_distance)):
#        scale_min = np.round(0.80*mean_distance_new[i_peak]) - xmax_pattern - pixend
#        if scale_min<0:
#            scale_min = 0
#
#        scale_max = np.round(1.2*mean_distance_new[i_peak]) - xmax_pattern - pixend
#        scale_peak = np.arange(scale_min,scale_max+1)
#
#        for i_scale in range(len(scale_peak)):
#            template_resize_peak = np.concatenate([template_truncated,np.zeros(scale_peak[i_scale]),pattern])
#            if len(I_detrend[:,0])>len(template_resize_peak):
#                template_resize_peak1 = np.concatenate((template_resize_peak,np.zeros(len(I_detrend[:,0])-len(template_resize_peak))))
#            
#            #cross correlation
#            corr_template = scipy.signal.correlate(I_detrend[:,0],template_resize_peak)
#
#            if len(I_detrend[:,0])>len(template_resize_peak):
#                val = np.dot(I_detrend[:,0],template_resize_peak1.T)
#            else:
#                I_detrend_2 = np.concatenate((I_detrend[:,0],np.zeros(len(template_resize_peak)-len(I_detrend[:,0]))))
#                val = np.dot(I_detrend_2,template_resize_peak.T)
#            corr_peak[i_scale,i_peak] = val
#
#            if param.plot_graph:
#                pl.xlim(0,len(I_detrend[:,0]))
#                pl.plot(I_detrend[:,0])
#                pl.plot(template_resize_peak)
#                pl.show(block=False)
#                
#                pl.plot(corr_peak[:,i_peak],marker='+',linestyle='None',color='r')
#                pl.title('correlation value against the displacement of the peak (px)')
#                pl.show(block=False)
#
#        max_peak = np.amax(corr_peak[:,i_peak])
#        index_scale_peak = np.where(corr_peak[:,i_peak]==max_peak)
#        good_scale_peak = scale_peak[index_scale_peak][0]
#        Mcorr = Mcorr1
#        Mcorr = np.resize(Mcorr,i_peak+2)
#        Mcorr[i_peak+1] = np.amax(corr_peak[:,0:(i_peak+1)])
#        flag = 0
#
#        #If the correlation coefficient is too low, put the peak at the mean position
#        if i_peak>0:
#            if (Mcorr[i_peak+1]-Mcorr[i_peak])<0.4*np.mean(Mcorr[1:i_peak+2]-Mcorr[0:i_peak+1]):
#                test = i_peak
#                template_resize_peak = np.concatenate((template_truncated,np.zeros(round(mean_distance[i_peak])-xmax_pattern-pixend),pattern))
#                good_scale_peak = np.round(mean_distance[i_peak]) - xmax_pattern - pixend
#                flag = 1
#        if i_peak==0:
#            if (Mcorr[i_peak+1] - Mcorr[i_peak])<0.4*Mcorr[0]:
#                template_resize_peak = np.concatenate((template_truncated,np.zeros(round(mean_distance[i_peak])-xmax_pattern-pixend),pattern))
#                good_scale_peak = round(mean_distance[i_peak]) - xmax_pattern - pixend
#                flag = 1
#        if flag==0:
#            template_resize_peak=np.concatenate((template_truncated,np.zeros(good_scale_peak),pattern))
#
#        #update mean-distance by a adjustement ratio 
#        mean_distance_new[i_peak] = good_scale_peak + xmax_pattern + pixend
#        mean_ratio[i_peak] = np.mean(mean_distance_new[:,0:i_peak]/mean_distance[:,0:i_peak])
#
#        template_truncated = template_resize_peak
#
#        if param.plot_graph:
#            pl.plot(I_detrend[:,0])
#            pl.plot(template_truncated)
#            pl.xlim(0,(len(I_detrend[:,0])-1))
#            pl.show()
#
#    #finding the maxima of the adjusted template
#    minpeakvalue = 0.5
#    loc_disk = np.arange(len(template_truncated))
#    count = 0
#    index_disk = 0
#    for i in range(len(template_truncated)):
#        if template_truncated[i]>=minpeakvalue:
#            if i==0:
#                if template_truncated[i]<template_truncated[i+1]:
#                    index_disk = i
#                    count  =  count + 1
#            elif i==(len(template_truncated)-1):
#                if template_truncated[i]<template_truncated[i-1]:
#                    index_disk = np.resize(index_disk,count+1)
#                    index_disk[len(index_disk)-1] = i
#            else:
#                if template_truncated[i]<template_truncated[i+1]:
#                    index_disk = np.resize(index_disk,count+1)
#                    index_disk[len(index_disk)-1] = i
#                    count = count+1
#                elif template_truncated[i]<template_truncated[i-1]:
#                    index_disk = np.resize(index_disk,count+1)
#                    index_disk[len(index_disk)-1] = i
#                    count = count+1
#        else:
#            if i==0:
#                index_disk = i
#                count  =  count + 1
#            else:
#                index_disk = np.resize(index_disk,count+1)
#                index_disk[len(index_disk)-1] = i
#                count = count+1
#                
#    mask_disk = np.ones(len(template_truncated), dtype=bool)
#    mask_disk[index_disk] = False
#    loc_disk = loc_disk[mask_disk]
#    X1 = np.where(loc_disk > I_detrend.shape[0])
#    mask_disk1 = np.ones(len(loc_disk), dtype=bool)
#    mask_disk1[X1] = False
#    loc_disk = loc_disk[mask_disk1]
#    loc_disk = loc_disk + start_centerline_y - 1
#
#    #=====================================================================
#    # Step 3: Building the labeled centerline and surface
#    #=====================================================================
#    sct.printv('\nBuilding the labeled centerline and surface... ',verbose)
#    
#    #orthogonal projection of the position of disk centers on the spinal cord center line
#    for i in range(len(loc_disk)):
#
#        #find which index of y matches with the disk
#        Index = np.array(np.where(y==loc_disk[i])).T
#        lim_plus = Index + 5
#        lim_minus = Index - 5
#
#        if lim_minus<1: lim_minus=1
#        if lim_plus>len(x): lim_plus=len(x)
#
#        #tangent vector to the centerline
#        Vx = x[lim_plus] - x[lim_minus]
#        Vz = z[lim_plus] - z[lim_minus]
#        Vy = y[lim_plus] - y[lim_minus]
#
#        d = Vx*x[Index] + Vy*y[Index] + Vz*z[Index]
#
#        intersection = np.ones(len(x))
#        for j in range(len(x)):
#            intersection[j] = np.abs((Vx*x[j]+Vy*y[j]+Vz*z[j]-d))
#        
#        min_intersection = np.amin(intersection)
#        index_intersection = np.array(np.where(min_intersection==intersection)).T
#        loc_disk[i] = y[index_intersection[0]]
#        
#    center_disk = centerline
#    for i in range(len(loc_disk)-1):
#        tmp = center_disk[:,loc_disk[i]:loc_disk[i+1],:]
#        tmp[np.where(tmp==1)] = i + level_start
#        center_disk[:,loc_disk[i]:loc_disk[i+1],:] = tmp
#    center_disk[np.where(center_disk==1)] = 0
#
#    #add C1 and C2
#    if level_start==2:
#        center_disk[x[0],(int(round(loc_disk[0] - C1C2_distance[1]))-1):loc_disk[0],z[0]] = 2
#        center_disk[x[0],(int(round(loc_disk[0] - C1C2_distance[0] - C1C2_distance[1]))-1):(round(loc_disk[0] - C1C2_distance[1])-1),z[0]] = 1
#
#    xc,yc,zc = np.where(center_disk>0)
#    
#    # Write NIFTI volumes
#    hdr.set_data_dtype('uint8') # set imagetype to uint8
#    sct.printv('\nWrite NIFTI volumes...',verbose)
#    img = nibabel.Nifti1Image(center_disk, None, hdr)
#    file_name = param.output_path + param.contrast + '_centerline.nii.gz'
#    nibabel.save(img,file_name)
#    sct.printv(('.. File created:' + file_name),verbose)
#
#    #generating output file
#    if ext_data == '.nii.gz':
#        os.system('fslchfiletype NIFTI_GZ '+ param.output_path + param.contrast + '_centerline.nii.gz')
#
#    # display elapsed time
#    elapsed_time = time.time() - start_time
#    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'

#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'USAGE: \n' \
        '  '+os.path.basename(__file__)+' -i <filename> -c <contrast> -l <centerline_binary_image> -m <mean_distance.mat> \n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        '  -i           input_file \n' \
        '  -c           contrast \n' \
        '  -l           Centerline binary Image  \n' \
        '  -m           mean_distance.mat \n' \
        '\n'\
        'OPTIONAL ARGUMENTS\n' \
        '  -o           Specify Output path.\n' \
        '  -a           shift_AP in mm. Default value is 17mm \n' \
        '  -s           size_AP in mm. Default value is 6mm \n' \
        '  -r           size_RL in mm. Default value is 5mm \n' \
        '  -v {0,1}     Set verbose=1 for printing text. Default value is 0 \n' \
        '  -g {0,1}     Set value to 1 for plotting graphs. Default value is 0 \n' \
        '  -h           help. Show this message.\n' \
        '\n'\
        'EXAMPLE:\n' \
        '  '+os.path.basename(__file__)+' -i t1.nii -c T1 -l segmentation_centerline_binary.nii -m mean_distance.mat\n'
    sys.exit(2)

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()    