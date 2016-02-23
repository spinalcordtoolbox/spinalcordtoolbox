#!/usr/bin/env python

# check if needed Python libraries are already installed or not
import os
import getopt
import commands
import math
import sys
import scipy
import scipy.signal
import scipy.fftpack
import pylab as pl

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

class label_class:
    def __init__(self):
    
        # PATH AND FILE NAME FOR ANATOMICAL IMAGE
        self.input_path                = ''
        self.input_anat                = ''
        self.contrast                  = ''
        # PATH FOR OUTPUT
        self.output_labled_centerline  = '_centerline'
        self.output_labled_surface     = ''                                       # optional
        self.mean_distance_mat         = ''
        
        # =======================================================
        # Spinal Cord Segmentation Parameters

        self.segmentation_do           = 0
        self.segmentation_centerline   = 'T2_errsm08_centerline'
        self.segmentation_surface      = 'T2_errsm08_surface'
        self.segmentation_straightened = 'T2_errsm08_straightened'

        # Spinal Cord labeling Parameters
        self.input_centerline          = ''                                        # optional
        self.input_surface             = ''                                        # optional
        self.shift_AP                  = 17                                        # shift the centerline on the spine in mm default : 17 mm
        self.size_AP                   = 6                                         # mean around the centerline in the anterior-posterior direction in mm
        self.size_RL                   = 5                                         # mean around the centerline in the right-left direction in mm
        
        # =======================================================
        # OTHER PARAMETERS
        self.surface_do                = 1

        self.verbose                   = 1                                         # display figures
        self.segmentation_image_type   = 1
        
            
#=======================================================================================================================
# main
#=======================================================================================================================

def main():
    
    print '\n\n\n==================================================='
    print '               Running: sct_labeling'
    print '===================================================\n'
    
    # Initialization
    start_time = time.time()
    label = label_class()
    
    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:c:l:f:m:o:a:s:r:v:')
    except getopt.GetoptError as err:
        print str(err)
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            label.input_anat = arg
        elif opt in ('-c'):
            label.contrast = arg
        elif opt in ('-l'):
            label.input_centerline = arg
        elif opt in ('-f'):
            label.input_surface = arg
        elif opt in ('-m'):
            label.mean_distance_mat = arg
        elif opt in ('-o'):
            label.output_labled_surface = arg
        elif opt in ('-a'):
            label.shift_AP = int(arg)
        elif opt in ('-s'):
            label.size_AP = int(arg)
        elif opt in ('-r'):
            label.size_RL = int(arg)
        elif opt in ('-v'):
            label.verbose = int(arg)

    # Display usage if a mandatory argument is not provided
    if label.input_anat == '' or label.contrast=='' or label.input_centerline=='' or label.mean_distance_mat=='':
        print '\n \n All mandatory arguments are not provided \n \n'
        usage()

    if label.contrast=='T1':
        label.segmentation_image_type = 1
    else:
        label.segmentation_image_type = 2

    # Extract path, file and extension
    label.input_path, file_data, ext_data = sct.extract_fname(label.input_anat)

    input_anat = label.input_anat
    if label.segmentation_do==1:
        input_centerline = label.input_path + label.segmentation_centerline + '.nii'
        input_surface = label.input_path + label.segmentation_surface + '.nii'
    else:
        input_centerline = label.input_centerline
        input_surface = label.input_surface

    # check existence of input files
    sct.check_file_exist(input_anat)
    sct.check_file_exist(label.mean_distance_mat)

    print 'Input File:',input_anat
    print 'Center_line file:', input_centerline
    print 'Contrast:',label.contrast
    print 'Mat File:',label.mean_distance_mat

    output_centerline_vertebra = label.input_path + label.contrast + label.output_labled_centerline + '.nii'
    output_surface_vertebra = label.input_path + label.output_labled_surface + '.nii.gz'
    surface_do = label.surface_do

    if label.contrast == 'T1':
        labeling_vertebrae_T1(label,input_anat,input_centerline,input_surface,output_centerline_vertebra,output_surface_vertebra,surface_do)
    else:
        labeling_vertebrae_T2(label,input_anat,input_centerline,input_surface,output_centerline_vertebra,output_surface_vertebra,surface_do)

    # display elapsed time
    elapsed_time = time.time() - start_time
    print '\nFinished! Elapsed time: '+str(int(round(elapsed_time)))+'s'

#=======================================================================================================================
# labeling_vertebrae_T1 function
#=======================================================================================================================

def labeling_vertebrae_T1(label,input_anat,input_centerline,input_surface,output_centerline_vertebra,output_surface_vertebra,surface_do):


    #==================================================
    # Reorientation of the data if needed
    #==================================================
    command = 'fslhd ' + input_anat
    result = commands.getoutput(command)
    orientation = result[result.find('qform_xorient')+15] + result[result.find('qform_yorient')+15] + result[result.find('qform_zorient')+15]

    if orientation!='ASR':
        
        print '\nReorient input volume to AP SI RL orientation...'
        sct.run(sct.fsloutput + 'fslswapdim ' + input_anat + ' AP SI RL ' + label.input_path + '/tmp.anat_orient')
        
        sct.run(sct.fsloutput + 'fslswapdim ' + input_centerline + ' AP SI RL ' + label.input_path + '/tmp.centerline_orient')

                
        #loading images
        print '\nLoading Images...'
        anat_file = nibabel.load(label.input_path + '/tmp.anat_orient')
        anat = anat_file.get_data()
        hdr = anat_file.get_header()
        dims = hdr['dim']
        scales = hdr['pixdim']
        #if surface_do==1:
            #surface_file = nibabel.load(input_surface_reorient)
            #surface = surface_file.get_data()
                
        centerline_file = nibabel.load(label.input_path + '/tmp.centerline_orient')
        centerline = centerline_file.get_data()
            
    else:
        #loading images
        print '\nLoading Images...'
        anat_file = nibabel.load(input_anat)
        anat = anat_file.get_data()
        hdr = anat_file.get_header()
        dims = hdr['dim']
        scales = hdr['pixdim']
    
        #if surface_do==1:
            #surface_file = nibabel.load(input_surface)
            #surface = surface_file.get_data()

        centerline_file = nibabel.load(input_centerline)
        centerline = centerline_file.get_data()    

    #==================================================
    # Calculation of the profile intensity
    #==================================================
    print '\nCalculation of Profile Intensity...'
    shift_AP = label.shift_AP*scales[1]
    size_AP = label.size_AP*scales[1]
    size_RL = label.size_RL*scales[3]
    
    np.uint16(anat)      

    #find coordinates of the centerline
    X,Y,Z = np.where(centerline>0)
    j = np.argsort(Y)
    y = Y[j]
    x = X[j]
    z = Z[j]

    #eliminating double in y
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

        if lim_minus<0:
            lim_minus = 0
        if lim_plus>=len(x1):
            lim_plus = len(x1) - 1

        # normal vector of the orthogonal plane to the centerline i.e tangent vector to the centerline
        Vx = x1[lim_plus] - x1[lim_minus]
        Vz = z[lim_plus] - z[lim_minus]
        Vy = y[lim_plus] - y[lim_minus]

        d = Vx*x1[index] + Vy*y[index] + Vz*z[index]

        for i_slice_RL in range(2*np.int(round(size_RL/scales[3]))):
            for i_slice_AP in range(2*np.int(round(size_AP/scales[1]))):
                result = (d - Vx*(x1[index] + i_slice_AP - size_AP - 1) - Vz*z[index])/Vy
                                
                if result > anat.shape[1]:
                    result = anat.shape[1]
                I[index] = I[index] + anat[np.int(round(x1[index]+i_slice_AP - size_AP - 1)),np.int(round(result)),np.int(round(z[index] + i_slice_RL - size_RL - 1))]

    # Detrending Intensity
    print '\nDetrending Intensity...'
    start_centerline_y = y[0]
    X = np.where(I==0)
    mask2 = np.ones((len(y),1), dtype=bool)
    mask2[X,0] = False
    #I = I[mask2]

    if label.verbose==1:
        pl.plot(I)
        pl.xlabel('direction superior-inferior')
        pl.ylabel('intensity')
        pl.title('Intensity profile along the shifted spinal cord centerline')
        pl.show()

    frequency = scipy.fftpack.fftfreq(len(I[:,0]), d=1)
    spectrum = np.abs(scipy.fftpack.fft(I[:,0], n=None, axis=-1, overwrite_x=False))

    Wn = np.amax(frequency)/10
    N = 5              #Order of the filter
    b, a = scipy.signal.iirfilter(N, Wn, rp=None, rs=None, btype='low', analog=False, ftype='bessel', output='ba')
    I_fit = scipy.signal.filtfilt(b, a, I[:,0], axis=-1, padtype='constant', padlen=None)
    if label.verbose==1:
        pl.plot(I[:,0])
        pl.plot(I_fit)
        pl.show()

    I_detrend = np.zeros((len(I[:,0]),1))
    I_detrend[:,0] = I[:,0] - I_fit
    
    I_detrend = I_detrend/(np.amax(I_detrend))
    if label.verbose==1:
        pl.plot(I_detrend[:,0])
        pl.xlabel('direction superior-inferior')
        pl.ylabel('intensity')
        pl.title('Intensity profile along the shifted spinal cord centerline after detrending and basic normalization')
        pl.show()

    info_1 = input('\nIs the more rostral vertebrae the C1 or C2 one? if yes, enter 1 otherwise 0:')
    if info_1==0:
        level_start = input('enter the level of the more rostral vertebra - choice of the more rostral vertebral level of the field of view:')
    else:
        level_start = 2

    mean_distance_dict = scipy.io.loadmat(label.mean_distance_mat)
    mean_distance = (mean_distance_dict.values()[2]).T
    C1C2_distance = mean_distance[0:2]
    mean_distance = mean_distance[level_start-1:len(mean_distance)]

    space = np.linspace(-5/scales[2], 5/scales[2], round(11/scales[2]), endpoint=True)
    pattern = (np.sinc((space*scales[2])/15))**(20)
    xmax_pattern = np.argmax(pattern)

    #==================================================
    # step 1 : Find the First Peak
    #==================================================
    print '\nFinding the First Peak...'
    pattern1 =  np.concatenate((pattern,np.zeros(len(I_detrend[:,0])-len(pattern))))

    #correlation between the pattern and the intensity profile
    corr_all = scipy.signal.correlate(I_detrend[:,0],pattern1)
    loc_corr = np.arange(-np.round((len(corr_all)/2)),np.round(len(corr_all)/2)+2)
    index_fp = 0
    count = 0
    for i in range(len(corr_all)):
        if corr_all[i]>0.1:
            if i==0:
                if corr_all[i]<corr_all[i+1]:
                    index_fp = i
                    count = count + 1
            elif i==(len(corr_all)-1):
                if corr_all[i]<corr_all[i-1]:
                    index_fp = np.resize(index_fp,count+1)
                    index_fp[len(index_fp)-1] = i
            else:
                if corr_all[i]<corr_all[i+1]:
                    index_fp = np.resize(index_fp,count+1)
                    index_fp[len(index_fp)-1] = i
                    count = count + 1
                elif corr_all[i]<corr_all[i-1]:
                    index_fp = np.resize(index_fp,count+1)
                    index_fp[len(index_fp)-1] = i
                    count = count + 1
        else:
            if i==0:
                index_fp = i
                count = count + 1
            else:
                index_fp = np.resize(index_fp,count+1)
                index_fp[len(index_fp)-1] = i
                count = count + 1                


    mask_fp = np.ones(len(corr_all), dtype=bool)
    mask_fp[index_fp] = False
    value = corr_all[mask_fp]
    loc_corr = loc_corr[mask_fp]    

    loc_corr = loc_corr - I_detrend.shape[0]
    loc_first_peak = xmax_pattern - loc_corr[np.amax(np.where(value>1))]
    Mcorr1 = value[np.amax(np.where(value>1))]

    #building the pattern that has to be added at each iteration in step 2

    if xmax_pattern<loc_first_peak:
        template_truncated = np.concatenate((np.zeros((loc_first_peak-xmax_pattern)),pattern))
        
    else:
        template_truncated = pattern[(xmax_pattern-loc_first_peak-1):]
    xend = np.amax(np.where(template_truncated>0.02))
    pixend = xend - loc_first_peak

    if label.verbose==1:
        pl.plot(template_truncated)
        pl.plot(I_detrend)
        pl.title('Detection of First Peak')
        pl.xlabel('direction anterior-posterior (mm)')
        pl.ylabel('intensity')
        pl.show()

    loc_peak_I = np.arange(len(I_detrend[:,0]))
    count = 0
    index_p = 0
    for i in range(len(I_detrend[:,0])):
        if I_detrend[i]>0.15:
            if i==0:
                if I_detrend[i,0]<I_detrend[i+1,0]:
                    index_p = i
                    count  =  count + 1
            elif i==(len(I_detrend[:,0])-1):
                if I_detrend[i,0]<I_detrend[i-1,0]:
                    index_p = np.resize(index_p,count+1)
                    index_p[len(index_p)-1] = i                
            else:
                if I_detrend[i,0]<I_detrend[i+1,0]:
                    index_p = np.resize(index_p,count+1)
                    index_p[len(index_p)-1] = i
                    count = count+1
                elif I_detrend[i,0]<I_detrend[i-1,0]:
                    index_p = np.resize(index_p,count+1)
                    index_p[len(index_p)-1] = i
                    count = count+1
        else:
            if i==0:
                index_p = i
                count  =  count + 1
            else:
                index_p = np.resize(index_p,count+1)
                index_p[len(index_p)-1] = i
                count = count+1

    mask_p = np.ones(len(I_detrend[:,0]), dtype=bool)
    mask_p[index_p] = False
    value_I = I_detrend[mask_p]
    loc_peak_I = loc_peak_I[mask_p]
    
    count = 0
    for i in range(len(loc_peak_I)-1):
        if i==0:
            if loc_peak_I[i+1]-loc_peak_I[i]<round(10/scales[1]):
                index = i
                count = count + 1
        else:
            if (loc_peak_I[i+1]-loc_peak_I[i])<round(10/scales[1]):
                index =  np.resize(index,count+1)
                index[len(index)-1] = i
                count = count + 1
            elif (loc_peak_I[i]-loc_peak_I[i-1])<round(10/scales[1]):
                index =  np.resize(index,count+1)
                index[len(index)-1] = i
                count = count + 1

    mask_I = np.ones(len(value_I), dtype=bool)
    mask_I[index] = False
    value_I = value_I[mask_I]
    loc_peak_I = loc_peak_I[mask_I]

    from scipy.interpolate import UnivariateSpline
    fit = UnivariateSpline(loc_peak_I,value_I)
    P = fit(np.arange(len(I_detrend)))

    for i in range(len(I_detrend)):
        if P[i]>0.1:
            I_detrend[i,0] = I_detrend[i,0]/P[i]

    if label.verbose==1:
        pl.xlim(0,len(I_detrend)-1)
        pl.plot(loc_peak_I,value_I)
        pl.plot(I_detrend)
        pl.plot(P,color='y')
        pl.title('Setting values of peaks at one by fitting a smoothing spline')
        pl.xlabel('direction superior-inferior (mm)')
        pl.ylabel('normalized intensity')
        pl.show(block=False)

    #===================================================================================
    # step 2 : Cross correlation between the adjusted template and the intensity profile
    #          local moving of template's peak from the first peak already found
    #===================================================================================
    print '\nFinding Cross correlation between the adjusted template and the intensity profile...'
    mean_distance_new = mean_distance
    mean_ratio = np.zeros(len(mean_distance))
    L = np.round(1.2*max(mean_distance)) - np.round(0.8*min(mean_distance))
    corr_peak  = np.zeros((L,len(mean_distance)))

    for i_peak in range(len(mean_distance)):
        scale_min = np.round(0.80*mean_distance_new[i_peak]) - xmax_pattern - pixend
        if scale_min<0:
            scale_min = 0

        scale_max = np.round(1.2*mean_distance_new[i_peak]) - xmax_pattern - pixend
        scale_peak = np.arange(scale_min,scale_max+1)

        for i_scale in range(len(scale_peak)):
            template_resize_peak = np.concatenate([template_truncated,np.zeros(scale_peak[i_scale]),pattern])
            if len(I_detrend[:,0])>len(template_resize_peak):
                template_resize_peak1 = np.concatenate((template_resize_peak,np.zeros(len(I_detrend[:,0])-len(template_resize_peak))))
            corr_template = scipy.signal.correlate(I_detrend[:,0],template_resize_peak)

            if len(I_detrend[:,0])>len(template_resize_peak):
                val = np.dot(I_detrend[:,0],template_resize_peak1.T)
            else:
                I_detrend_2 = np.concatenate((I_detrend[:,0],np.zeros(len(template_resize_peak)-len(I_detrend[:,0]))))
                val = np.dot(I_detrend_2,template_resize_peak.T)
            corr_peak[i_scale,i_peak] = val

            if label.verbose==1:
                pl.xlim(0,len(I_detrend[:,0]))
                pl.plot(I_detrend[:,0])
                pl.plot(template_resize_peak)
                pl.show(block=False)
                
                pl.plot(corr_peak[:,i_peak],marker='+',linestyle='None',color='r')
                pl.title('correlation value against the displacement of the peak (px)')
                pl.show(block=False)

        max_peak = np.amax(corr_peak[:,i_peak])
        index_scale_peak = np.where(corr_peak[:,i_peak]==max_peak)
        good_scale_peak = scale_peak[index_scale_peak][0]
        Mcorr = Mcorr1
        Mcorr = np.resize(Mcorr,i_peak+2)
        Mcorr[i_peak+1] = np.amax(corr_peak[:,0:(i_peak+1)])
        flag = 0

        if i_peak>0:
            if (Mcorr[i_peak+1]-Mcorr[i_peak])<0.4*np.mean(Mcorr[1:i_peak+2]-Mcorr[0:i_peak+1]):
                test = i_peak
                template_resize_peak = np.concatenate((template_truncated,np.zeros(round(mean_distance[i_peak])-xmax_pattern-pixend),pattern))
                good_scale_peak = np.round(mean_distance[i_peak]) - xmax_pattern - pixend
                flag = 1
        if i_peak==0:
            if (Mcorr[i_peak+1] - Mcorr[i_peak])<0.4*Mcorr[0]:
                template_resize_peak = np.concatenate((template_truncated,np.zeros(round(mean_distance[i_peak])-xmax_pattern-pixend),pattern))
                good_scale_peak = round(mean_distance[i_peak]) - xmax_pattern - pixend
                flag = 1
        if flag==0:
            template_resize_peak=np.concatenate((template_truncated,np.zeros(good_scale_peak),pattern))

        mean_distance_new[i_peak] = good_scale_peak + xmax_pattern + pixend
        mean_ratio[i_peak] = np.mean(mean_distance_new[:,0:i_peak]/mean_distance[:,0:i_peak])

        template_truncated = template_resize_peak

        if label.verbose==1:
            pl.plot(I_detrend[:,0])
            pl.plot(template_truncated)
            pl.xlim(0,(len(I_detrend[:,0])-1))
            pl.show()

    minpeakvalue = 0.5
    loc_disk = np.arange(len(template_truncated))
    count = 0
    index_disk = 0
    for i in range(len(template_truncated)):
        if template_truncated[i]>=minpeakvalue:
            if i==0:
                if template_truncated[i]<template_truncated[i+1]:
                    index_disk = i
                    count  =  count + 1
            elif i==(len(template_truncated)-1):
                if template_truncated[i]<template_truncated[i-1]:
                    index_disk = np.resize(index_disk,count+1)
                    index_disk[len(index_disk)-1] = i
            else:
                if template_truncated[i]<template_truncated[i+1]:
                    index_disk = np.resize(index_disk,count+1)
                    index_disk[len(index_disk)-1] = i
                    count = count+1
                elif template_truncated[i]<template_truncated[i-1]:
                    index_disk = np.resize(index_disk,count+1)
                    index_disk[len(index_disk)-1] = i
                    count = count+1
        else:
            if i==0:
                index_disk = i
                count  =  count + 1
            else:
                index_disk = np.resize(index_disk,count+1)
                index_disk[len(index_disk)-1] = i
                count = count+1
                
    mask_disk = np.ones(len(template_truncated), dtype=bool)
    mask_disk[index_disk] = False
    loc_disk = loc_disk[mask_disk]
    X1 = np.where(loc_disk > I_detrend.shape[0])
    mask_disk1 = np.ones(len(loc_disk), dtype=bool)
    mask_disk1[X1] = False
    loc_disk = loc_disk[mask_disk1]
    loc_disk = loc_disk + start_centerline_y - 1

    #=====================================================================
    # Step 3: Building the labeled centerline and surface
    #=====================================================================
    print '\nBuilding the labeled centerline and surface... '
    for i in range(len(loc_disk)):

        Index = np.array(np.where(y==loc_disk[i])).T
        lim_plus = Index + 5
        lim_minus = Index - 5

        if lim_minus<1: lim_minus=1
        if lim_plus>len(x): lim_plus=len(x)

        Vx = x[lim_plus] - x[lim_minus]
        Vz = z[lim_plus] - z[lim_minus]
        Vy = y[lim_plus] - y[lim_minus]

        d = Vx*x1[Index] + Vy*y[Index] + Vz*z[Index]

        intersection = np.ones(len(x))
        for j in range(len(x)):
            intersection[j] = np.abs((Vx*x[j]+Vy*y[j]+Vz*z[j]-d))
        
        min_intersection = np.amin(intersection)
        index_intersection = np.where(min_intersection==intersection)
        loc_disk[i] = y[index_intersection]
        
    center_disk = centerline
    for i in range(len(loc_disk)-1):
        tmp = center_disk[:,loc_disk[i]:loc_disk[i+1],:]
        tmp[np.where(tmp==1)] = i + level_start
        center_disk[:,loc_disk[i]:loc_disk[i+1],:] = tmp
    center_disk[np.where(center_disk==1)] = 0

    if level_start==2:
        center_disk[x[0],(int(round(loc_disk[0] - C1C2_distance[1]))-1):loc_disk[0],z[0]] = 2
        center_disk[x[0],(int(round(loc_disk[0] - C1C2_distance[0] - C1C2_distance[1]))-1):(round(loc_disk[0] - C1C2_distance[1])-1),z[0]] = 1

    xc,yc,zc = np.where(center_disk>0)
    print center_disk[xc,yc,zc]
    # Write NIFTI volumes
    hdr.set_data_dtype('uint8') # set imagetype to uint8
    print '\nWrite NIFTI volumes...'
    img = nibabel.Nifti1Image(center_disk, None, hdr)
    file_name = label.input_path + label.contrast + '_centerline.nii.gz'
    nibabel.save(img,file_name)
    print '.. File created:' + file_name

#=======================================================================================================================
# labeling_vertebrae_T2 function
#=======================================================================================================================
def labeling_vertebrae_T2(label,input_anat,input_centerline,input_surface,output_centerline_vertebra,output_surface_vertebra,surface_do):

    command = 'fslhd ' + input_anat
    
    result = commands.getoutput(command)
    orientation = result[result.find('qform_xorient')+15] + result[result.find('qform_yorient')+15] + result[result.find('qform_zorient')+15]
    
    if orientation!='ASR':
        
        print '\nReorient input volume to AP SI RL orientation...'
        sct.run(sct.fsloutput + 'fslswapdim tmp.anat AP SI RL ' + label.input_path + '/tmp.anat_orient')
        
        sct.run(sct.fsloutput + 'fslswapdim tmp.centerline AP SI RL ' + label.input_path + '/tmp.centerline_orient')
        
        
        #load_images
        anat_file = nibabel.load(label.input_path + '/tmp.anat_orient')
        anat = anat_file.get_data()
        hdr = anat_file.get_header()
        dims = hdr['dim']
        scales = hdr['pixdim']
        #if surface_do==1:
        #surface_file = nibabel.load(input_surface_reorient)
        #surface = surface_file.get_data()
        
        centerline_file = nibabel.load(label.input_path + '/tmp.centerline_orient')
        centerline = centerline_file.get_data()

    else:
        # loading images
        print '\nLoading Images...'
        anat_file = nibabel.load(input_anat)
        anat = anat_file.get_data()
        hdr = anat_file.get_header()
        dims = hdr['dim']
        scales = hdr['pixdim']
    
        #if surface_do==1:
            #surface_file = nibabel.load(input_surface)
            #surface = surface_file.get_data()
    
        centerline_file = nibabel.load(input_centerline)
        centerline = centerline_file.get_data()

    
    
    #==================================================
    # Calculation of the profile intensity
    #==================================================
    print '\nCalculation of the profile intensity...'
    shift_AP = label.shift_AP*scales[1]
    size_AP = label.size_AP*scales[1]
    size_RL = label.size_RL*scales[3]
    
    np.uint16(anat)

    X,Y,Z = np.where(centerline>0)

    j = np.argsort(Y)
    y = Y[j]
    x = X[j]
    z = Z[j]

    #eliminating double in y
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
        
        if lim_minus<0:
            lim_minus = 0
        if lim_plus>=len(x1):
            lim_plus = len(x1) - 1

        # normal vector of the orthogonal plane to the centerline i.e tangent vector to the centerline
        Vx = x1[lim_plus] - x1[lim_minus]
        Vz = z[lim_plus] - z[lim_minus]
        Vy = y[lim_plus] - y[lim_minus]

        d = Vx*x1[index] + Vy*y[index] + Vz*z[index]

        for i_slice_RL in range(2*np.int(round(size_RL/scales[3]))):
            for i_slice_AP in range(2*np.int(round(size_AP/scales[1]))):
                result = (d - Vx*(x1[index] + i_slice_AP - size_AP - 1) - Vz*z[index])/Vy
                
                if result > anat.shape[1]:
                    result = anat.shape[1]
                I[index] = I[index] + anat[np.int(round(x1[index]+i_slice_AP - size_AP - 1)),np.int(round(result)),np.int(round(z[index] + i_slice_RL - size_RL - 1))]

    # Detrending Intensity
    print '\nDetrending Intensity...'
    start_centerline_y = y[0]
    X = np.where(I==0)
    mask2 = np.ones((len(y),1), dtype=bool)
    mask2[X,0] = False
    #I = I[mask2]
    
    if label.verbose==1:
        pl.plot(I)
        pl.xlabel('direction superior-inferior')
        pl.ylabel('intensity')
        pl.title('Intensity profile along the shifted spinal cord centerline')
        pl.show(block=False)
                    
    frequency = scipy.fftpack.fftfreq(len(I[:,0]), d=1)
    spectrum = np.abs(scipy.fftpack.fft(I[:,0], n=None, axis=-1, overwrite_x=False))
    
    Wn = np.amax(frequency)/10
    N = 5              #Order of the filter
    b, a = scipy.signal.iirfilter(N, Wn, rp=None, rs=None, btype='low', analog=False, ftype='bessel', output='ba')
    I_fit = scipy.signal.filtfilt(b, a, I[:,0], axis=-1, padtype='constant', padlen=None)

    if label.verbose==1:
        pl.plot(I[:,0])
        pl.plot(I_fit)
        pl.show()
    
    I_detrend = np.zeros((len(I[:,0]),1))
    I_detrend[:,0] = I[:,0] - I_fit

    I_detrend = I_detrend/abs((np.amin(I_detrend)))
    if label.verbose==1:
        pl.plot(I_detrend)
        pl.xlabel('direction superior-inferior')
        pl.ylabel('intensity')
        pl.title('Intensity profile along the shifted spinal cord centerline after detrending and basic normalization')
        pl.show(block=False)
    
    info_1 = input('Is the more rostral vertebrae the C1 or C2 one? if yes, enter 1 otherwise 0:')
    if info_1==0:
        level_start = input('enter the level of the more rostral vertebra - choice of the more rostral vertebral level of the field of view:')
    else:
        level_start = 2
    
    mean_distance_dict = scipy.io.loadmat(label.mean_distance_mat)
    mean_distance = (mean_distance_dict.values()[2]).T
    C1C2_distance = mean_distance[0:2]
    mean_distance = mean_distance[level_start+1:len(mean_distance)]
    
    space = np.linspace(-5/scales[2], 5/scales[2], round(11/scales[2]), endpoint=True)
    pattern = (np.sinc((space*scales[2])/15))**(20)
    xmax_pattern = np.argmin(pattern)
    pixend = len(pattern) - xmax_pattern

    #==================================================
    # step 1 : find the first peak
    #==================================================
    print '\nFinding the First Peak...'
    pattern1 =  np.concatenate((pattern,np.zeros(len(I_detrend[:,0])-len(pattern))))
    corr_all = scipy.signal.correlate(I_detrend[:,0],pattern1)
    loc_corr = np.arange(-np.round((len(corr_all)/2)),np.round(len(corr_all)/2)+2)
    index_fp = 0
    count = 0
    for i in range(len(corr_all)):
        if corr_all[i]>0.1:
            if i==0:
                if corr_all[i]<corr_all[i+1]:
                    index_fp = i
                    count = count + 1
            elif i==(len(corr_all)-1):
                if corr_all[i]<corr_all[i-1]:
                    index_fp = np.resize(index_fp,count+1)
                    index_fp[len(index_fp)-1] = i
            else:
                if corr_all[i]<corr_all[i+1]:
                    index_fp = np.resize(index_fp,count+1)
                    index_fp[len(index_fp)-1] = i
                    count = count + 1
                elif corr_all[i]<corr_all[i-1]:
                    index_fp = np.resize(index_fp,count+1)
                    index_fp[len(index_fp)-1] = i
                    count = count + 1
        else:
            if i==0:
                index_fp = i
                count = count + 1
            else:
                index_fp = np.resize(index_fp,count+1)
                index_fp[len(index_fp)-1] = i
                count = count + 1


    mask_fp = np.ones(len(corr_all), dtype=bool)
    mask_fp[index_fp] = False
    value = corr_all[mask_fp]
    loc_corr = loc_corr[mask_fp]

    loc_corr = loc_corr - I_detrend.shape[0]

    loc_first_peak = xmax_pattern - loc_corr[np.amax(np.where(value>0.6))]
    Mcorr1 = value[np.amax(np.where(value>0.6))]

    #building the pattern that has to be added at each iteration in step 2
    if loc_first_peak>=0:
        template_truncated = pattern[(loc_first_peak+1):]
    else:
        template_truncated = np.concatenate((np.zeros(abs(loc_first_peak)),pattern))

    xend = len(template_truncated)

    if label.verbose==1:
        pl.plot(template_truncated)
        pl.plot(I_detrend)
        pl.title('Detection of First Peak')
        pl.xlabel('direction anterior-posterior (mm)')
        pl.ylabel('intensity')
        pl.show(block=False)

    # smoothing the intensity curve----
    I_detrend[:,0] = scipy.ndimage.filters.gaussian_filter1d(I_detrend[:,0],10)

    loc_peak_I = np.arange(len(I_detrend[:,0]))
    count = 0
    index_p = 0
    for i in range(len(I_detrend[:,0])):
        if I_detrend[i]>0.05:
            if i==0:
                if I_detrend[i,0]<I_detrend[i+1,0]:
                    index_p = i
                    count  =  count + 1
            elif i==(len(I_detrend[:,0])-1):
                if I_detrend[i,0]<I_detrend[i-1,0]:
                    index_p = np.resize(index_p,count+1)
                    index_p[len(index_p)-1] = i
            else:
                if I_detrend[i,0]<I_detrend[i+1,0]:
                    index_p = np.resize(index_p,count+1)
                    index_p[len(index_p)-1] = i
                    count = count+1
                elif I_detrend[i,0]<I_detrend[i-1,0]:
                    index_p = np.resize(index_p,count+1)
                    index_p[len(index_p)-1] = i
                    count = count+1
        else:
            if i==0:
                index_p = i
                count  =  count + 1
            else:
                index_p = np.resize(index_p,count+1)
                index_p[len(index_p)-1] = i
                count = count+1

    mask_p = np.ones(len(I_detrend[:,0]), dtype=bool)
    mask_p[index_p] = False
    value_I = I_detrend[mask_p]
    loc_peak_I = loc_peak_I[mask_p]

    count = 0
    for i in range(len(loc_peak_I)-1):
        if i==0:
            if loc_peak_I[i+1]-loc_peak_I[i]<round(10/scales[1]):
                index = i
                count = count + 1
        else:
            if (loc_peak_I[i+1]-loc_peak_I[i])<round(10/scales[1]):
                index =  np.resize(index,count+1)
                index[len(index)-1] = i
                count = count + 1
            elif (loc_peak_I[i]-loc_peak_I[i-1])<round(10/scales[1]):
                index =  np.resize(index,count+1)
                index[len(index)-1] = i
                count = count + 1

    mask_I = np.ones(len(value_I), dtype=bool)
    mask_I[index] = False
    value_I = -value_I[mask_I]
    loc_peak_I = loc_peak_I[mask_I]

    from scipy.interpolate import UnivariateSpline
    fit = UnivariateSpline(loc_peak_I,value_I)
    P = fit(np.arange(len(I_detrend)))

    if label.verbose==1:
        pl.xlim(0,len(I_detrend)-1)
        pl.plot(loc_peak_I,value_I)
        pl.plot(I_detrend)
        pl.plot(P)
        pl.title('Setting values of peaks at one by fitting a smoothing spline')
        pl.xlabel('direction superior-inferior (mm)')
        pl.ylabel('normalized intensity')
        pl.show(block=False)

    for i in range(len(I_detrend)):
        if P[i]>0.1:
            I_detrend[i,0] = I_detrend[i,0]/abs(P[i])

    #===================================================================================
    # step 2 : Cross correlation between the adjusted template and the intensity profile
    #          local moving of template's peak from the first peak already found
    #===================================================================================

    mean_distance_new = mean_distance
    mean_ratio = np.zeros(len(mean_distance))
    L = np.round(1.2*max(mean_distance)) - np.round(0.8*min(mean_distance))
    corr_peak  = np.nan(np.zeros((L,len(mean_distance))))

    for i_peak in range(len(mean_distance)):
        scale_min = np.round(0.80*mean_distance_new[i_peak]) - xmax_pattern - pixend
        if scale_min<0:
            scale_min = 0
    
        scale_max = np.round(1.2*mean_distance_new[i_peak]) - xmax_pattern - pixend
        scale_peak = np.arange(scale_min,scale_max+1)
    
        for i_scale in range(len(scale_peak)):
            template_resize_peak = np.concatenate([template_truncated,np.zeros(scale_peak[i_scale]),pattern])
            if len(I_detrend[:,0])>len(template_resize_peak):
                template_resize_peak1 = np.concatenate((template_resize_peak,np.zeros(len(I_detrend[:,0])-len(template_resize_peak))))
            corr_template = scipy.signal.correlate(I_detrend[:,0],template_resize_peak)
        
            if len(I_detrend[:,0])>len(template_resize_peak):
                val = np.dot(I_detrend[:,0],template_resize_peak1.T)
            else:
                I_detrend_2 = np.concatenate((I_detrend[:,0],np.zeros(len(template_resize_peak)-len(I_detrend[:,0]))))
                val = np.dot(I_detrend_2,template_resize_peak.T)
            corr_peak[i_scale,i_peak] = val
        
            if label.verbose==1:
                pl.xlim(0,len(I_detrend[:,0]))
                pl.plot(I_detrend[:,0])
                pl.plot(template_resize_peak)
                pl.show(block=False)
            
                pl.plot(corr_peak[:,i_peak],marker='+',linestyle='None',color='r')
                pl.title('correlation value against the displacement of the peak (px)')
                pl.show(block=False)
    
        max_peak = np.amax(corr_peak[:,i_peak])
        index_scale_peak = np.where(corr_peak[:,i_peak]==max_peak)
        good_scale_peak = scale_peak[index_scale_peak][0]
        Mcorr = Mcorr1
        Mcorr = np.resize(Mcorr,i_peak+2)
        Mcorr[i_peak+1] = np.amax(corr_peak[:,0:(i_peak+1)])
        flag = 0
    
        if i_peak>0:
            if (Mcorr[i_peak+1]-Mcorr[i_peak])<0.4*np.mean(Mcorr[1:i_peak+2]-Mcorr[0:i_peak+1]):
                test = i_peak
                template_resize_peak = np.concatenate((template_truncated,np.zeros(round(mean_distance[i_peak])-xmax_pattern-pixend),pattern))
                good_scale_peak = np.round(mean_distance[i_peak]) - xmax_pattern - pixend
                flag = 1
        if i_peak==0:
            if (Mcorr[i_peak+1] - Mcorr[i_peak])<0.4*Mcorr[0]:
                template_resize_peak = np.concatenate((template_truncated,np.zeros(round(mean_distance[i_peak])-xmax_pattern-pixend),pattern))
                good_scale_peak = round(mean_distance[i_peak]) - xmax_pattern - pixend
                flag = 1
        if flag==0:
            template_resize_peak=np.concatenate((template_truncated,np.zeros(good_scale_peak),pattern))
    
        mean_distance_new[i_peak] = good_scale_peak + xmax_pattern + pixend
        mean_ratio[i_peak] = np.mean(mean_distance_new[:,0:i_peak]/mean_distance[:,0:i_peak])
    
        template_truncated = template_resize_peak
    
        if label.verbose==1:
            pl.plot(I_detrend[:,0])
            pl.plot(template_truncated)
            pl.xlim(0,(len(I_detrend[:,0])-1))
            pl.show(block=False)

    minpeakvalue = 0.5
    loc_disk = np.arange(len(template_truncated))
    count = 0
    index_disk = 0
    for i in range(len(template_truncated)):
        if template_truncated[i]>=minpeakvalue:
            if i==0:
                if template_truncated[i]<template_truncated[i+1]:
                    index_disk = i
                    count  =  count + 1
            elif i==(len(template_truncated)-1):
                if template_truncated[i]<template_truncated[i-1]:
                    index_disk = np.resize(index_disk,count+1)
                    index_disk[len(index_disk)-1] = i
            else:
                if template_truncated[i]<template_truncated[i+1]:
                    index_disk = np.resize(index_disk,count+1)
                    index_disk[len(index_disk)-1] = i
                    count = count+1
                elif template_truncated[i]<template_truncated[i-1]:
                    index_disk = np.resize(index_disk,count+1)
                    index_disk[len(index_disk)-1] = i
                    count = count+1
        else:
            if i==0:
                index_disk = i
                count  =  count + 1
            else:
                index_disk = np.resize(index_disk,count+1)
                index_disk[len(index_disk)-1] = i
                count = count+1

    mask_disk = np.ones(len(template_truncated), dtype=bool)
    mask_disk[index_disk] = False
    loc_disk = loc_disk[mask_disk]
    X1 = np.where(loc_disk > I_detrend.shape[0])
    mask_disk1 = np.ones(len(loc_disk), dtype=bool)
    mask_disk1[X1] = False
    loc_disk = loc_disk[mask_disk1]
    loc_disk = loc_disk + start_centerline_y - 1

    #=====================================================================
    # Step 3: Building of labeled centerline and surface
    #=====================================================================
    print '\nBuilding of Labeled Centerline and Surface...'
    for i in range(len(loc_disk)):
    
        Index = np.array(np.where(y==loc_disk[i])).T
        lim_plus = Index + 5
        lim_minus = Index - 5
    
        if lim_minus<1:
            lim_minus=1
        if lim_plus>len(x):
            lim_plus=len(x)
    
        Vx = x[lim_plus] - x[lim_minus]
        Vz = z[lim_plus] - z[lim_minus]
        Vy = y[lim_plus] - y[lim_minus]
    
        d = Vx*x1[Index] + Vy*y[Index] + Vz*z[Index]
    
        intersection = np.ones(len(x))
        for j in range(len(x)):
            intersection[j] = np.abs((Vx*x[j]+Vy*y[j]+Vz*z[j]-d))
    
        min_intersection = np.amin(intersection)
        index_intersection = np.where(min_intersection==intersection)
        loc_disk[i] = y[index_intersection]

    center_disk = np.array(centerline)
    for i in range(len(loc_disk)-1):
        tmp = center_disk[loc_disk[i]:loc_disk[i+1]]
        tmp[np.where(tmp==1)] = i + level_start
        center_disk[loc_disk[i]:loc_disk[i+1]] = tmp

    center_disk[np.where(center_disk==1)] = 0

    if level_start==2:
        center_disk[x[0],(int(round(loc_disk[0] - C1C2_distance[1]))-1):loc_disk[0],z[0]] = 2
        center_disk[x[0],(int(round(loc_disk[0] - C1C2_distance[0] - C1C2_distance[1]))-1):int(round(loc_disk[0] - C1C2_distance[1]-1)),z[0]] = 1
        
    # Write NIFTI volumes
    hdr.set_data_dtype('uint8') # set imagetype to uint8
    print '\nWrite NIFTI volumes...'
    img = nibabel.Nifti1Image(center_disk, None, hdr)
    file_name = output_centerline_vertebra
    nibabel.save(img,file_name)
    print '.. File created:' + file_name

#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'USAGE: \n' \
        '  batch_labeling.py -i <filename> \n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        '  -i           input_file \n' \
        '  -c           contrast \n' \
        '  -l           Centerline binary Image  \n' \
        '  -m           mean_distance.mat \n' \
        '\n'\
        'OPTIONAL ARGUMENTS\n' \
        '  -o           output_surface \n' \
        '  -f           input surface \n' \
        '  -a           shift_AP in mm. Default value is 17mm \n' \
        '  -s           size_AP in mm. Default value is 6mm \n' \
        '  -r           size_RL in mm. Default value is 5mm \n' \
        '  -h           help. Show this message.\n' \
        '\n'\
        'EXAMPLE:\n' \
        '  '+os.path.basename(__file__)+'-i t1.nii -c T1 -l segmentation_centerline_binary.nii -m mean_distance.mat\n'
    sys.exit(2)

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # call main function
    main()    