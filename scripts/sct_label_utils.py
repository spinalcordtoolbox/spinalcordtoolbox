#!/usr/bin/env python
#########################################################################################
#
# Create, remove or display labels.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad
# Modified: 2014-10-29
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: currently it seems like cross_radius is given in pixel instead of mm

import os, sys
import getopt
import commands
import sys
import sct_utils as sct
import nibabel
import numpy as np


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.fname_label_output = 'labels.nii.gz'
        self.labels = []
        self.verbose = 1


#=======================================================================================================================
# main
#=======================================================================================================================
def main():

    # Initialization
    fname_label = ''
    fname_label_output = param.fname_label_output
    cross_radius = 5
    dilate = False
    fname_ref = ''
    type_process = ''
    output_level = 0  # 0: image ; 1: txt file

    # get path of the toolbox
    status, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        status, path_sct_data = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')
        fname_label = path_sct_data+'/mt/mt1.nii.gz'
        param.labels = '5,5,2,1:5,7,2,3'
        type_process = 'create'
        cross_radius = 5
        dilate = True
    else:
        # Check input param
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hi:o:c:r:t:l:dx:')
        except getopt.GetoptError as err:
            print str(err)
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ('-i'):
                fname_label = arg
            elif opt in ('-o'):
                fname_label_output = arg
            elif opt in ('-c'):
                cross_radius = int(arg)
            elif opt in ('-d'):
                dilate = True
            elif opt in ('-r'):
                fname_ref = arg
            elif opt in ('-t'):
                type_process = arg
            elif opt in ('-l'):
                output_level = int(arg)
            elif opt in '-x':
                param.labels = arg

    # display usage if a mandatory argument is not provided
    if fname_label == '' or type_process == '':
        sct.printv('\nERROR: All mandatory arguments are not provided. See usage (add -h).\n', 1, 'error')

    # check existence of input files
    sct.check_file_exist(fname_label)
    if fname_ref != '':
        sct.check_file_exist(fname_ref)
    
    # extract path/file/extension
    path_label, file_label, ext_label = sct.extract_fname(fname_label)
    path_label_output, file_label_output, ext_label_output = sct.extract_fname(fname_label_output)

    # read nifti input file
    img = nibabel.load(fname_label)
    # 3d array for each x y z voxel values for the input nifti image
    data = img.get_data()
    hdr = img.get_header()

    # Get dimensions
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_label)

    # switch to process
    if type_process == 'cross':
        data = cross(data, cross_radius, fname_ref, dilate, px, py)
    if type_process == 'plan':
        data = plan(data, cross_radius)
    elif type_process == 'remove':
        data = remove_label(data, fname_ref)
    elif type_process == 'disk':
        extract_disk_position(data, fname_ref, output_level, fname_label_output)
    elif type_process == 'centerline':
        extract_centerline(data, fname_label_output)
        output_level = 1
    elif type_process == 'segmentation':
        extract_segmentation(data, fname_label_output)
        output_level = 1
    elif type_process == 'fraction-volume':
        fraction_volume(data,fname_ref,fname_label_output)
        output_level = 1
    elif type_process == 'write-vert-levels':
        write_vertebral_levels(data,fname_ref)
    elif type_process == 'display-voxel':
        display_voxel(data)
        output_level = 1
    elif type_process == 'create':
        data = create_label(data)
        output_level = 0

    # write nifti file
    if (output_level == 0):
        hdr.set_data_dtype('int32') # set imagetype to uint8, previous: int32.
        print '\nWrite NIFTI volumes...'
        data.astype('int')
        img = nibabel.Nifti1Image(data, None, hdr)
        nibabel.save(img, 'tmp.'+file_label_output+'.nii.gz')
        sct.generate_output_file('tmp.'+file_label_output+'.nii.gz', file_label_output+ext_label_output)


# cross
# ==========================================================================================
def cross(data, cross_radius, fname_ref, dilate, px, py):
    X, Y, Z = (data > 0).nonzero()
    a = len(X)
    d = cross_radius # cross radius in pixel
    dx = d/px # cross radius in mm
    dy = d/py

    # for all points with non-zeros neighbors, force the neighbors to 0
    for i in range(0,a):
        value = int(data[X[i]][Y[i]][Z[i]])
        data[X[i]][Y[i]][Z[i]] = 0 # remove point on the center of the spinal cord
        if fname_ref == '':
            data[X[i]][Y[i]+dy][Z[i]] = value*10+1 # add point at distance from center of spinal cord
            data[X[i]+dx][Y[i]][Z[i]] = value*10+2
            data[X[i]][Y[i]-dy][Z[i]] = value*10+3
            data[X[i]-dx][Y[i]][Z[i]] = value*10+4

            # dilate cross to 3x3
            if dilate:
                data[X[i]-1][Y[i]+dy-1][Z[i]] = data[X[i]][Y[i]+dy-1][Z[i]] = data[X[i]+1][Y[i]+dy-1][Z[i]] = data[X[i]+1][Y[i]+dy][Z[i]] = data[X[i]+1][Y[i]+dy+1][Z[i]] = data[X[i]][Y[i]+dy+1][Z[i]] = data[X[i]-1][Y[i]+dy+1][Z[i]] = data[X[i]-1][Y[i]+dy][Z[i]] = data[X[i]][Y[i]+dy][Z[i]]
                data[X[i]+dx-1][Y[i]-1][Z[i]] = data[X[i]+dx][Y[i]-1][Z[i]] = data[X[i]+dx+1][Y[i]-1][Z[i]] = data[X[i]+dx+1][Y[i]][Z[i]] = data[X[i]+dx+1][Y[i]+1][Z[i]] = data[X[i]+dx][Y[i]+1][Z[i]] = data[X[i]+dx-1][Y[i]+1][Z[i]] = data[X[i]+dx-1][Y[i]][Z[i]] = data[X[i]+dx][Y[i]][Z[i]]
                data[X[i]-1][Y[i]-dy-1][Z[i]] = data[X[i]][Y[i]-dy-1][Z[i]] = data[X[i]+1][Y[i]-dy-1][Z[i]] = data[X[i]+1][Y[i]-dy][Z[i]] = data[X[i]+1][Y[i]-dy+1][Z[i]] = data[X[i]][Y[i]-dy+1][Z[i]] = data[X[i]-1][Y[i]-dy+1][Z[i]] = data[X[i]-1][Y[i]-dy][Z[i]] = data[X[i]][Y[i]-dy][Z[i]]
                data[X[i]-dx-1][Y[i]-1][Z[i]] = data[X[i]-dx][Y[i]-1][Z[i]] = data[X[i]-dx+1][Y[i]-1][Z[i]] = data[X[i]-dx+1][Y[i]][Z[i]] = data[X[i]-dx+1][Y[i]+1][Z[i]] = data[X[i]-dx][Y[i]+1][Z[i]] = data[X[i]-dx-1][Y[i]+1][Z[i]] = data[X[i]-dx-1][Y[i]][Z[i]] = data[X[i]-dx][Y[i]][Z[i]]
        else:
            # read nifti input file
            img_ref = nibabel.load(fname_ref)
            # 3d array for each x y z voxel values for the input nifti image
            data_ref = img_ref.get_data()
            profile = []; p_median = []; p_gradient = []
            for j in range(0,d+1):
                profile.append(data_ref[X[i]][Y[i]+j][Z[i]])
            for k in range(1,d):
                a = np.array([profile[k-1],profile[k],profile[k+1]])
                p_median.append(np.median(a))
            for l in range(0,d-2):
                p_gradient.append(p_median[l+1]-p_median[l])
            d1 = p_gradient.index(max(p_gradient))

            profile = []; p_median = []; p_gradient = []
            for j in range(0,d+1):
                profile.append(data_ref[X[i]+j][Y[i]][Z[i]])
            for k in range(1,d):
                a = np.array([profile[k-1],profile[k],profile[k+1]])
                p_median.append(np.median(a))
            for l in range(0,d-2):
                p_gradient.append(p_median[l+1]-p_median[l])
            d2 = p_gradient.index(max(p_gradient))

            profile = []; p_median = []; p_gradient = []
            for j in range(0,d+1):
                profile.append(data_ref[X[i]][Y[i]-j][Z[i]])
            for k in range(1,d):
                a = np.array([profile[k-1],profile[k],profile[k+1]])
                p_median.append(np.median(a))
            for l in range(0,d-2):
                p_gradient.append(p_median[l+1]-p_median[l])
            d3 = p_gradient.index(max(p_gradient))

            profile = []; p_median = []; p_gradient = []
            for j in range(0,d+1):
                profile.append(data_ref[X[i]-j][Y[i]][Z[i]])
            for k in range(1,d):
                a = np.array([profile[k-1],profile[k],profile[k+1]])
                p_median.append(np.median(a))
            for l in range(0,d-2):
                p_gradient.append(p_median[l+1]-p_median[l])
            d4 = p_gradient.index(max(p_gradient))

            data[X[i]][Y[i]+d1][Z[i]] = value*10+1 # add point at distance from center of spinal cord
            data[X[i]+d2][Y[i]][Z[i]] = value*10+2
            data[X[i]][Y[i]-d3][Z[i]] = value*10+3
            data[X[i]-d4][Y[i]][Z[i]] = value*10+4

            # dilate cross to 3x3
            if dilate:
                data[X[i]-1][Y[i]+d1-1][Z[i]] = data[X[i]][Y[i]+d1-1][Z[i]] = data[X[i]+1][Y[i]+d1-1][Z[i]] = data[X[i]+1][Y[i]+d1][Z[i]] = data[X[i]+1][Y[i]+d1+1][Z[i]] = data[X[i]][Y[i]+d1+1][Z[i]] = data[X[i]-1][Y[i]+d1+1][Z[i]] = data[X[i]-1][Y[i]+d1][Z[i]] = data[X[i]][Y[i]+d1][Z[i]]
                data[X[i]+d2-1][Y[i]-1][Z[i]] = data[X[i]+d2][Y[i]-1][Z[i]] = data[X[i]+d2+1][Y[i]-1][Z[i]] = data[X[i]+d2+1][Y[i]][Z[i]] = data[X[i]+d2+1][Y[i]+1][Z[i]] = data[X[i]+d2][Y[i]+1][Z[i]] = data[X[i]+d2-1][Y[i]+1][Z[i]] = data[X[i]+d2-1][Y[i]][Z[i]] = data[X[i]+d2][Y[i]][Z[i]]
                data[X[i]-1][Y[i]-d3-1][Z[i]] = data[X[i]][Y[i]-d3-1][Z[i]] = data[X[i]+1][Y[i]-d3-1][Z[i]] = data[X[i]+1][Y[i]-d3][Z[i]] = data[X[i]+1][Y[i]-d3+1][Z[i]] = data[X[i]][Y[i]-d3+1][Z[i]] = data[X[i]-1][Y[i]-d3+1][Z[i]] = data[X[i]-1][Y[i]-d3][Z[i]] = data[X[i]][Y[i]-d3][Z[i]]
                data[X[i]-d4-1][Y[i]-1][Z[i]] = data[X[i]-d4][Y[i]-1][Z[i]] = data[X[i]-d4+1][Y[i]-1][Z[i]] = data[X[i]-d4+1][Y[i]][Z[i]] = data[X[i]-d4+1][Y[i]+1][Z[i]] = data[X[i]-d4][Y[i]+1][Z[i]] = data[X[i]-d4-1][Y[i]+1][Z[i]] = data[X[i]-d4-1][Y[i]][Z[i]] = data[X[i]-d4][Y[i]][Z[i]]

    return data


# plan
# ==========================================================================================
def plan(data, width):
    X, Y, Z = (data > 0).nonzero()
    # for all points with non-zeros neighbors, force the neighbors to 0
    for i in range(0,len(X)):
        value = int(data[X[i]][Y[i]][Z[i]])
        data[:,:,Z[i]-width:Z[i]+width] = 100+5*value
    return data

# create_label
#=======================================================================================================================
def create_label(data):

    # create labels volume (all zeros)
    data_label = data*0

    # parse argument for labels
    list_labels = param.labels.split(':')  # parse with space

    # loop across labels
    for i in range(0, len(list_labels)):
        # get labels coordinates and value
        x, y, z, v = list_labels[i].split(',')
        # display info
        sct.printv('Label #'+str(i)+': '+str(x)+','+str(y)+','+str(z)+' --> '+str(v), 1)
        # assing value
        data_label[x, y, z] = int(v)

    return data_label


#=======================================================================================================================
def remove_label(data, fname_ref):
    X, Y, Z = (data > 0).nonzero()

    img_ref = nibabel.load(fname_ref)
    # 3d array for each x y z voxel values for the input nifti image
    data_ref = img_ref.get_data()
    X_ref, Y_ref, Z_ref = (data_ref > 0).nonzero()

    nbLabel = len(X)
    nbLabel_ref = len(X_ref)
    for i in range(0,nbLabel):
        value = data[X[i]][Y[i]][Z[i]]
        isInRef = False
        for j in range(0,nbLabel_ref):
            value_ref = data_ref[X_ref[j]][Y_ref[j]][Z_ref[j]]
            # the following line could make issues when down sampling input, for example 21,00001 not = 21,0
            #if value_ref == value:
            if abs(value - value_ref) < 0.1:
                data[X[i]][Y[i]][Z[i]] = value_ref
                isInRef = True
        if isInRef == False:
            data[X[i]][Y[i]][Z[i]] = 0

    return data


# need binary centerline and segmentation with vertebral level. output_level=1 -> write .txt file. output_level=1 -> write centerline with vertebral levels
#=======================================================================================================================
def extract_disk_position(data_level, fname_centerline, output_level, fname_label_output):
    X, Y, Z = (data_level > 0).nonzero()
    
    img_centerline = nibabel.load(fname_centerline)
    # 3d array for each x y z voxel values for the input nifti image
    data_centerline = img_centerline.get_data()
    Xc, Yc, Zc = (data_centerline > 0).nonzero()
    nbLabel = len(X)
    nbLabel_centerline = len(Xc)
    # sort Xc, Yc, and Zc depending on Yc
    cent = [Xc, Yc, Zc]
    indices = range(nbLabel_centerline)
    indices.sort(key = cent[1].__getitem__)
    for i, sublist in enumerate(cent):
        cent[i] = [sublist[j] for j in indices]
    Xc = []
    Yc = []
    Zc = []
    # remove double values
    for i in range(0,len(cent[1])):
        if Yc.count(cent[1][i])==0:
            Xc.append(cent[0][i])
            Yc.append(cent[1][i])
            Zc.append(cent[2][i])
    nbLabel_centerline = len(Xc)
    
    centerline_level = [0 for a in range(nbLabel_centerline)]
    for i in range(0,nbLabel_centerline):
        centerline_level[i] = data_level[Xc[i]][Yc[i]][Zc[i]]
        data_centerline[Xc[i]][Yc[i]][Zc[i]] = 0
    for i in range(0,nbLabel_centerline-1):
        centerline_level[i] = abs(centerline_level[i+1]-centerline_level[i])
    centerline_level[-1] = 0

    C = [i for i, e in enumerate(centerline_level) if e != 0]
    nb_disks = len(C)
    
    if output_level==0:
        for i in range(0,nb_disks):
            data_centerline[Xc[C[i]]][Yc[C[i]]][Zc[C[i]]] = data_level[Xc[C[i]]][Yc[C[i]]][Zc[C[i]]]
    elif output_level==1:
        fo = open(fname_label_output, "wb")
        for i in range(0,nb_disks):
            line = (data_level[Xc[C[i]]][Yc[C[i]]][Zc[C[i]]],Xc[C[i]],Yc[C[i]],Zc[C[i]])
            fo.write("%i %i %i %i\n" %line)
        fo.close()

    return data_centerline


#=======================================================================================================================
def extract_centerline(data,fname_label_output):
    # the Z image is assume to be in second dimension
    X, Y, Z = (data > 0).nonzero()
    cent = [X, Y, Z]
    indices = range(0,len(X))
    indices.sort(key = cent[1].__getitem__)
    for i, sublist in enumerate(cent):
        cent[i] = [sublist[j] for j in indices]
    X = []; Y = []; Z = []
    # remove double values
    for i in range(0,len(cent[1])):
        if Y.count(cent[1][i])==0:
            X.append(cent[0][i])
            Y.append(cent[1][i])
            Z.append(cent[2][i])
    
    fo = open(fname_label_output, "wb")
    for i in range(0,len(X)):
        line = (X[i],Y[i],Z[i])
        fo.write("%i %i %i\n" %line)
    fo.close()


#=======================================================================================================================
def extract_segmentation(data,fname_label_output):
    # the Z image is assume to be in second dimension
    X, Y, Z = (data > 0).nonzero()
    cent = [X, Y, Z]
    indices = range(0,len(X))
    indices.sort(key = cent[1].__getitem__)
    for i, sublist in enumerate(cent):
        cent[i] = [sublist[j] for j in indices]
    X = []; Y = []; Z = []
    # remove double values
    for i in range(0,len(cent[1])):
        X.append(cent[0][i])
        Y.append(cent[1][i])
        Z.append(cent[2][i])
    
    fo = open(fname_label_output, "wb")
    for i in range(0,len(X)):
        line = (X[i],Y[i],Z[i])
        fo.write("%i %i %i\n" %line)
    fo.close()


#=======================================================================================================================
def fraction_volume(data,fname_ref,fname_label_output):
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_ref)
    img_ref = nibabel.load(fname_ref)
    # 3d array for each x y z voxel values for the input nifti image
    data_ref = img_ref.get_data()
    Xr, Yr, Zr = (data_ref > 0).nonzero()
    ref_matrix = [Xr, Yr, Zr]
    indices = range(0,len(Xr))
    indices.sort(key = ref_matrix[1].__getitem__)
    for i, sublist in enumerate(ref_matrix):
        ref_matrix[i] = [sublist[j] for j in indices]
    Xr = []; Yr = []; Zr = []
    for i in range(0,len(ref_matrix[1])):
        Xr.append(ref_matrix[0][i])
        Yr.append(ref_matrix[1][i])
        Zr.append(ref_matrix[2][i])
    
    X, Y, Z = (data > 0).nonzero()
    data_matrix = [X, Y, Z]
    indices = range(0,len(X))
    indices.sort(key = data_matrix[1].__getitem__)
    for i, sublist in enumerate(data_matrix):
        data_matrix[i] = [sublist[j] for j in indices]
    X = []; Y = []; Z = []
    for i in range(0,len(data_matrix[1])):
        X.append(data_matrix[0][i])
        Y.append(data_matrix[1][i])
        Z.append(data_matrix[2][i])

    volume_fraction = []
    for i in range(ny):
        r = []
        for j,p in enumerate(Yr):
            if p == i:
                r.append(j)
        d = []
        for j,p in enumerate(Y):
            if p == i:
                d.append(j)
        volume_ref = 0.0
        for k in range(len(r)):
            value = data_ref[Xr[r[k]]][Yr[r[k]]][Zr[r[k]]]
            if value > 0.5:
                volume_ref = volume_ref + value #suppose 1mm isotropic resolution
        volume_data = 0.0
        for k in range(len(d)):
            value = data[X[d[k]]][Y[d[k]]][Z[d[k]]]
            if value > 0.5:
                volume_data = volume_data + value #suppose 1mm isotropic resolution
        if volume_ref!=0:
            volume_fraction.append(volume_data/volume_ref)
        else:
            volume_fraction.append(0)

    fo = open(fname_label_output, "wb")
    for i in range(ny):
        fo.write("%i %f\n" %(i,volume_fraction[i]))
    fo.close()


#=======================================================================================================================
def write_vertebral_levels(data,fname_vert_level_input):
    fo = open(fname_vert_level_input)
    vertebral_levels = fo.readlines()
    vert = [int(n[2]) for n in [line.strip().split() for line in vertebral_levels]]
    vert.reverse()
    fo.close()
    
    X, Y, Z = (data > 0).nonzero()
    length_points = len(X)
    
    for i in range(0,length_points):
        if Y[i] > vert[0]:
            data[X[i]][Y[i]][Z[i]] = 0
        elif Y[i] < vert[-1]:
            data[X[i]][Y[i]][Z[i]] = 0
        else:
            for k in range(0,len(vert)-1):
                if vert[k+1] < Y[i] <= vert[k]:
                    data[X[i]][Y[i]][Z[i]] = k+1


#=======================================================================================================================
def display_voxel(data):
    # the Z image is assume to be in second dimension
    X, Y, Z = (data > 0).nonzero()
    for k in range(0,len(X)):
        print 'Position=('+str(X[k])+','+str(Y[k])+','+str(Z[k])+') -- Value= '+str(data[X[k],Y[k],Z[k]])


#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Utility function for labels.

USAGE
  """+os.path.basename(__file__)+""" -i <data> -t process <process>

MANDATORY ARGUMENTS
  -i <data>        labels or image to create labels on. Must be 3D.
  -t <process>     process:
                     cross: create a cross. Must use flag "-c"
                     remove: remove labels. Must use flag "-r".
                     display-voxel: display all labels in file
                     create: create labels. Must use flag "-l" to list labels.

OPTIONAL ARGUMENTS
  -x <x,y,z,v>     labels. Use ":" if you have multiple labels.
                     x: x-coordinates
                     y: y-coordinates
                     z: z-coordinates
                     v: value of label
  -o <output>      output volume
  -r <volume>      reference volume for label removing.
  -c <radius>      cross radius in mm (default=5mm).
  -v {0,1}         verbose. Default="""+str(param_default.verbose)+"""
  -d               dilate.
  -h               help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i t2.nii.gz -c 5\n"""

    # exit program
    sys.exit(2)

    
#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_default = Param()
    # call main function
    main()