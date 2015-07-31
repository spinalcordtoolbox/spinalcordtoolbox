#!/usr/bin/env python
#########################################################################################
#
# crop image with GUI.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Karun Raju, Julien Cohen-Adad
# Modified: 2014-07-30
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: get slab instead of mid slice

import sys
import os
import commands
import getopt
import math
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nibabel
import time
from sct_orientation import set_orientation
import sct_utils as sct

# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')


class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1
        self.remove_temp_files = 1


class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print 'click', event
        if event.inaxes!=self.line.axes:
            # if user clicked outside the axis, ignore
            return
        if event.button == 2 or event.button == 3:
            # if right button, remove last point
            del self.xs[-1]
            del self.ys[-1]
        if len(self.xs) >= 2:
            # if user already clicked 2 times, ignore
            return
        if event.button == 1:
            # if left button, add point
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
        # update figure
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()


#=======================================================================================================================
# main
#=======================================================================================================================
def main():
    # Initialization
    fname_data = ''
    suffix_out = '_crop'
    remove_temp_files = param.remove_temp_files
    verbose = param.verbose
    fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI
    remove_temp_files = param.remove_temp_files
    
    # Parameters for debug mode
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
        fname_data = path_sct+'/testing/data/errsm_23/t2/t2.nii.gz'
        remove_temp_files = 0
    else:
        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:],'hi:r:v:')
        except getopt.GetoptError:
            usage()
        if not opts:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt in ('-i'):
                fname_data = arg
            elif opt in ('-r'):
                remove_temp_files = int(arg)
            elif opt in ('-v'):
                verbose = int(arg)

    # display usage if a mandatory argument is not provided
    if fname_data == '':
        usage()

    # Check file existence
    sct.printv('\nCheck file existence...', verbose)
    sct.check_file_exist(fname_data, verbose)

    # Get dimensions of data
    sct.printv('\nGet dimensions of data...', verbose)
    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_data)
    sct.printv('.. '+str(nx)+' x '+str(ny)+' x '+str(nz), verbose)
    # check if 4D data
    if not nt == 1:
        sct.printv('\nERROR in '+os.path.basename(__file__)+': Data should be 3D.\n', 1, 'error')
        sys.exit(2)

    # print arguments
    print '\nCheck parameters:'
    print '  data ................... '+fname_data
    print

    # Extract path/file/extension
    path_data, file_data, ext_data = sct.extract_fname(fname_data)
    path_out, file_out, ext_out = '', file_data+suffix_out, ext_data

    # create temporary folder
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")+'/'
    sct.run('mkdir '+path_tmp)

    # copy files into tmp folder
    sct.run('isct_c3d '+fname_data+' -o '+path_tmp+'data.nii')

    # go to tmp folder
    os.chdir(path_tmp)

    # change orientation
    sct.printv('\nChange orientation to RPI...', verbose)
    set_orientation('data.nii', 'RPI', 'data_rpi.nii')

    # get image of medial slab
    sct.printv('\nGet image of medial slab...', verbose)
    image_array = nibabel.load('data_rpi.nii').get_data()
    nx, ny, nz = image_array.shape
    scipy.misc.imsave('image.jpg', image_array[math.floor(nx/2), :, :])

    # Display the image
    sct.printv('\nDisplay image and get cropping region...', verbose)
    fig = plt.figure()
    # fig = plt.gcf()
    # ax = plt.gca()
    ax = fig.add_subplot(111)
    img = mpimg.imread("image.jpg")
    implot = ax.imshow(img.T)
    implot.set_cmap('gray')
    plt.gca().invert_yaxis()
    # mouse callback
    ax.set_title('Left click on the top and bottom of your cropping field.\n Right click to remove last point.\n Close window when your done.')
    line, = ax.plot([], [], 'ro')  # empty line
    cropping_coordinates = LineBuilder(line)
    plt.show()
    # disconnect callback
    # fig.canvas.mpl_disconnect(line)

    # check if user clicked two times
    if len(cropping_coordinates.xs) != 2:
        sct.printv('\nERROR: You have to select two points. Exit program.\n', 1, 'error')
        sys.exit(2)

    # convert coordinates to integer
    zcrop = [int(i) for i in cropping_coordinates.ys]

    # sort coordinates
    zcrop.sort()

    # crop image
    sct.printv('\nCrop image...', verbose)
    sct.run(fsloutput+'fslroi data_rpi.nii data_rpi_crop.nii 0 -1 0 -1 '+str(zcrop[0])+' '+str(zcrop[1]-zcrop[0]+1))

    # come back to parent folder
    os.chdir('..')

    sct.printv('\nGenerate output files...', verbose)
    sct.generate_output_file(path_tmp+'data_rpi_crop.nii', path_out+file_out+ext_out)

    # Remove temporary files
    if remove_temp_files == 1:
        print('\nRemove temporary files...')
        sct.run('rm -rf '+path_tmp)

    # to view results
    print '\nDone! To view results, type:'
    print 'fslview '+path_out+file_out+ext_out+' &'
    print



#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Quickly crop an image in the superior-inferior direction by clicking at the top and bottom points
  of the desired field-of-view.

USAGE
  """+os.path.basename(__file__)+""" -i <volume>

MANDATORY ARGUMENTS
  -i <volume>           volume to crop

OPTIONAL ARGUMENTS
  -h                    help. Show this message
  -v {0,1}              verbose. Default = """+str(param_default.verbose)+"""
  -r {0,1}              remove temporary files. Default="""+str(param_default.remove_temp_files)+"""
EXAMPLE
  """+os.path.basename(__file__)+""" -i t1.nii.gz\n"""

    #Exit Program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    param = Param()
    param_default = Param()
    # call main function
    main()