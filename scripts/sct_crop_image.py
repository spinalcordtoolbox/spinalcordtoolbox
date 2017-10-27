#!/usr/bin/env python
#########################################################################################
#
# sct_crop_image and crop image wrapper.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin Leener, Julien Cohen-Adad, Olivier Comtois
# Modified: 2015-05-16
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import math
import time

from msct_parser import Parser
import os
import scipy
import nibabel
import sct_utils as sct
from msct_image import Image


class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        sct.printv('click', event)
        if event.inaxes != self.line.axes:
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


class ImageCropper(object):
    def __init__(self, input_file, output_file=None, mask=None, start=None, end=None, dim=None, shift=None, background=None, bmax=False, ref=None, mesh=None, rm_tmp_files=1, verbose=1, rm_output_file=0):
        self.input_filename = input_file
        self.output_filename = output_file
        self.mask = mask
        self.start = start
        self.end = end
        self.dim = dim
        self.shift = shift
        self.background = background
        self.bmax = bmax
        self.ref = ref
        self.mesh = mesh
        self.rm_tmp_files = rm_tmp_files
        self.verbose = verbose
        self.cmd = None
        self.result = None
        self.rm_output_file = rm_output_file

    def crop(self):
        """
        Crop image (change dimension)
        """

        # create command line
        self.cmd = "isct_crop_image" + " -i " + self.input_filename + " -o " + self.output_filename
        # Handling optional arguments

        # if mask is specified, find -start and -end arguments
        if self.mask is not None:
            # if user already specified -start or -end arguments, let him know they will be ignored
            if self.start is not None or self.end is not None:
                sct.printv('WARNING: Mask was specified for cropping. Arguments -start and -end will be ignored', 1, 'warning')
            self.start, self.end, self.dim = find_mask_boundaries(self.mask)

        if self.start is not None:
            self.cmd += " -start " + ','.join(map(str, self.start))
        if self.end is not None:
            self.cmd += " -end " + ','.join(map(str, self.end))
        if self.dim is not None:
            self.cmd += " -dim " + ','.join(map(str, self.dim))
        if self.shift is not None:
            self.cmd += " -shift " + ','.join(map(str, self.shift))
        if self.background is not None:
            self.cmd += " -b " + str(self.background)
        if self.bmax is True:
            self.cmd += " -bmax"
        if self.ref is not None:
            self.cmd += " -ref " + self.ref
        if self.mesh is not None:
            self.cmd += " -mesh " + self.mesh

        verb = 0
        if self.verbose == 1:
            verb = 2
        if self.mask is not None and self.background is not None:
            self.crop_from_mask_with_background()
        else:
            # Run command line
            sct.run(self.cmd, verb)

        self.result = Image(self.output_filename, verbose=self.verbose)

        # removes the output file created by the script if it is not needed
        if self.rm_output_file:
            try:
                os.remove(self.output_filename)
            except OSError:
                sct.printv("WARNING : Couldn't remove output file. Either it is opened elsewhere or "
                           "it doesn't exist.", self.verbose, 'warning')
        else:
            # Complete message
            sct.printv('\nDone! To view results, type:', self.verbose)
            sct.printv("fslview " + self.output_filename + " &\n", self.verbose, 'info')

        return self.result

    # mask the image in order to keep only voxels in the mask
    # doesn't change the image dimension
    def crop_from_mask_with_background(self):
        from numpy import asarray, einsum
        image_in = Image(self.input_filename)
        data_array = asarray(image_in.data)
        data_mask = asarray(Image(self.mask).data)
        assert data_array.shape == data_mask.shape

        # Element-wise matrix multiplication:
        new_data = None
        dim = len(data_array.shape)
        if dim == 3:
            new_data = einsum('ijk,ijk->ijk', data_mask, data_array)
        elif dim == 2:
            new_data = einsum('ij,ij->ij', data_mask, data_array)

        if self.background != 0:
            from sct_maths import get_data_or_scalar
            data_background = get_data_or_scalar(str(self.background), data_array)
            data_mask_inv = data_mask.max() - data_mask
            if dim == 3:
                data_background = einsum('ijk,ijk->ijk', data_mask_inv, data_background)
            elif dim == 2:
                data_background = einsum('ij,ij->ij', data_mask_inv, data_background)
            new_data += data_background

        # set image out
        image_in.setFileName(self.output_filename)
        image_in.data = new_data
        image_in.save()

    # shows the gui to crop the image
    def crop_with_gui(self):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        # Initialization
        fname_data = self.input_filename
        suffix_out = '_crop'
        remove_temp_files = self.rm_tmp_files
        verbose = self.verbose

        # Check file existence
        sct.printv('\nCheck file existence...', verbose)
        sct.check_file_exist(fname_data, verbose)

        # Get dimensions of data
        sct.printv('\nGet dimensions of data...', verbose)
        nx, ny, nz, nt, px, py, pz, pt = Image(fname_data).dim
        sct.printv('.. ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)
        # check if 4D data
        if not nt == 1:
            sct.printv('\nERROR in ' + os.path.basename(__file__) + ': Data should be 3D.\n', 1, 'error')
            sys.exit(2)

        # sct.printv(arguments)
        sct.printv('\nCheck parameters:')
        sct.printv('  data ................... ' + fname_data)

        # Extract path/file/extension
        path_data, file_data, ext_data = sct.extract_fname(fname_data)
        path_out, file_out, ext_out = '', file_data + suffix_out, ext_data

        # create temporary folder
        path_tmp = 'tmp.' + time.strftime("%y%m%d%H%M%S") + '/'
        sct.run('mkdir ' + path_tmp)

        # copy files into tmp folder
        from sct_convert import convert
        sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
        convert(fname_data, path_tmp + 'data.nii')

        # go to tmp folder
        os.chdir(path_tmp)

        # change orientation
        sct.printv('\nChange orientation to RPI...', verbose)
        sct.run('sct_image -i data.nii -setorient RPI -o data_rpi.nii')

        # get image of medial slab
        sct.printv('\nGet image of medial slab...', verbose)
        image_array = nibabel.load('data_rpi.nii').get_data()
        nx, ny, nz = image_array.shape
        scipy.misc.imsave('image.jpg', image_array[math.floor(nx / 2), :, :])

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
        nii = Image('data_rpi.nii')
        data_crop = nii.data[:, :, zcrop[0]:zcrop[1]]
        nii.data = data_crop
        nii.setFileName('data_rpi_crop.nii')
        nii.save()

        # come back to parent folder
        os.chdir('..')

        sct.printv('\nGenerate output files...', verbose)
        sct.generate_output_file(path_tmp + 'data_rpi_crop.nii', path_out + file_out + ext_out)

        # Remove temporary files
        if remove_temp_files == 1:
            sct.printv('\nRemove temporary files...')
            sct.run('rm -rf ' + path_tmp)

        # to view results
        sct.printv('\nDone! To view results, type:')
        sct.printv('fslview ' + path_out + file_out + ext_out + ' &')
        sct.printv()

def get_parser():
        # Initialize parser
    parser = Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description('Tools to crop an image. Either through command line or GUI')
    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="input image.",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name="-g",
                      type_value="multiple_choice",
                      description="1: use the GUI to crop, 0: use the command line to crop",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='0')

    # Command line mandatory arguments
    parser.usage.addSection("\nCOMMAND LINE RELATED MANDATORY ARGUMENTS")
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="output image. This option is REQUIRED for the command line execution",
                      mandatory=False,
                      example=['t1', 't2'])

    # Optional arguments section
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="1: display on, 0: display off (default)",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')

    parser.add_option(name="-h",
                      type_value=None,
                      description="Displays help",
                      mandatory=False)

    # GUI optional argument
    parser.usage.addSection("\nGUI RELATED OPTIONAL ARGUMENTS")
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="Remove temporary files. Default = 1",
                      mandatory=False,
                      example=['0', '1'])

    # Command line optional arguments
    parser.usage.addSection("\nCOMMAND LINE RELATED OPTIONAL ARGUMENTS")
    parser.add_option(name="-m",
                      type_value="file",
                      description="cropping around the mask",
                      mandatory=False)
    parser.add_option(name="-start",
                      type_value=[[','], 'float'],
                      description="start slices, ]0,1[: percentage, 0 & >1: slice number",
                      mandatory=False,
                      example="40,30,5")
    parser.add_option(name="-end",
                      type_value=[[','], 'float'],
                      description="end slices, ]0,1[: percentage, 0: last slice, >1: slice number, <0: last slice - value",
                      mandatory=False,
                      example="60,100,10")
    parser.add_option(name="-dim",
                      type_value=[[','], 'int'],
                      description="dimension to crop, from 0 to n-1, default is 1",
                      mandatory=False,
                      example="0,1,2")
    parser.add_option(name="-shift",
                      type_value=[[','], 'int'],
                      description="adding shift when used with mask, default is 0",
                      mandatory=False,
                      example="10,10,5")
    parser.add_option(name="-b",
                      type_value="float",
                      description="replace voxels outside cropping region with background value. \n"
                                  "If both the -m and the -b flags are used : the image is croped \"exactly\" around the mask with a background (and not around a rectangle area including the mask). the shape of the image isn't change.",
                      mandatory=False)
    parser.add_option(name="-bmax",
                      type_value=None,
                      description="maximize the cropping of the image (provide -dim if you want to specify the dimensions)",
                      mandatory=False)
    parser.add_option(name="-ref",
                      type_value="file",
                      description="crop input image based on reference image (works only for 3D images)",
                      mandatory=False,
                      example="ref.nii.gz")
    parser.add_option(name="-mesh",
                      type_value="file",
                      description="mesh to crop",
                      mandatory=False)
    parser.add_option(name="-rof",
                      type_value="multiple_choice",
                      description="remove output file created when cropping",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1'])
    parser.add_option(name="-bzmax",
                      type_value=None,
                      description="maximize the cropping of the image (provide -dim if you want to specify the dimensions)",
                      deprecated_by="-bmax",
                      mandatory=False)
    return parser


def find_mask_boundaries(fname_mask):
    """
    Find boundaries of a mask, i.e., min and max indices of non-null voxels in all dimensions.
    :param fname:
    :return: float: ind_start, ind_end
    """
    from numpy import nonzero, min, max
    # open mask
    data = Image(fname_mask).data
    data_nonzero = nonzero(data)
    # find min and max boundaries of the mask
    dim = len(data_nonzero)
    ind_start = [min(data_nonzero[i]) for i in range(dim)]
    ind_end = [max(data_nonzero[i]) for i in range(dim)]
    # create string indices
    # ind_start = ','.join(str(i) for i in xyzmin)
    # ind_end = ','.join(str(i) for i in xyzmax)
    # return values
    return ind_start, ind_end, range(dim)


if __name__ == "__main__":
    sct.start_stream_logger()
    parser = get_parser()
    # Fetching script arguments
    arguments = parser.parse(sys.argv[1:])

    # assigning variables to arguments
    input_filename = arguments["-i"]
    exec_choice = 0
    if "-g" in arguments:
        exec_choice = bool(int(arguments["-g"]))

    # cropping with GUI
    cropper = ImageCropper(input_filename)
    if exec_choice:
        fname_data = arguments["-i"]
        if "-r" in arguments:
            cropper.rm_tmp_files = int(arguments["-r"])
        if "-v" in arguments:
            cropper.verbose = int(arguments["-v"])

        cropper.crop_with_gui()

    # cropping with specified command-line arguments
    else:
        if "-o" in arguments:
            cropper.output_filename = arguments["-o"]
        else:
            sct.printv("An output file needs to be specified using the command line")
            sys.exit(2)
        # Handling optional arguments
        if "-m" in arguments:
            cropper.mask = arguments["-m"]
        if "-start" in arguments:
            cropper.start = arguments["-start"]
        if "-end" in arguments:
            cropper.end = arguments["-end"]
        if "-dim" in arguments:
            cropper.dim = arguments["-dim"]
        if "-shift" in arguments:
            cropper.shift = arguments["-shift"]
        if "-b" in arguments:
            cropper.background = arguments["-b"]
        if "-bmax" in arguments:
            cropper.bmax = True
        if "-ref" in arguments:
            cropper.ref = arguments["-ref"]
        if "-mesh" in arguments:
            cropper.mesh = arguments["-mesh"]

        cropper.crop()
