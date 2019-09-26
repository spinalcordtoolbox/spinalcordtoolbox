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

from __future__ import absolute_import, division

import sys, io, os, math, time, argparse

import imageio
import numpy as np
import scipy
import nibabel
import matplotlib
matplotlib.use('tkagg')
import sct_utils as sct
import spinalcordtoolbox.image as msct_image
from spinalcordtoolbox.image import Image, zeros_like
from spinalcordtoolbox.utils import Metavar, SmartFormatter


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

        img_in = Image(self.input_filename)

        self.cmd = ["isct_crop_image", "-i", self.input_filename, "-o", self.output_filename]
        # Handling optional arguments

        # if mask is specified, find -start and -end arguments
        if self.mask is not None:
            # if user already specified -start or -end arguments, let him know they will be ignored
            if self.start is not None or self.end is not None:
                sct.printv('WARNING: Mask was specified for cropping. Arguments -start and -end will be ignored', 1, 'warning')
            self.start, self.end, self.dim = find_mask_boundaries(self.mask)

        if self.start is not None:
            self.cmd += ["-start", ','.join(map(str, self.start))]
        if self.end is not None:
            self.cmd += ["-end", ','.join(map(str, self.end))]
        if self.dim is not None:
            self.cmd += ["-dim", ','.join(map(str, self.dim))]
        if self.shift is not None:
            self.cmd += ["-shift", ','.join(map(str, self.shift))]
        if self.background is not None:
            self.cmd += ["-b", str(self.background)]
        if self.bmax is True:
            self.cmd += ["-bmax"]
        if self.ref is not None:
            self.cmd += ["-ref", self.ref]
        if self.mesh is not None:
            self.cmd += ["-mesh", self.mesh]

        verb = 0
        if self.verbose == 1:
            verb = 2
        if self.mask is not None and self.background is not None:
            self.crop_from_mask_with_background()
        else:
            # Run command line
            sct.run(self.cmd, verb, is_sct_binary=True)

        self.result = Image(self.output_filename, verbose=self.verbose)

        # removes the output file created by the script if it is not needed
        if self.rm_output_file:
            try:
                os.remove(self.output_filename)
            except OSError:
                sct.printv("WARNING : Couldn't remove output file. Either it is opened elsewhere or "
                           "it doesn't exist.", self.verbose, 'warning')
        else:
            if self.verbose >= 1:
                sct.display_viewer_syntax([self.output_filename])

        return self.result

    # mask the image in order to keep only voxels in the mask
    # doesn't change the image dimension
    def crop_from_mask_with_background(self):

        image_in = Image(self.input_filename)
        data_array = np.asarray(image_in.data)
        data_mask = np.asarray(Image(self.mask).data)
        assert data_array.shape == data_mask.shape

        # Element-wise matrix multiplication:
        new_data = None
        dim = len(data_array.shape)
        if dim == 3:
            new_data = data_mask * data_array
        elif dim == 2:
            new_data = data_mask * data_array

        if self.background != 0:
            from sct_maths import get_data_or_scalar
            data_background = get_data_or_scalar(str(self.background), data_array)
            data_mask_inv = data_mask.max() - data_mask
            if dim == 3:
                data_background = data_mask_inv * data_background
            elif dim == 2:
                data_background = data_mask_inv * data_background
            new_data += data_background

        image_out = msct_image.empty_like(image_in)
        image_out.data = new_data
        image_out.save(self.output_filename)

    # shows the gui to crop the image
    def crop_with_gui(self):
        """
        Launch a GUI. The medial sagittal plane of the image is shown. User selects two points: top-left and bottom-
        right of the cropping window.
        :return:
        """

        from spinalcordtoolbox.gui import base
        from spinalcordtoolbox.gui.sagittal import launch_sagittal_dialog

        # Change orientation to SAL
        img_in = Image(self.input_filename)
        native_orientation = img_in.orientation
        img_in.change_orientation('SAL')

        # Launch GUI
        params = base.AnatomicalParams()
        params.vertebraes = [1, 2]  # TODO: use these labels instead ['top-left (S-A)', 'bottom-right (I-P)']
        params.input_file_name = self.input_filename
        params.output_file_name = self.output_filename
        params.subtitle = "Click on the top-left and bottom-right of the image to select your cropping window."
        img_labels = zeros_like(img_in)
        launch_sagittal_dialog(img_in, img_labels, params)

        # Extract coordinates
        cropping_coord = img_labels.getNonZeroCoordinates(sorting='value')

        # import matplotlib.pyplot as plt
        # import matplotlib.image as mpimg
        # # Initialization
        # fname_data = self.input_filename
        # suffix_out = '_crop'  # TODO: change that
        # remove_temp_files = self.rm_tmp_files
        # verbose = self.verbose
        #
        # # Check file existence
        # sct.printv('\nCheck file existence...', verbose)
        # sct.check_file_exist(fname_data, verbose)
        #
        # # Get dimensions of data
        # sct.printv('\nGet dimensions of data...', verbose)
        # nx, ny, nz, nt, px, py, pz, pt = Image(fname_data).dim
        # sct.printv('.. ' + str(nx) + ' x ' + str(ny) + ' x ' + str(nz), verbose)
        # # check if 4D data
        # if not nt == 1:
        #     sct.printv('\nERROR in ' + os.path.basename(__file__) + ': Data should be 3D.\n', 1, 'error')
        #     sys.exit(2)
        #
        # # sct.printv(arguments)
        # sct.printv('\nCheck parameters:')
        # sct.printv('  data ................... ' + fname_data)
        #
        # # Extract path/file/extension
        # path_data, file_data, ext_data = sct.extract_fname(fname_data)
        # path_out, file_out, ext_out = '', file_data + suffix_out, ext_data
        #
        # path_tmp = sct.tmp_create() + "/"
        #
        # # copy files into tmp folder
        # from sct_convert import convert
        # sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
        # convert(fname_data, os.path.join(path_tmp, "data.nii"))
        #
        # # go to tmp folder
        # curdir = os.getcwd()
        # os.chdir(path_tmp)
        #
        # # change orientation
        # sct.printv('\nChange orientation to RPI...', verbose)
        # Image('data.nii').change_orientation("RPI").save('data_rpi.nii')
        #
        # # get image of medial slab
        # sct.printv('\nGet image of medial slab...', verbose)
        # image_array = nibabel.load('data_rpi.nii').get_data()
        # nx, ny, nz = image_array.shape
        # imageio.imwrite('image.jpg', image_array[math.floor(nx / 2), :, :])
        #
        # # Display the image
        # sct.printv('\nDisplay image and get cropping region...', verbose)
        # fig = plt.figure()
        # # fig = plt.gcf()
        # # ax = plt.gca()
        # ax = fig.add_subplot(111)
        # img = mpimg.imread("image.jpg")
        # implot = ax.imshow(img.T)
        # implot.set_cmap('gray')
        # plt.gca().invert_yaxis()
        # # mouse callback
        # ax.set_title('Left click on the top and bottom of your cropping field.\n Right click to remove last point.\n Close window when your done.')
        # line, = ax.plot([], [], 'ro')  # empty line
        # cropping_coordinates = LineBuilder(line)
        # plt.show()
        # # disconnect callback
        # # fig.canvas.mpl_disconnect(line)
        #
        # # check if user clicked two times
        # if len(cropping_coordinates.xs) != 2:
        #     sct.printv('\nERROR: You have to select two points. Exit program.\n', 1, 'error')
        #     sys.exit(2)
        #
        # # convert coordinates to integer
        # zcrop = [int(i) for i in cropping_coordinates.ys]
        #
        # # sort coordinates
        # zcrop.sort()

        # Crop image
        data_crop = img_in.data[cropping_coord[0].x:cropping_coord[1].x, cropping_coord[0].y:cropping_coord[1].y, :]
        img_out = Image(param=data_crop, hdr=img_in.hdr)
        img_out.change_orientation(native_orientation)
        img_out.absolutepath = 'data_crop.nii'
        img_out.save()
        # nii.save()

        # come back
        # os.chdir(curdir)

        # sct.printv('\nGenerate output files...', verbose)
        # sct.generate_output_file(os.path.join(path_tmp, "data_rpi_crop.nii"), os.path.join(path_out, file_out + ext_out))

        # # Remove temporary files
        # if remove_temp_files == 1:
        #     sct.printv('\nRemove temporary files...')
        #     sct.rmtree(path_tmp)

        # sct.display_viewer_syntax(files=[os.path.join(path_out, file_out + ext_out)])


def get_parser():

    # Mandatory arguments
    parser = argparse.ArgumentParser(
        description='Tools to crop an image. Either through command line or GUI',
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip(".py"))

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        "-i",
        required=True,
        help='Input image. Example: t2.nii.gz',
        metavar=Metavar.file,
        )

    # Command line mandatory arguments only for CLI execution
    requiredCommandArguments = parser.add_argument_group("\nCOMMAND LINE RELATED MANDATORY ARGUMENTS")
    requiredCommandArguments.add_argument(
        "-o",
        help='Output image. This option is REQUIRED for the command line execution Example: t1.nii.gz',
        metavar=Metavar.str,
        required=False
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")
    optional.add_argument(
        "-g",
        type=int,
        help="0: Cropping via command line | 1: Cropping via GUI",
        choices=(0, 1),
        default=0,
    )
    optional.add_argument(
        "-v",
        type=int,
        help="1: display on, 0: display off (default)",
        required=False,
        choices=(0, 1),
        default = 1)
    # GUI optional argument
    guiOptionalArguments = parser.add_argument_group("\nGUI RELATED OPTIONAL ARGUMENTS")
    guiOptionalArguments.add_argument(
        "-r",
        type=int,
        help="Remove temporary files. Default = 1",
        required=False,
        choices=(0, 1))
    # Command line optional arguments
    commandOptionalArguments = parser.add_argument_group("\nCOMMAND LINE RELATED OPTIONAL ARGUMENTS")
    commandOptionalArguments.add_argument(
        "-m",
        help="Cropping around the mask",
        metavar=Metavar.file,
        required=False)
    commandOptionalArguments.add_argument(
        "-start",
        help='Start slices, ]0,1[: percentage, 0 & >1: slice number. Example: 40,30,5',
        metavar=Metavar.list,
        required = False)
    commandOptionalArguments.add_argument(
        "-end",
        help='End slices, ]0,1[: percentage, 0: last slice, >1: slice number, <0: last slice - value. '
             'Example: 60,100,10',
        metavar=Metavar.list,
        required = False)
    commandOptionalArguments.add_argument(
        "-dim",
        help='Dimension to crop, from 0 to n-1, default is 1. Example: 0,1,2',
        metavar=Metavar.list,
        required = False)
    commandOptionalArguments.add_argument(
        "-shift",
        help='adding shift when used with mask, default is 0. Example: 10,10,5',
        metavar=Metavar.list,
        required = False)
    commandOptionalArguments.add_argument(
        "-b",
        type=float,
        help="Replace voxels outside cropping region with background value. If both the -m and the -b flags are used, "
             "the image is croped \"exactly\" around the mask with a background (and not around a rectangle area "
             "including the mask). The shape of the image does not change.",
        metavar=Metavar.float,
        required=False)
    commandOptionalArguments.add_argument(
        "-bmax",
        help="Maximize the cropping of the image (provide -dim if you want to specify the dimensions).",
        metavar='',
        required=False)
    commandOptionalArguments.add_argument(
        "-ref",
        help='Crop input image based on reference image (works only for 3D images). Example: ref.nii.gz',
        metavar=Metavar.file,
        required = False)
    commandOptionalArguments.add_argument(
        "-mesh",
        help="Mesh to crop",
        metavar=Metavar.file,
        required=False)
    commandOptionalArguments.add_argument(
        "-rof",
        type=int,
        help="Remove output file created when cropping",
        required=False,
        default=0,
        choices=(0, 1))

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
    return ind_start, ind_end, list(range(dim))


def main(args=None):
    """
    Main function
    :param args:
    :return:
    """
    # get parser args
    if args is None:
        args = None if sys.argv[1:] else ['--help']
    parser = get_parser()
    arguments = parser.parse_args(args=args)

    # assigning variables to arguments
    input_filename = arguments.i
    cropping_with_gui = arguments.g

    # initialize ImageCropper
    cropper = ImageCropper(input_filename)
    cropper.verbose = arguments.v
    sct.init_sct(log_level=cropper.verbose, update=True)  # Update log level

    if cropping_with_gui:
        if arguments.r is not None:
            cropper.rm_tmp_files = arguments.r
        cropper.crop_with_gui()

    else:
        if arguments.o is not None:
            cropper.output_filename = arguments.o
        else:
            sct.printv("An output file needs to be specified using the command line")
            sys.exit(2)
        # Handling optional arguments
        if arguments.m is not None:
            cropper.mask = arguments.m
        if arguments.start is not None:
            cropper.start = (arguments.start).split(",")
        if arguments.start is not None:
            cropper.end = (arguments.end).split(",")
        if arguments.dim is not None:
            cropper.dim = (arguments.dim).split(",")
        if arguments.shift is not None:
            cropper.shift = (arguments.shift).split(",")
        if arguments.b is not None:
            cropper.background = arguments.b
        if arguments.bmax is not None:
            cropper.bmax = True
        if arguments.ref is not None:
            cropper.ref = arguments.ref
        if arguments.mesh is not None:
            cropper.mesh = arguments.mesh

        cropper.crop()


if __name__ == "__main__":
    sct.init_sct()
    main()

