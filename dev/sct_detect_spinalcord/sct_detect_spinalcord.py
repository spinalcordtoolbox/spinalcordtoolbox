#!/usr/bin/env python
#########################################################################################
#
# Spinal Cord Automatic Detection
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Modified: 2015-07-27
#
# About the license: see the file LICENSE
#########################################################################################

import sys
from msct_base_classes import BaseScript, Algorithm
from msct_parser import Parser
from msct_image import Image
import os
import sct_utils as sct
import numpy as np
from sct_straighten_spinalcord import smooth_centerline
import sct_convert as conv


def smooth_minimal_path(img, nb_pixels=1):
    """
    Function intended to smooth the minimal path result in the R-L/A-P directions with a gaussian filter
    of a kernel of size nb_pixels
    :param img: Image to be smoothed (is intended to be minimal path image)
    :param nb_pixels: kernel size of the gaussian filter
    :return: returns a smoothed image
    """

    nx, ny, nz, nt, px, py, pz, pt = img.dim
    from scipy.ndimage.filters import gaussian_filter
    raw_orientation = img.change_orientation()

    img.data = gaussian_filter(img.data, [nb_pixels/px, nb_pixels/py, 0])

    img.change_orientation(raw_orientation)
    return img


def symmetry_detector_right_left(data, cropped_xy=0):
    """
    This function
    :param img: input image used for the algorithm
    :param cropped_xy: 1 when we want to crop around the center for the correlation, 0 when not
    :return: returns an image that is the body symmetry (correlation between left and right side of the image)
    """
    from scipy.ndimage.filters import gaussian_filter

    # Change orientation and define variables for
    data = np.squeeze(data)
    dim = data.shape

    img_data = gaussian_filter(data, [0, 5, 5])

    # Cropping around center of image to remove side noise
    if cropped_xy:
        x_mid = np.round(dim[0]/2)
        x_crop_min = int(x_mid - (0.25/2)*dim[0])
        x_crop_max = int(x_mid + (0.25/2)*dim[0])

        img_data[0:x_crop_min,:,:] = 0
        img_data[x_crop_max:-1,:,:] = 0

    # Acquiring a slice and inverted slice for correlation
    slice_p = np.squeeze(np.sum(img_data, 1))
    slice_p_reversed = np.flipud(slice_p)

    # initialise containers for correlation
    m, n = slice_p.shape
    cross_corr = ((2*m)-1, n)
    cross_corr = np.zeros(cross_corr)
    for iz in range(0, np.size(slice_p[1])):
        corr1 = slice_p[:, iz]
        corr2 = slice_p_reversed[:, iz]
        cross_corr[:, iz] = np.double(np.correlate(corr1, corr2, "full"))
        max_value = np.max(cross_corr[:, iz])
        if max_value == 0:
            cross_corr[:, iz] = 0
        else:
            cross_corr[:, iz] = cross_corr[:, iz]/max_value
    data_out = np.zeros((dim[0], dim[2]))
    index1 = np.round(np.linspace(0,2*m-3, m))
    index2 = np.round(np.linspace(1,2*m-2, m))
    for i in range(0,m):
        indx1 = int(index1[i])
        indx2 = int(index2[i])
        out1 = cross_corr[indx1, :]
        out2 = cross_corr[indx2, :]
        data_out[i, :] = 0.5*(out1 + out2)
    result = np.hstack([data_out[:, np.newaxis, :] for i in range(0, dim[1])])

    return result


def normalize_array_histogram(array):
    """
    Equalizes the data in array
    :param array:
    :return:
    """
    array_min = np.amin(array)
    array -= array_min
    array_max = np.amax(array)
    array /= array_max

    return array


def get_minimum_path(data, smooth_factor=np.sqrt(2), invert=1, verbose=1, debug=0):
    """
    This method returns the minimal path of the image
    :param data: input data of the image
    :param smooth_factor:factor used to smooth the directions that are not up-down
    :param invert: inverts the image data for the algorithm. The algorithm works better if the image data is inverted
    :param verbose:
    :param debug:
    :return:
    """
    [m, n, p] = data.shape
    max_value = np.amax(data)
    if invert:
        data=max_value-data
    J1 = np.ones([m, n, p])*np.inf
    J2 = np.ones([m, n, p])*np.inf
    J1[:, :, 0] = 0
    for row in range(1, p):
        pJ = J1[:, :, row-1]
        cP = np.squeeze(data[1:-2, 1:-2, row])
        VI = np.dstack((cP*smooth_factor, cP*smooth_factor, cP, cP*smooth_factor, cP*smooth_factor))

        Jq = np.dstack((pJ[0:-3, 1:-2], pJ[1:-2, 0:-3], pJ[1:-2, 1:-2], pJ[1:-2, 2:-1], pJ[2:-1, 1:-2]))
        J1[1:-2, 1:-2, row] = np.min(Jq+VI, 2)
        pass

    J2[:, :, p-1] = 0
    for row in range(p-2, -1, -1):
        pJ = J2[:, :, row+1]
        cP = np.squeeze(data[1:-2, 1:-2, row])
        VI = np.dstack((cP*smooth_factor, cP*smooth_factor, cP, cP*smooth_factor, cP*smooth_factor))

        Jq = np.dstack((pJ[0:-3, 1:-2], pJ[1:-2, 0:-3], pJ[1:-2, 1:-2], pJ[1:-2, 2:-1], pJ[2:-1, 1:-2]))
        J2[1:-2, 1:-2, row] = np.min(Jq+VI, 2)
        pass

    result = J1+J2
    if invert:
        percent = np.percentile(result, 50)
        result[result > percent] = percent

        result_min = np.amin(result)
        result_max = np.amax(result)
        result = np.divide(np.subtract(result, result_min), result_max)
        result_max = np.amax(result)

    result = 1-result

    result[result == np.inf] = 0
    result[result == np.nan] = 0

    return result, J1, J2


def get_minimum_path_nii(fname):
    from msct_image import Image
    data=Image(fname)
    vesselness_data = data.data
    raw_orient=data.change_orientation()
    result ,J1, J2 = get_minimum_path(data.data, invert=1)
    data.data = result
    data.change_orientation(raw_orient)
    data.file_name += '_minimalpath'
    data.save()


def ind2sub(array_shape, ind):
    """

    :param array_shape: shape of the array
    :param ind: index number
    :return: coordinates equivalent to the index number for a given array shape
    """
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1])  # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols


def get_centerline(data, dim):
    """
    This function extracts the highest value per slice from a minimal path image
    and builds the centerline from it
    :param data:
    :param dim:
    :return:
    """
    centerline = np.zeros(dim)

    data[data == np.inf] = 0
    data[data == np.nan] = 0

    for iz in range(0, dim[2]):
        ind = np.argmax(data[:, :, iz])
        X, Y = ind2sub(data[:, :, iz].shape,ind)
        centerline[X,Y,iz] = 1

    return centerline


class SymmetryDetector(Algorithm):
    def __init__(self, input_image, contrast=None, verbose=0, direction="lr", nb_sections=1, crop_xy=1):
        super(SymmetryDetector, self).__init__(input_image)
        self._contrast = contrast
        self._verbose = verbose
        self.direction = direction
        self.nb_sections = nb_sections
        self.crop_xy = crop_xy

    @property
    def contrast(self):
        return self._contrast

    @contrast.setter
    def contrast(self, value):
        if value in ['t1', 't2']:
            self._contrast = value
        else:
            raise Exception('ERROR: contrast value must be t1 or t2')

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        if value in [0, 1]:
            self._verbose = value
        else:
            raise Exception('ERROR: verbose value must be an integer and equal to 0 or 1')

    def execute(self):
        """
        This method executes the symmetry detection
        :return: returns the symmetry data
        """
        img = Image(self.input_image)
        raw_orientation = img.change_orientation()
        data = np.squeeze(img.data)
        dim = data.shape
        section_length = dim[1]/self.nb_sections

        result = np.zeros(dim)

        for i in range(0, self.nb_sections):
            if (i+1)*section_length > dim[1]:
                y_length = (i+1)*section_length - ((i+1)*section_length - dim[1])
                result[:, i*section_length:i*section_length + y_length, :] = symmetry_detector_right_left(data[:, i*section_length:i*section_length + y_length, :],  cropped_xy=self.crop_xy)
            sym = symmetry_detector_right_left(data[:, i*section_length:(i+1)*section_length, :], cropped_xy=self.crop_xy)
            result[:, i*section_length:(i+1)*section_length, :] = sym

        result_image = Image(img)
        if len(result_image.data) == 4:
            result_image.data = result[:,:,:,np.newaxis]
        else:
            result_image.data = result

        result_image.change_orientation(raw_orientation)

        return result_image.data


class ScadScript(BaseScript):
    def __init__(self):
        super(ScadScript, self).__init__()

    @staticmethod
    def get_parser():
        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('''This program automatically detect the spinal cord in a MR image and output a centerline of the spinal cord.''')
        parser.add_option(name="-i",
                          type_value="file",
                          description="input image.",
                          mandatory=True,
                          example="t2.nii.gz")
        parser.add_option(name="-t",
                          type_value="multiple_choice",
                          description="type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark",
                          mandatory=True,
                          example=['t1', 't2'])
        parser.usage.addSection("General options")
        parser.add_option(name="-o",
                          type_value="string",
                          description="Centerline file name (result file name)",
                          mandatory=False,
                          example="out.nii.gz")
        parser.add_option(name="-sym",
                          type_value="multiple_choice",
                          description="Uses right-left symmetry of the image to improve accuracy.",
                          mandatory=False,
                          default_value="0",
                          example=['0', '1'])
        parser.add_option(name="-sym_exp",
                          type_value="int",
                          description="Weight symmetry value (only use with flag -sym). Minimum weight: 0, maximum weight: 100.",
                          mandatory=False,
                          default_value="10")
        parser.add_option(name="-r",
                          type_value="multiple_choice",
                          description= "Removes the temporary folder and debug folder used for the algorithm at the end of execution",
                          mandatory=False,
                          default_value="0",
                          example=['0', '1'])
        parser.add_option(name="-sc_rad",
                          type_value="int",
                          description="Gives approximate radius of spinal cord to help the algorithm",
                          mandatory=False,
                          default_value="4",
                          example="4")

        parser.add_option(name="-v",
                          type_value="multiple_choice",
                          description="1: display on, 0: display off (default)",
                          mandatory=False,
                          example=["0", "1"],
                          default_value="1")
        parser.add_option(name="-h",
                          type_value=None,
                          description="display this help",
                          mandatory=False)
        return parser


class SCAD(Algorithm):
    def __init__(self, input_image, contrast=None, verbose=1, rm_tmp_file=0,output_filename=None, debug=0, vesselness_provided=0, minimum_path_exponent=100, enable_symmetry=0, symmetry_exponent=0, spinalcord_radius = 3):
        """
        Constructor for the automatic spinal cord detection
        :param output_filename: Name of the result file of the centerline detection. Must contain the extension (.nii / .nii.gz)
        :param input_image:
        :param contrast:
        :param verbose:
        :param rm_tmp_file:
        :param debug:
        :param produce_output: Produce output debug files,
        :param vesselness_provided: Activate if the vesselness filter image is already provided (to save time),
               the image is expected to be in the same folder as the input image
        :return:
        """
        super(SCAD, self).__init__(input_image, produce_output=1-rm_tmp_file)
        self._contrast = contrast
        self._verbose = verbose
        self.output_filename = input_image.file_name + "_centerline.nii.gz"
        if output_filename is not None:
            self.output_filename = output_filename
        self.rm_tmp_file = rm_tmp_file
        self.debug = debug
        self.vesselness_provided = vesselness_provided
        self.minimum_path_exponent = minimum_path_exponent
        self.enable_symmetry = enable_symmetry
        self.symmetry_exponent = symmetry_exponent
        self.spinalcord_radius = spinalcord_radius

        # attributes used in the algorithm
        self.raw_orientation = None
        self.raw_symmetry = None
        self.J1_min_path = None
        self.J2_min_path = None
        self.minimum_path_data = None
        self.minimum_path_powered = None
        self.smoothed_min_path = None
        self.spine_detect_data = None
        self.centerline_with_outliers = None

        self.debug_folder = None


    @property
    def contrast(self):
        return self._contrast

    @contrast.setter
    def contrast(self, value):
        if value in ['t1', 't2']:
            self._contrast = value
        else:
            raise Exception('ERROR: contrast value must be t1 or t2')

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        if value in [0, 1]:
            self._verbose = value
        else:
            raise Exception('ERROR: verbose value must be an integer and equal to 0 or 1')

    def produce_output_files(self):
        """
        Method used to output all debug files at the same time. To be used after the algorithm is executed

        :return:
        """
        path_tmp = sct.tmp_create(basename="scad_output")
        # getting input image header
        img = self.input_image.copy()

        # saving body symmetry
        img.data = self.raw_symmetry
        img.change_orientation(self.raw_orientation)
        img.file_name += "body_symmetry"
        img.save()

        # saving minimum paths
        img.data = self.minimum_path_data
        img.change_orientation(self.raw_orientation)
        img.file_name = "min_path"
        img.save()
        img.data = self.J1_min_path
        img.change_orientation(self.raw_orientation)
        img.file_name = "J1_min_path"
        img.save()
        img.data = self.J2_min_path
        img.change_orientation(self.raw_orientation)
        img.file_name = "J2_min_path"
        img.save()

        # saving minimum path powered
        img.data = self.minimum_path_powered
        img.change_orientation(self.raw_orientation)
        img.file_name = "min_path_powered_"+str(self.minimum_path_exponent)
        img.save()

        # saving smoothed min path
        img = self.smoothed_min_path.copy()
        img.change_orientation(self.raw_orientation)
        img.file_name = "min_path_power_"+str(self.minimum_path_exponent)+"_smoothed"
        img.save()

        # save symmetry_weighted_minimal_path
        img.data = self.spine_detect_data
        img.change_orientation(self.raw_orientation)
        img.file_name = "symmetry_weighted_minimal_path"
        img.save()

    def output_debug_file(self, img, data, file_name):
        """
        This method writes a nifti file that corresponds to a step in the algorithm for easy debug.
        The new nifti file uses the header from the the image passed as parameter
        :param data: data to be written to file
        :param file_name: filename...
        :return: None
        """
        if self.produce_output:
            current_folder = os.getcwd()
            os.chdir(self.debug_folder)
            try:
                img = Image(img)
                img.data = data
                img.change_orientation(self.raw_orientation)
                img.file_name = file_name
                img.save()
            except Exception, e:
                print e
            os.chdir(current_folder)

    def setup_debug_folder(self):
        """
        Sets up the folder for the step by step files for this algorithm
        The folder's absolute path can be found in the self.debug_folder property
        :return: None
        """
        if self.produce_output:
            self.debug_folder = sct.tmp_create()
            conv.convert(self.input_image.absolutepath, os.path.join(self.debug_folder, "raw.nii.gz"))

    def execute(self):
        curdir = os.getcwd()
        print('Execution of the SCAD algorithm in %s' % curdir)

        original_name = self.input_image.file_name
        vesselness_file_name = "imageVesselNessFilter.nii.gz"
        raw_file_name = "raw.nii"

        self.setup_debug_folder()

        if self.debug:
            import matplotlib.pyplot as plt # import for debug purposes

        # create tmp and copy input
        path_tmp = sct.tmp_create()
        conv.convert(self.input_image.absolutepath, path_tmp+raw_file_name)

        if self.vesselness_provided:
            sct.run('cp '+vesselness_file_name+' '+path_tmp+vesselness_file_name)
        os.chdir(path_tmp)

        # get input image information
        img = Image(raw_file_name)

        # save original orientation and change image to RPI
        self.raw_orientation = img.change_orientation()

        # get body symmetry
        if self.enable_symmetry:
            from msct_image import change_data_orientation
            sym = SymmetryDetector(raw_file_name, self.contrast, crop_xy=0)
            self.raw_symmetry = sym.execute()
            img.change_orientation(self.raw_orientation)
            self.output_debug_file(img, self.raw_symmetry, "body_symmetry")
            img.change_orientation()

        # vesselness filter
        if not self.vesselness_provided:
            sct.run('isct_vesselness -i '+raw_file_name+' -t ' + self._contrast+" -radius "+str(self.spinalcord_radius))

        # load vesselness filter data and perform minimum path on it
        img = Image(vesselness_file_name)
        self.output_debug_file(img, img.data, "Vesselness_Filter")
        img.change_orientation()
        self.minimum_path_data, self.J1_min_path, self.J2_min_path = get_minimum_path(img.data, invert=1, debug=1)
        self.output_debug_file(img, self.minimum_path_data, "minimal_path")
        self.output_debug_file(img, self.J1_min_path, "J1_minimal_path")
        self.output_debug_file(img, self.J2_min_path, "J2_minimal_path")

        # Apply an exponent to the minimum path
        self.minimum_path_powered = np.power(self.minimum_path_data, self.minimum_path_exponent)
        self.output_debug_file(img, self.minimum_path_powered, "minimal_path_power_"+str(self.minimum_path_exponent))

        # Saving in Image since smooth_minimal_path needs pixel dimensions
        img.data = self.minimum_path_powered

        # smooth resulting minimal path
        self.smoothed_min_path = smooth_minimal_path(img)
        self.output_debug_file(img, self.smoothed_min_path.data, "minimal_path_smooth")

        # normalise symmetry values between 0 and 1
        if self.enable_symmetry:
            normalised_symmetry = normalize_array_histogram(self.raw_symmetry)
            self.output_debug_file(img, self.smoothed_min_path.data, "minimal_path_smooth")

        # multiply normalised symmetry data with the minimum path result
            from msct_image import change_data_orientation
            self.spine_detect_data = np.multiply(self.smoothed_min_path.data, change_data_orientation(np.power(normalised_symmetry, self.symmetry_exponent), self.raw_orientation, "RPI"))
            self.output_debug_file(img, self.spine_detect_data, "symmetry_x_min_path")
            # extract the centerline from the minimal path image
            self.centerline_with_outliers = get_centerline(self.spine_detect_data, self.spine_detect_data.shape)
        else:
            # extract the centerline from the minimal path image
            self.centerline_with_outliers = get_centerline(self.smoothed_min_path.data, self.smoothed_min_path.data.shape)
        self.output_debug_file(img, self.centerline_with_outliers, "centerline_with_outliers")

        # saving centerline with outliers to have
        img.data = self.centerline_with_outliers
        img.change_orientation()
        img.file_name = "centerline_with_outliers"
        img.save()

        # use a b-spline to smooth out the centerline
        x, y, z, dx, dy, dz = smooth_centerline("centerline_with_outliers.nii.gz")

        # save the centerline
        nx, ny, nz, nt, px, py, pz, pt = img.dim
        img.data = np.zeros((nx, ny, nz))
        for i in range(0, np.size(x)-1):
            img.data[int(x[i]), int(y[i]), int(z[i])] = 1

        self.output_debug_file(img, img.data, "centerline")
        img.change_orientation(self.raw_orientation)
        img.file_name = "centerline"
        img.save()

        # copy back centerline
        os.chdir(curdir)
        conv.convert(os.path.join(path_tmp, img.file_name + img.ext), self.output_filename)
        if self.rm_tmp_file == 1:
            import shutil
            shutil.rmtree(path_tmp)

        print "To view the output with FSL :"
        sct.printv("fslview "+self.input_image.absolutepath+" "+self.output_filename+" -l Red", self.verbose, "info")


if __name__ == "__main__":
    parser = ScadScript.get_parser()

    arguments = parser.parse(sys.argv[1:])

    input_image = Image(arguments["-i"])
    contrast_type = arguments["-t"]

    scad = SCAD(input_image, contrast=contrast_type)

    if "-o" in arguments:
        scad.output_filename = arguments["-o"]
    # if "-p" in arguments:
    #     scad.produce_output = int(arguments["-p"])
    if "-r" in arguments:
        scad.rm_tmp_file = int(arguments["-r"])
    if "-sym" in arguments:
        scad.enable_symmetry = int(arguments["-sym"])
    if "-sym_exp" in arguments:
        scad.symmetry_exponent = int(arguments["-sym_exp"])
    if "-sc_rad" in arguments:
        scad.spinalcord_radius = int(arguments["-sc_rad"])
    scad.execute()
