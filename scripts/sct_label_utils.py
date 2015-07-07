#!/usr/bin/env python
#########################################################################################
#
# All sort of utilities for labels.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Cohen-Adad
# Modified: 2015-02-11
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: currently it seems like cross_radius is given in pixel instead of mm

from msct_parser import Parser
from msct_image import Image

import sys
import sct_utils as sct
import math


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.fname_label_output = 'labels.nii.gz'
        self.labels = []
        self.verbose = 1


class ProcessLabels(object):
    def __init__(self, fname_label, fname_output=None, fname_ref=None, cross_radius=5, dilate=False,
                 coordinates=None, verbose=1):
        self.image_input = Image(fname_label, verbose=verbose)

        if fname_ref is not None:
            self.image_ref = Image(fname_ref, verbose=verbose)

        self.fname_output = fname_output
        self.cross_radius = cross_radius
        self.dilate = dilate
        self.coordinates = coordinates
        self.verbose = verbose

    def process(self, type_process):
        if type_process == 'cross':
            self.output_image = self.cross()
        elif type_process == 'plan':
            self.output_image = self.plan(self.cross_radius, 100, 5)
        elif type_process == 'plan_ref':
            self.output_image = self.plan_ref()
        elif type_process == 'increment':
            self.output_image = self.increment_z_inverse()
        elif type_process == 'disks':
            self.output_image = self.labelize_from_disks()
        elif type_process == 'MSE':
            self.MSE()
            self.fname_output = None
        elif type_process == 'remove':
            self.output_image = self.remove_label()
        elif type_process == 'centerline':
            self.extract_centerline()
        elif type_process == 'display-voxel':
            self.display_voxel()
            self.fname_output = None
        elif type_process == 'create':
            self.output_image = self.create_label()
        elif type_process == 'add':
            self.output_image = self.create_label(add=True)
        elif type_process == 'diff':
            self.diff()
            self.fname_output = None
        elif type_process == 'dist-inter':  # second argument is in pixel distance
            self.distance_interlabels(5)
            self.fname_output = None
        elif type_process == 'cubic-to-point':
            self.output_image = self.cubic_to_point()
        else:
            sct.printv('Error: The chosen process is not available.',1,'error')

        # save the output image as minimized integers
        if self.fname_output is not None:
            self.output_image.setFileName(self.fname_output)
            if type_process != 'plan_ref':
                self.output_image.save('minimize_int')
            else: self.output_image.save()


    def cross(self):
        image_output = Image(self.image_input, self.verbose)
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(self.image_input.absolutepath)

        coordinates_input = self.image_input.getNonZeroCoordinates()
        d = self.cross_radius  # cross radius in pixel
        dx = d / px  # cross radius in mm
        dy = d / py

        # for all points with non-zeros neighbors, force the neighbors to 0
        for coord in coordinates_input:
            image_output.data[coord.x][coord.y][coord.z] = 0  # remove point on the center of the spinal cord
            image_output.data[coord.x][coord.y + dy][
                coord.z] = coord.value * 10 + 1  # add point at distance from center of spinal cord
            image_output.data[coord.x + dx][coord.y][coord.z] = coord.value * 10 + 2
            image_output.data[coord.x][coord.y - dy][coord.z] = coord.value * 10 + 3
            image_output.data[coord.x - dx][coord.y][coord.z] = coord.value * 10 + 4

            # dilate cross to 3x3
            if self.dilate:
                image_output.data[coord.x - 1][coord.y + dy - 1][coord.z] = image_output.data[coord.x][coord.y + dy - 1][coord.z] = \
                    image_output.data[coord.x + 1][coord.y + dy - 1][coord.z] = image_output.data[coord.x + 1][coord.y + dy][coord.z] = \
                    image_output.data[coord.x + 1][coord.y + dy + 1][coord.z] = image_output.data[coord.x][coord.y + dy + 1][coord.z] = \
                    image_output.data[coord.x - 1][coord.y + dy + 1][coord.z] = image_output.data[coord.x - 1][coord.y + dy][coord.z] = \
                    image_output.data[coord.x][coord.y + dy][coord.z]
                image_output.data[coord.x + dx - 1][coord.y - 1][coord.z] = image_output.data[coord.x + dx][coord.y - 1][coord.z] = \
                    image_output.data[coord.x + dx + 1][coord.y - 1][coord.z] = image_output.data[coord.x + dx + 1][coord.y][coord.z] = \
                    image_output.data[coord.x + dx + 1][coord.y + 1][coord.z] = image_output.data[coord.x + dx][coord.y + 1][coord.z] = \
                    image_output.data[coord.x + dx - 1][coord.y + 1][coord.z] = image_output.data[coord.x + dx - 1][coord.y][coord.z] = \
                    image_output.data[coord.x + dx][coord.y][coord.z]
                image_output.data[coord.x - 1][coord.y - dy - 1][coord.z] = image_output.data[coord.x][coord.y - dy - 1][coord.z] = \
                    image_output.data[coord.x + 1][coord.y - dy - 1][coord.z] = image_output.data[coord.x + 1][coord.y - dy][coord.z] = \
                    image_output.data[coord.x + 1][coord.y - dy + 1][coord.z] = image_output.data[coord.x][coord.y - dy + 1][coord.z] = \
                    image_output.data[coord.x - 1][coord.y - dy + 1][coord.z] = image_output.data[coord.x - 1][coord.y - dy][coord.z] = \
                    image_output.data[coord.x][coord.y - dy][coord.z]
                image_output.data[coord.x - dx - 1][coord.y - 1][coord.z] = image_output.data[coord.x - dx][coord.y - 1][coord.z] = \
                    image_output.data[coord.x - dx + 1][coord.y - 1][coord.z] = image_output.data[coord.x - dx + 1][coord.y][coord.z] = \
                    image_output.data[coord.x - dx + 1][coord.y + 1][coord.z] = image_output.data[coord.x - dx][coord.y + 1][coord.z] = \
                    image_output.data[coord.x - dx - 1][coord.y + 1][coord.z] = image_output.data[coord.x - dx - 1][coord.y][coord.z] = \
                    image_output.data[coord.x - dx][coord.y][coord.z]

        return image_output

    def plan(self, width, offset=0, gap=1):
        """
        This function creates a plan of thickness="width" and changes its value with an offset and a gap between labels.
        """
        image_output = Image(self.image_input, self.verbose)
        image_output.data *= 0
        coordinates_input = self.image_input.getNonZeroCoordinates()

        # for all points with non-zeros neighbors, force the neighbors to 0
        for coord in coordinates_input:
            image_output.data[:,:,coord.z-width:coord.z+width] = offset + gap * coord.value

        return image_output

    def plan_ref(self):
        """
        This function generate a plan in the reference space for each label present in the input image
        """

        image_output = Image(self.image_ref, self.verbose)
        image_output.data *= 0

        image_input_neg = Image(self.image_input, self.verbose).copy()
        image_input_pos = Image(self.image_input, self.verbose).copy()
        image_input_neg.data *=0
        image_input_pos.data *=0
        X, Y, Z = (self.image_input.data< 0).nonzero()
        for i in range(len(X)):
            image_input_neg.data[X[i], Y[i], Z[i]] = -self.image_input.data[X[i], Y[i], Z[i]] # in order to apply getNonZeroCoordinates
        X_pos, Y_pos, Z_pos = (self.image_input.data> 0).nonzero()
        for i in range(len(X_pos)):
            image_input_pos.data[X_pos[i], Y_pos[i], Z_pos[i]] = self.image_input.data[X_pos[i], Y_pos[i], Z_pos[i]]

        coordinates_input_neg = image_input_neg.getNonZeroCoordinates()
        coordinates_input_pos = image_input_pos.getNonZeroCoordinates()

        image_output.changeType('float32')
        for coord in coordinates_input_neg:
            image_output.data[:, :, coord.z] = -coord.value #PB: takes the int value of coord.value
        for coord in coordinates_input_pos:
            image_output.data[:, :, coord.z] = coord.value

        return image_output
    def cubic_to_point(self):
        """
        This function calculates the center of mass of each group of labels and returns a file of same size with only a label by group at the center of mass.
        It is to be used after applying homothetic warping field to a label file as the labels will be dilated.
        :return:
        """
        from scipy import ndimage
        from numpy import array,mean
        data = self.image_input.data


        # pb: doesn't work if several groups have same value
        image_output = self.image_input.copy()
        data_output = image_output.data
        data_output *= 0
        coordinates = self.image_input.getNonZeroCoordinates(sorting='value')
        #list of present values
        list_values = []
        for i,coord in enumerate(coordinates):
            if i == 0 or coord.value != coordinates[i-1].value:
                list_values.append(coord.value)

        # make list of group of labels coordinates per value
        list_group_labels = []
        list_barycenter = []
        for i in range(0, len(list_values)):
            #mean_coord = mean(array([[coord.x, coord.y, coord.z] for coord in coordinates if coord.value==i]))
            list_group_labels.append([])
            list_group_labels[i] = [coordinates[j] for j in range(len(coordinates)) if coordinates[j].value == list_values[i]]
            # find barycenter: first define each case as a coordinate instance then calculate the value
            list_barycenter.append([0,0,0,0])
            sum_x = 0
            sum_y = 0
            sum_z = 0
            for j in range(len(list_group_labels[i])):
                sum_x += list_group_labels[i][j].x
                sum_y += list_group_labels[i][j].y
                sum_z += list_group_labels[i][j].z
            list_barycenter[i][0] = int(round(sum_x/len(list_group_labels[i])))
            list_barycenter[i][1] = int(round(sum_y/len(list_group_labels[i])))
            list_barycenter[i][2] = int(round(sum_z/len(list_group_labels[i])))
            list_barycenter[i][3] = list_group_labels[i][0].value

        # put value of group at each center of mass
        for i in range(len(list_values)):
            data_output[list_barycenter[i][0],list_barycenter[i][1], list_barycenter[i][2]] = list_barycenter[i][3]

        return image_output

        # Process to use if one wants to calculate the center of mass of a group of labels ordered by volume (and not by value)
        # image_output = self.image_input.copy()
        # data_output = image_output.data
        # data_output *= 0
        # nx = image_output.data.shape[0]
        # ny = image_output.data.shape[1]
        # nz = image_output.data.shape[2]
        # print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
        #
        # z_centerline = [iz for iz in range(0, nz, 1) if data[:,:,iz].any() ]
        # nz_nonz = len(z_centerline)
        # if nz_nonz==0 :
        #     print '\nERROR: Label file is empty'
        #     sys.exit()
        # x_centerline = [0 for iz in range(0, nz_nonz, 1)]
        # y_centerline = [0 for iz in range(0, nz_nonz, 1)]
        # print '\nGet center of mass for each slice of the label file ...'
        # for iz in xrange(len(z_centerline)):
        #     x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(array(data[:,:,z_centerline[iz]]))
        #
        # ## Calculate mean coordinate according to z for each cube of labels:
        # list_cube_labels_x = [[]]
        # list_cube_labels_y = [[]]
        # list_cube_labels_z = [[]]
        # count = 0
        # for i in range(nz_nonz-1):
        #     # Make a list of group of slices that contains a non zero value
        #     # check if the group is only one slice of height (at first slice)
        #     if i==0 and z_centerline[i] - z_centerline[i+1] != -1:
        #         list_cube_labels_z[count].append(z_centerline[i])
        #         list_cube_labels_x[count].append(x_centerline[i])
        #         list_cube_labels_y[count].append(y_centerline[i])
        #         list_cube_labels_z.append([])
        #         list_cube_labels_x.append([])
        #         list_cube_labels_y.append([])
        #         count += 1
        #     # check if the group is only one slice of height (in the middle)
        #     if i>0 and z_centerline[i-1] - z_centerline[i] != -1 and z_centerline[i] - z_centerline[i+1] != -1:
        #         list_cube_labels_z[count].append(z_centerline[i])
        #         list_cube_labels_x[count].append(x_centerline[i])
        #         list_cube_labels_y[count].append(y_centerline[i])
        #         list_cube_labels_z.append([])
        #         list_cube_labels_x.append([])
        #         list_cube_labels_y.append([])
        #         count += 1
        #     if z_centerline[i] - z_centerline[i+1] == -1:
        #         # Verify if the value has already been recovered and add if not
        #         #If the group is empty add first value do not if it is not empty as it will copy it for a second time
        #         if len(list_cube_labels_z[count]) == 0 :#or list_cube_labels[count][-1] != z_centerline[i]:
        #             list_cube_labels_z[count].append(z_centerline[i])
        #             list_cube_labels_x[count].append(x_centerline[i])
        #             list_cube_labels_y[count].append(y_centerline[i])
        #         list_cube_labels_z[count].append(z_centerline[i+1])
        #         list_cube_labels_x[count].append(x_centerline[i+1])
        #         list_cube_labels_y[count].append(y_centerline[i+1])
        #         if i+2 < nz_nonz-1 and z_centerline[i+1] - z_centerline[i+2] != -1:
        #             list_cube_labels_z.append([])
        #             list_cube_labels_x.append([])
        #             list_cube_labels_y.append([])
        #             count += 1
        #
        # z_label_mean = [0 for i in range(len(list_cube_labels_z))]
        # x_label_mean = [0 for i in range(len(list_cube_labels_z))]
        # y_label_mean = [0 for i in range(len(list_cube_labels_z))]
        # v_label_mean = [0 for i in range(len(list_cube_labels_z))]
        # for i in range(len(list_cube_labels_z)):
        #     for j in range(len(list_cube_labels_z[i])):
        #         z_label_mean[i] += list_cube_labels_z[i][j]
        #         x_label_mean[i] += list_cube_labels_x[i][j]
        #         y_label_mean[i] += list_cube_labels_y[i][j]
        #     z_label_mean[i] = int(round(z_label_mean[i]/len(list_cube_labels_z[i])))
        #     x_label_mean[i] = int(round(x_label_mean[i]/len(list_cube_labels_x[i])))
        #     y_label_mean[i] = int(round(y_label_mean[i]/len(list_cube_labels_y[i])))
        #     # We suppose that the labels' value of the group is the value of the barycentre
        #     v_label_mean[i] = data[x_label_mean[i],y_label_mean[i], z_label_mean[i]]
        #
        # ## Put labels of value one into mean coordinates
        # for i in range(len(z_label_mean)):
        #     data_output[x_label_mean[i],y_label_mean[i], z_label_mean[i]] = v_label_mean[i]
        #
        # return image_output

    def increment_z_inverse(self):
        """
        This function increments all the labels present in the input image, inversely ordered by Z.
        Therefore, labels are incremented from top to bottom, assuming a RPI orientation
        Labels are assumed to be non-zero.
        """
        image_output = Image(self.image_input, self.verbose)
        image_output.data *= 0
        coordinates_input = self.image_input.getNonZeroCoordinates(sorting='z', reverse_coord=True)

        # for all points with non-zeros neighbors, force the neighbors to 0
        for i, coord in enumerate(coordinates_input):
            image_output.data[coord.x, coord.y, coord.z] = i + 1

        return image_output

    def labelize_from_disks(self):
        """
        This function creates an image with regions labelized depending on values from reference.
        Typically, user inputs an segmentation image, and labels with disks position, and this function produces
        a segmentation image with vertebral levels labelized.
        Labels are assumed to be non-zero and incremented from top to bottom, assuming a RPI orientation
        """
        image_output = Image(self.image_input, self.verbose)
        image_output.data *= 0
        coordinates_input = self.image_input.getNonZeroCoordinates()
        coordinates_ref = self.image_ref.getNonZeroCoordinates(sorting='value')

        # for all points in input, find the value that has to be set up, depending on the vertebral level
        for i, coord in enumerate(coordinates_input):
            for j in range(0, len(coordinates_ref)-1):
                if coordinates_ref[j+1].z < coord.z <= coordinates_ref[j].z:
                    image_output.data[coord.x, coord.y, coord.z] = coordinates_ref[j].value

        return image_output

    def symmetrizer(self, side='left'):
        """
        This function symmetrize the input image. One side of the image will be copied on the other side. We assume a
        RPI orientation.
        :param side: string 'left' or 'right'. Side that will be copied on the other side.
        :return:
        """
        image_output = Image(self.image_input, self.verbose)

        image_output[0:]

        """inspiration: (from atlas creation matlab script)
        temp_sum = temp_g + temp_d;
        temp_sum_flip = temp_sum(end:-1:1,:);
        temp_sym = (temp_sum + temp_sum_flip) / 2;

        temp_g(1:end / 2,:) = 0;
        temp_g(1 + end / 2:end,:) = temp_sym(1 + end / 2:end,:);
        temp_d(1:end / 2,:) = temp_sym(1:end / 2,:);
        temp_d(1 + end / 2:end,:) = 0;

        tractsHR
        {label_l}(:,:, num_slice_ref) = temp_g;
        tractsHR
        {label_r}(:,:, num_slice_ref) = temp_d;
        """

        return image_output

    def MSE(self, threshold_mse=0):
        """
        This function computes the Mean Square Distance Error between two sets of labels (input and ref).
        Moreover, a warning is generated for each label mismatch.
        If the MSE is above the threshold provided (by default = 0mm), a log is reported with the filenames considered here.
        """
        coordinates_input = self.image_input.getNonZeroCoordinates()
        coordinates_ref = self.image_ref.getNonZeroCoordinates()

        # check if all the labels in both the images match
        if len(coordinates_input) != len(coordinates_ref):
            sct.printv('ERROR: labels mismatch', 1, 'warning')
        for coord in coordinates_input:
            if round(coord.value) not in [round(coord_ref.value) for coord_ref in coordinates_ref]:
                sct.printv('ERROR: labels mismatch', 1, 'warning')
        for coord_ref in coordinates_ref:
            if round(coord_ref.value) not in [round(coord.value) for coord in coordinates_input]:
                sct.printv('ERROR: labels mismatch', 1, 'warning')

        result = 0.0
        for coord in coordinates_input:
            for coord_ref in coordinates_ref:
                if round(coord_ref.value) == round(coord.value):
                    result += (coord_ref.z - coord.z) ** 2
                    break
        result = math.sqrt(result / len(coordinates_input))
        sct.printv('MSE error in Z direction = ' + str(result) + ' mm')

        if result > threshold_mse:
            f = open(self.image_input.path + 'error_log_' + self.image_input.file_name + '.txt', 'w')
            f.write(
                'The labels error (MSE) between ' + self.image_input.file_name + ' and ' + self.image_ref.file_name + ' is: ' + str(
                    result))
            f.close()

        return result

    def create_label(self, add=False):
        """
        This function create an image with labels listed by the user.
        This method works only if the user inserted correct coordinates.

        self.coordinates is a list of coordinates (class in msct_types).
        a Coordinate contains x, y, z and value.
        If only one label is to be added, coordinates must be completed with '[]'
        examples:
        For one label:  object_define=ProcessLabels( fname_label, coordinates=[coordi]) where coordi is a 'Coordinate' object from msct_types
        For two labels: object_define=ProcessLabels( fname_label, coordinates=[coordi1, coordi2]) where coordi1 and coordi2 are 'Coordinate' objects from msct_types
        """
        image_output = self.image_input.copy()
        if not add:
            image_output.data *= 0

        # loop across labels
        for i, coord in enumerate(self.coordinates):
            # display info
            sct.printv('Label #' + str(i) + ': ' + str(coord.x) + ',' + str(coord.y) + ',' + str(coord.z) + ' --> ' +
                       str(coord.value), 1)
            image_output.data[coord.x, coord.y, coord.z] = coord.value

        return image_output

    def remove_label(self):
        """
        This function compares two label images and remove any labels in input image that are not in reference image.
        """
        image_output = Image(self.image_input, self.verbose)
        coordinates_input = self.image_input.getNonZeroCoordinates()
        coordinates_ref = self.image_ref.getNonZeroCoordinates()

        for coord in coordinates_input:
            value = self.image_input.data[coord.x, coord.y, coord.z]
            isInRef = False
            for coord_ref in coordinates_ref:
                # the following line could make issues when down sampling input, for example 21,00001 not = 21,0
                if abs(coord.value - coord_ref.value) < 0.5:
                    image_output.data[coord.x, coord.y, coord.z] = int(round(coord_ref.value))
                    isInRef = True
            if isInRef == False:
                image_output.data[coord.x, coord.y, coord.z] = 0

        return image_output

    def extract_centerline(self):
        """
        This function write a text file with the coordinates of the centerline.
        The image is suppose to be RPI
        """
        coordinates_input = self.image_input.getNonZeroCoordinates(sorting='z')

        fo = open(self.fname_output, "wb")
        for coord in coordinates_input:
            line = (coord.x,coord.y, coord.z)
            fo.write("%i %i %i\n" % line)
        fo.close()

    def display_voxel(self):
        """
        This function displays all the labels that are contained in the input image.
        The image is suppose to be RPI to display voxels. But works also for other orientations
        """
        coordinates_input = self.image_input.getNonZeroCoordinates(sorting='z')
        useful_notation = ''
        for coord in coordinates_input:
            print 'Position=(' + str(coord.x) + ',' + str(coord.y) + ',' + str(coord.z) + ') -- Value= ' + str(coord.value)
            if useful_notation != '':
                useful_notation = useful_notation + ':'
            useful_notation = useful_notation + str(coord.x) + ',' + str(coord.y) + ',' + str(coord.z) + ',' + str(coord.value)
        print 'Useful notation:'
        print useful_notation
        return coordinates_input

    def diff(self):
        """
        This function detects any label mismatch between input image and reference image
        """
        coordinates_input = self.image_input.getNonZeroCoordinates()
        coordinates_ref = self.image_ref.getNonZeroCoordinates()

        print "Label in input image that are not in reference image:"
        for coord in coordinates_input:
            isIn = False
            for coord_ref in coordinates_ref:
                if coord.value == coord_ref.value:
                    isIn = True
                    break
            if not isIn:
                print coord.value

        print "Label in ref image that are not in input image:"
        for coord_ref in coordinates_ref:
            isIn = False
            for coord in coordinates_input:
                if coord.value == coord_ref.value:
                    isIn = True
                    break
            if not isIn:
                print coord_ref.value

    def distance_interlabels(self, max_dist):
        """
        This function calculates the distances between each label in the input image.
        If a distance is larger than max_dist, a warning message is displayed.
        """
        coordinates_input = self.image_input.getNonZeroCoordinates()

        # for all points with non-zeros neighbors, force the neighbors to 0
        for i in range(0, len(coordinates_input) - 1):
            dist = math.sqrt((coordinates_input[i].x - coordinates_input[i+1].x)**2 + (coordinates_input[i].y - coordinates_input[i+1].y)**2 + (coordinates_input[i].z - coordinates_input[i+1].z)**2)
            if dist < max_dist:
                print 'Warning: the distance between label ' + str(i) + '[' + str(coordinates_input[i].x) + ',' + str(coordinates_input[i].y) + ',' + str(
                    coordinates_input[i].z) + ']=' + str(coordinates_input[i].value) + ' and label ' + str(i+1) + '[' + str(
                    coordinates_input[i+1].x) + ',' + str(coordinates_input[i+1].y) + ',' + str(coordinates_input[i+1].z) + ']=' + str(
                    coordinates_input[i+1].value) + ' is larger than ' + str(max_dist) + '. Distance=' + str(dist)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = Param()
    param_default = Param()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Utility function for labels.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="labels or image to create labels on. Must be 3D.",
                      mandatory=True,
                      example="t2_labels.nii.gz")
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="output volume.",
                      mandatory=False,
                      example="t2_labels_cross.nii.gz",
                      default_value="labels.nii.gz")
    parser.add_option(name="-t",
                      type_value="str",
                      description="""process:\ncross: create a cross. Must use flag "-c"\nremove: remove labels. Must use flag "-r"\ndisplay-voxel: display all labels in file\ncreate: create labels. Must use flag "-x" to list labels\nadd: add label to an existing image (-i).\nincrement: increment labels from top to bottom (in z direction, suppose RPI orientation)\nMSE: compute Mean Square Error between labels input and reference input "-r"\ncubic-to-point: transform each volume of labels by value into a discrete single voxel label. """,
                      mandatory=True,
                      example="create")
    parser.add_option(name="-x",
                      type_value=[[':'], 'Coordinate'],
                      description="""labels x,y,z,v. Use ":" if you have multiple labels.\nx: x-coordinates\ny: y-coordinates\nz: z-coordinates\nv: value of label""",
                      mandatory=False,
                      example="1,5,2,6:3,7,2,1:3,7,9,32")
    parser.add_option(name="-r",
                      type_value="file",
                      description="reference volume for label removing.",
                      mandatory=False)
    parser.add_option(name="-c",
                      type_value="int",
                      description="cross radius in mm (default=5mm).",
                      mandatory=False)
    parser.add_option(name="-d",
                      type_value=None,
                      description="dilatation bool for cross generation ('-c' option).",
                      mandatory=False)
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="verbose. Default=" + str(param_default.verbose),
                      mandatory=False,
                      example=['0', '1'])
    arguments = parser.parse(sys.argv[1:])

    input_filename = arguments["-i"]
    process_type = arguments["-t"]
    input_fname_output = None
    input_fname_ref = None
    input_cross_radius = 5
    input_dilate = False
    input_coordinates = None
    input_verbose = '1'
    input_fname_output = arguments["-o"]
    if "-r" in arguments:
        input_fname_ref = arguments["-r"]
    if "-x" in arguments:
        input_coordinates = arguments["-x"]
    if "-c" in arguments:
        input_cross_radius = arguments["-c"]
    if "-d" in arguments:
        input_dilate = arguments["-d"]
    if "-v" in arguments:
        input_verbose = arguments["-v"]
    processor = ProcessLabels(input_filename, fname_output=input_fname_output, fname_ref=input_fname_ref, cross_radius=input_cross_radius, dilate=input_dilate, coordinates=input_coordinates, verbose=input_verbose)
    processor.process(process_type)
