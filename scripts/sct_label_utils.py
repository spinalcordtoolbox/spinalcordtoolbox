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

import os
import sys
import sct_utils as sct
import numpy as np
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
    def __init__(self, fname_label, type_process, fname_output=None, fname_ref=None, cross_radius=5, dilate=False,
                 coordinates=None, verbose='1'):
        self.image_input = Image(fname_label)

        if fname_ref is not None:
            self.image_ref = Image(fname_ref)

        self.type_process = type_process

        self.fname_output = fname_output
        self.cross_radius = cross_radius
        self.dilate = dilate
        self.coordinates = coordinates
        self.verbose = verbose

    def process(self):
        if self.type_process == 'cross':
            self.output_image = self.cross()
        elif self.type_process == 'plan':
            self.output_image = self.plan(self.cross_radius, 100, 5)
        elif self.type_process == 'plan_ref':
            self.output_image = self.plan_ref()
        elif self.type_process == 'increment':
            self.output_image = self.increment_z_inverse()
        elif self.type_process == 'MSE':
            self.MSE()
        elif self.type_process == 'remove':
            self.output_image = self.remove_label()
        elif self.type_process == 'disk':
            self.output_image = self.extract_disk_position(self.fname_output)
        elif self.type_process == 'centerline':
            self.extract_centerline()
        elif self.type_process == 'segmentation':
            self.extract_segmentation()
        elif self.type_process == 'display-voxel':
            self.display_voxel()
        elif self.type_process == 'create':
            self.output_image = self.create_label()
        elif self.type_process == 'diff':
            self.diff()
        elif self.type_process == 'dist-inter':  # second argument is in pixel distance
            self.distance_interlabels(5)

        # save the output image as minimized integers
        if self.fname_output is not None:
            self.output_image.setFileName(self.fname_output)
            self.output_image.save('minimize_int')


    def cross(self):
        image_output = Image(im_ref=self.image_input)
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(self.image_input.absolutepath)

        X, Y, Z = (self.image_input.data > 0).nonzero()
        number_of_labels = len(X)
        d = self.cross_radius  # cross radius in pixel
        dx = d / px  # cross radius in mm
        dy = d / py

        # for all points with non-zeros neighbors, force the neighbors to 0
        for i in range(0, number_of_labels):
            value = int(np.round(image_output.data[X[i]][Y[i]][Z[i]]))
            image_output.data[X[i]][Y[i]][Z[i]] = 0  # remove point on the center of the spinal cord
            image_output.data[X[i]][Y[i] + dy][
                Z[i]] = value * 10 + 1  # add point at distance from center of spinal cord
            image_output.data[X[i] + dx][Y[i]][Z[i]] = value * 10 + 2
            image_output.data[X[i]][Y[i] - dy][Z[i]] = value * 10 + 3
            image_output.data[X[i] - dx][Y[i]][Z[i]] = value * 10 + 4

            # dilate cross to 3x3
            if self.dilate:
                image_output.data[X[i] - 1][Y[i] + dy - 1][Z[i]] = image_output.data[X[i]][Y[i] + dy - 1][Z[i]] = \
                    image_output.data[X[i] + 1][Y[i] + dy - 1][Z[i]] = image_output.data[X[i] + 1][Y[i] + dy][Z[i]] = \
                    image_output.data[X[i] + 1][Y[i] + dy + 1][Z[i]] = image_output.data[X[i]][Y[i] + dy + 1][Z[i]] = \
                    image_output.data[X[i] - 1][Y[i] + dy + 1][Z[i]] = image_output.data[X[i] - 1][Y[i] + dy][Z[i]] = \
                    image_output.data[X[i]][Y[i] + dy][Z[i]]
                image_output.data[X[i] + dx - 1][Y[i] - 1][Z[i]] = image_output.data[X[i] + dx][Y[i] - 1][Z[i]] = \
                    image_output.data[X[i] + dx + 1][Y[i] - 1][Z[i]] = image_output.data[X[i] + dx + 1][Y[i]][Z[i]] = \
                    image_output.data[X[i] + dx + 1][Y[i] + 1][Z[i]] = image_output.data[X[i] + dx][Y[i] + 1][Z[i]] = \
                    image_output.data[X[i] + dx - 1][Y[i] + 1][Z[i]] = image_output.data[X[i] + dx - 1][Y[i]][Z[i]] = \
                    image_output.data[X[i] + dx][Y[i]][Z[i]]
                image_output.data[X[i] - 1][Y[i] - dy - 1][Z[i]] = image_output.data[X[i]][Y[i] - dy - 1][Z[i]] = \
                    image_output.data[X[i] + 1][Y[i] - dy - 1][Z[i]] = image_output.data[X[i] + 1][Y[i] - dy][Z[i]] = \
                    image_output.data[X[i] + 1][Y[i] - dy + 1][Z[i]] = image_output.data[X[i]][Y[i] - dy + 1][Z[i]] = \
                    image_output.data[X[i] - 1][Y[i] - dy + 1][Z[i]] = image_output.data[X[i] - 1][Y[i] - dy][Z[i]] = \
                    image_output.data[X[i]][Y[i] - dy][Z[i]]
                image_output.data[X[i] - dx - 1][Y[i] - 1][Z[i]] = image_output.data[X[i] - dx][Y[i] - 1][Z[i]] = \
                    image_output.data[X[i] - dx + 1][Y[i] - 1][Z[i]] = image_output.data[X[i] - dx + 1][Y[i]][Z[i]] = \
                    image_output.data[X[i] - dx + 1][Y[i] + 1][Z[i]] = image_output.data[X[i] - dx][Y[i] + 1][Z[i]] = \
                    image_output.data[X[i] - dx - 1][Y[i] + 1][Z[i]] = image_output.data[X[i] - dx - 1][Y[i]][Z[i]] = \
                    image_output.data[X[i] - dx][Y[i]][Z[i]]

        return image_output

    def plan(self, width, offset=0, gap=1):
        """
        This function creates a plan of thickness="width" and changes its value with an offset and a gap between labels.
        """
        image_output = Image(im_ref_zero=self.image_input)
        X, Y, Z = (self.image_input.data > 0).nonzero()

        # for all points with non-zeros neighbors, force the neighbors to 0
        for i in range(0, len(X)):
            value = int(image_output.data[X[i]][Y[i]][Z[i]])
            image_output.data[:, :, Z[i] - width:Z[i] + width] = offset + gap * value

        return image_output

    def plan_ref(self):
        """
        This function generate a plan in the reference space for each label present in the input image
        """
        image_output = Image(im_ref_zero=self.image_ref)
        X, Y, Z = (self.image_input.data != 0).nonzero()

        # for all points with non-zeros neighbors, force the neighbors to 0
        for i in range(0, len(X)):
            image_output.data[:, :, Z[i]] = self.image_input.data[X[i]][Y[i]][Z[i]]

        return image_output

    def increment_z_inverse(self):
        """
        This function increments all the labels present in the input image, inversely ordered by Z.
        Therefore, labels are incremented from top to bottom, assuming a RPI orientation
        Labels are assumed to be non-zero.
        """
        image_output = Image(im_ref_zero=self.image_input)
        X, Y, Z = (self.image_input.data > 0).nonzero()

        X_sort = [X[i] for i in Z.argsort()]
        X_sort.reverse()
        Y_sort = [Y[i] for i in Z.argsort()]
        Y_sort.reverse()
        Z_sort = [Z[i] for i in Z.argsort()]
        Z_sort.reverse()
        # for all points with non-zeros neighbors, force the neighbors to 0
        for i in range(0, len(Z_sort)):
            image_output.data[X_sort[i], Y_sort[i], Z_sort[i]] = i + 1

        return image_output

    def MSE(self, threshold_mse=0):
        """
        This function computes the Mean Square Distance Error between two sets of labels (input and ref).
        Moreover, a warning is generated for each label mismatch.
        If the MSE is above the threshold provided (by default = 0mm), a log is reported with the filenames considered here.
        """
        X, Y, Z = (self.image_input.data > 0).nonzero()
        data_labels = [[X[i], Y[i], Z[i], self.image_input.data[X[i], Y[i], Z[i]]] for i in range(0, len(X))]

        X_ref, Y_ref, Z_ref = (self.image_ref.data > 0).nonzero()
        ref_labels = [[X_ref[i], Y_ref[i], Z_ref[i], self.image_ref.data[X_ref[i], Y_ref[i], Z_ref[i]]] for i in
                      range(0, len(X_ref))]

        # check if all the labels in both the images match
        if len(X) != len(X_ref):
            sct.printv('ERROR: labels mismatch', 1, 'warning')
        for value in data_labels:
            if round(value[3]) not in [round(v[3]) for v in ref_labels]:
                sct.printv('ERROR: labels mismatch', 1, 'warning')
        for value in ref_labels:
            if round(value[3]) not in [round(v[3]) for v in data_labels]:
                sct.printv('ERROR: labels mismatch', 1, 'warning')

        result = 0.0
        for value in data_labels:
            for v in ref_labels:
                if round(v[3]) == round(value[3]):
                    result = result + (value[2] - v[2]) * (value[2] - v[2])
                    break
        result = math.sqrt(result / len(X))
        sct.printv('MSE error in Z direction = ' + str(result) + ' mm')

        if result > threshold_mse:
            f = open(self.image_input.path + 'error_log_' + self.image_input.file_name + '.txt', 'w')
            f.write(
                'The labels error (MSE) between ' + self.image_input.file_name + ' and ' + self.image_ref.file_name + ' is: ' + str(
                    result))
            f.close()

        return result

    def create_label(self):
        """
        This function create an image with labels listed by the user.
        This method works only if the user inserted correct coordinates.

        self.coordinates is a list of coordinates (class in msct_types).
        a Coordinate contains x, y, z and value.
        """
        image_output = Image(im_ref_zero=self.image_input)

        # loop across labels
        for i,coordinate in enumerate(self.coordinates):
            # display info
            sct.printv('Label #' + str(i) + ': ' + str(coordinate.x) + ',' + str(coordinate.y) + ',' + str(coordinate.z) + ' --> ' + str(coordinate.value), 1)
            image_output.data[coordinate.x, coordinate.y, coordinate.z] = int(coordinate.value)

        return image_output

    def remove_label(self):
        """
        This function compares two label images and remove any labels in input image that are not in reference image.
        """
        image_output = Image(im_ref=self.image_input)
        X, Y, Z = (self.image_input.data > 0).nonzero()

        X_ref, Y_ref, Z_ref = (self.image_ref.data > 0).nonzero()

        nbLabel = len(X)
        nbLabel_ref = len(X_ref)
        for i in range(0, nbLabel):
            value = self.image_input.data[X[i]][Y[i]][Z[i]]
            isInRef = False
            for j in range(0, nbLabel_ref):
                value_ref = self.image_ref.data[X_ref[j]][Y_ref[j]][Z_ref[j]]
                # the following line could make issues when down sampling input, for example 21,00001 not = 21,0
                #if value_ref == value:
                if abs(value - value_ref) < 0.1:
                    image_output.data[X[i]][Y[i]][Z[i]] = int(round(value_ref))
                    isInRef = True
            if isInRef == False:
                image_output.data[X[i]][Y[i]][Z[i]] = 0

        return image_output

    def extract_disk_position(self, output_filename):
        """
        This function extracts the intervertebral disks position based on the vertebral level labeled image and an image of the centerline (input as the reference image)
        if output_filename is .txt, a text file is created with the location of each intervertebral disk. If the filename is a nifti file, the function create a file with label located at intervertebral disk positions.
        """
        X, Y, Z = (self.image_input.data > 0).nonzero()
        Xc, Yc, Zc = (self.image_ref.data > 0).nonzero() # position of the centerline

        image_output = Image(im_ref=self.image_ref)

        nbLabel = len(X)
        nbLabel_centerline = len(Xc)
        # sort Xc, Yc, and Zc depending on Yc
        cent = [Xc, Yc, Zc]
        indices = range(nbLabel_centerline)
        indices.sort(key=cent[1].__getitem__)
        for i, sublist in enumerate(cent):
            cent[i] = [sublist[j] for j in indices]
        Xc = []
        Yc = []
        Zc = []
        # remove double values
        for i in range(0, len(cent[1])):
            if Yc.count(cent[1][i]) == 0:
                Xc.append(cent[0][i])
                Yc.append(cent[1][i])
                Zc.append(cent[2][i])
        nbLabel_centerline = len(Xc)

        centerline_level = [0 for a in range(nbLabel_centerline)]
        for i in range(0, nbLabel_centerline):
            centerline_level[i] = self.image_input.data[Xc[i]][Yc[i]][Zc[i]]
            image_output.data[Xc[i]][Yc[i]][Zc[i]] = 0
        for i in range(0, nbLabel_centerline - 1):
            centerline_level[i] = abs(centerline_level[i + 1] - centerline_level[i])
        centerline_level[-1] = 0

        C = [i for i, e in enumerate(centerline_level) if e != 0]
        nb_disks = len(C)

        path, file_name, ext = sct.extract_fname(output_filename)
        if ext == '.txt':
            fo = open(output_filename, "wb")
            for i in range(0, nb_disks):
                line = (self.image_input.data[Xc[C[i]]][Yc[C[i]]][Zc[C[i]]], Xc[C[i]], Yc[C[i]], Zc[C[i]])
                fo.write("%i %i %i %i\n" % line)
            fo.close()
            return None
        else:
            for i in range(0, nb_disks):
                image_output.data[Xc[C[i]]][Yc[C[i]]][Zc[C[i]]] = self.image_input.data[Xc[C[i]]][Yc[C[i]]][Zc[C[i]]]
            return image_output

    def extract_centerline(self):
        """
        This function
        """
        # the Z image is assume to be in second dimension
        X, Y, Z = (self.image_input.data > 0).nonzero()
        cent = [X, Y, Z]
        indices = range(0, len(X))
        indices.sort(key=cent[1].__getitem__)
        for i, sublist in enumerate(cent):
            cent[i] = [sublist[j] for j in indices]
        X = [];
        Y = [];
        Z = []
        # remove double values
        for i in range(0, len(cent[1])):
            if Y.count(cent[1][i]) == 0:
                X.append(cent[0][i])
                Y.append(cent[1][i])
                Z.append(cent[2][i])

        fo = open(self.fname_output, "wb")
        for i in range(0, len(X)):
            line = (X[i], Y[i], Z[i])
            fo.write("%i %i %i\n" % line)
        fo.close()

    def extract_segmentation(self):
        """
        This function
        """
        # the Z image is assume to be in second dimension
        X, Y, Z = (self.image_input.data > 0).nonzero()
        cent = [X, Y, Z]
        indices = range(0, len(X))
        indices.sort(key=cent[1].__getitem__)
        for i, sublist in enumerate(cent):
            cent[i] = [sublist[j] for j in indices]
        X = [];
        Y = [];
        Z = []
        # remove double values
        for i in range(0, len(cent[1])):
            X.append(cent[0][i])
            Y.append(cent[1][i])
            Z.append(cent[2][i])

        fo = open(self.fname_output, "wb")
        for i in range(0, len(X)):
            line = (X[i], Y[i], Z[i])
            fo.write("%i %i %i\n" % line)
        fo.close()

    def display_voxel(self):
        """
        This function displays all the labels that are contained in the input image.
        """
        # the Z image is assume to be in second dimension
        X, Y, Z = (self.image_input.data > 0).nonzero()
        useful_notation = ''
        for k in range(0, len(X)):
            print 'Position=(' + str(X[k]) + ',' + str(Y[k]) + ',' + str(Z[k]) + ') -- Value= ' + str(
                self.image_input.data[X[k]][Y[k]][Z[k]])
            if useful_notation != '':
                useful_notation = useful_notation + ':'
            useful_notation = useful_notation + str(X[k]) + ',' + str(Y[k]) + ',' + str(Z[k]) + ',' + str(
                int(self.image_input.data[X[k]][Y[k]][Z[k]]))

        print 'Useful notation:'
        print useful_notation

    def diff(self):
        """
        This function detects any label mismatch between input image and reference image
        """
        X, Y, Z = (self.image_input.data > 0).nonzero()
        data_labels = [[X[i], Y[i], Z[i], self.image_input.data[X[i], Y[i], Z[i]]] for i in range(0, len(X))]

        X_ref, Y_ref, Z_ref = (self.image_ref.data > 0).nonzero()
        ref_labels = [[X_ref[i], Y_ref[i], Z_ref[i], self.image_ref.data[X_ref[i], Y_ref[i], Z_ref[i]]] for i in range(0, len(X_ref))]

        print "Label in input image that are not in ref image:"
        for i in range(0, len(data_labels)):
            isIn = False
            for j in range(0, len(ref_labels)):
                if data_labels[i][3] == ref_labels[j][3]:
                    isIn = True
            if not isIn:
                print data_labels[i][3]

        print "Label in ref image that are not in input image:"
        for i in range(0, len(ref_labels)):
            isIn = False
            for j in range(0, len(data_labels)):
                if ref_labels[i][3] == data_labels[j][3]:
                    isIn = True
            if not isIn:
                print ref_labels[i][3]

    def distance_interlabels(self, max_dist):
        """
        This function calculates the distances between each label in the input image.
        If a distance is larger than max_dist, a warning message is displayed.
        """
        X, Y, Z = (self.image_input.data > 0).nonzero()

        # for all points with non-zeros neighbors, force the neighbors to 0
        for i in range(0, len(X) - 1):
            dist = math.sqrt((X[i] - X[i + 1]) ** 2 + (Y[i] - Y[i + 1]) ** 2 + (Z[i] - Z[i + 1]) ** 2)
            if dist < max_dist:
                print 'Warning: the distance between label ' + str(i) + '[' + str(X[i]) + ',' + str(Y[i]) + ',' + str(
                    Z[i]) + ']=' + str(self.image_input.data[X[i]][Y[i]][Z[i]]) + ' and label ' + str(i + 1) + '[' + str(
                    X[i + 1]) + ',' + str(Y[i + 1]) + ',' + str(Z[i + 1]) + ']=' + str(
                    self.image_input.data[X[i + 1]][Y[i + 1]][Z[i + 1]]) + ' is larger than ' + str(max_dist) + '. Distance=' + str(dist)

#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print """
""" + os.path.basename(__file__) + """
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Utility function for labels.

USAGE
  """ + os.path.basename(__file__) + """ -i <data> -t <process>

MANDATORY ARGUMENTS
  -i <data>        labels or image to create labels on. Must be 3D.
  -o <output>      output volume.
  -t <process>     process:
                     cross: create a cross. Must use flag "-c"
                     remove: remove labels. Must use flag "-r".
                     display-voxel: display all labels in file
                     create: create labels. Must use flag "-x" to list labels.

OPTIONAL ARGUMENTS
  -x <x,y,z,v>     labels. Use ":" if you have multiple labels.
                     x: x-coordinates
                     y: y-coordinates
                     z: z-coordinates
                     v: value of label
  -r <volume>      reference volume for label removing.
  -c <radius>      cross radius in mm (default=5mm).
  -v {0,1}         verbose. Default=""" + str(param_default.verbose) + """
  -d               dilate.
  -h               help. Show this message

EXAMPLE
  """ + os.path.basename(__file__) + """ -i t2.nii.gz -c 5\n"""

    # exit program
    sys.exit(2)


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
                      description="""process:
                            cross: create a cross. Must use flag "-c"
                            remove: remove labels. Must use flag "-r"
                            display-voxel: display all labels in file
                            create: create labels. Must use flag "-x" to list labels
                            increment: increment labels from top to bottom (in z direction, suppose RPI orientation)
                            MSE: compute Mean Square Error between labels input and reference input "-r"
                            """,
                      mandatory=True,
                      example="cross")
    parser.add_option(name="-x",
                      type_value=[[':'], 'Coordinate'],
                      description="""labels x,y,z,v. Use ":" if you have multiple labels.
                            x: x-coordinates
                            y: y-coordinates
                            z: z-coordinates
                            v: value of label""",
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
                      type_value="bool",
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
    if "-o" in arguments:
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
    processor = ProcessLabels(input_filename, process_type, fname_output=input_fname_output, fname_ref=input_fname_ref, cross_radius=input_cross_radius, dilate=input_dilate, coordinates=input_coordinates, verbose=input_verbose)
    processor.process()
