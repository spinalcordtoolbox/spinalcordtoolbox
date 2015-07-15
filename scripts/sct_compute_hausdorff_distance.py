#!/usr/bin/env python
#
# Thinning with the Zhang-Suen algorithm (1984) --> code taken from  https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm
# Computation of the distances between two skeleton
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Sara Dupont
# CREATED: 2015-07-15
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sys
import time
import os
import numpy as np
import sct_utils as sct
from msct_image import Image
from msct_parser import Parser
import msct_gmseg_utils as sct_gm


# TODO: check that images are 2D AND adapt for 3D
# TODO: display results ==> not only max : with a violin plot of h1 and h2 distribution ?
# TODO: add the option Hyberbolic Hausdorff's distance : see  choi and seidel paper


class Param:
    def __init__(self):
        self.debug = 0
        self.thinning = True
        self.verbose = 1


class Thinning:
    def __init__(self, im):
        self.image = im
        self.image.data = (self.image.data > 0).astype(int)
        self.thinned_image = Image(param=self.zhang_suen(self.image.data), absolutepath=self.image.path + self.image.file_name + '_thinned' + self.image.ext, hdr=self.image.hdr)

    def get_neighbours(self, x, y, image):
        """
        Return 8-neighbours of image point P1(x,y), in a clockwise order
        code from https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm
        :param x:
        :param y:
        :param image:
        :return:
        """
        img = image
        x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
        return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]    # P6,P7,P8,P9

    def transitions(self, neighbours):
        """
        No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence
        code from https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm
        :param neighbours:
        :return:
        """
        n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
        return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

    def zhang_suen(self, image):
        """
        the Zhang-Suen Thinning Algorithm
        code from https://github.com/linbojin/Skeletonization-by-Zhang-Suen-Thinning-Algorithm
        :param image:
        :return:
        """
        image_thinned = image.copy()  # deepcopy to protect the original image
        changing1 = changing2 = 1  # the points to be removed (set as 0)
        while changing1 or changing2:  # iterates until no further changes occur in the image
            # Step 1
            changing1 = []
            rows, columns = image_thinned.shape  # x for rows, y for columns
            for x in range(1, rows - 1):         # No. of  rows
                for y in range(1, columns - 1):  # No. of columns
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = self.get_neighbours(x, y, image_thinned)
                    if (image_thinned[x][y] == 1 and    # Condition 0: Point P1 in the object regions
                        2 <= sum(n) <= 6 and    # Condition 1: 2<= N(P1) <= 6
                        self.transitions(n) == 1 and    # Condition 2: S(P1)=1
                        P2 * P4 * P6 == 0 and    # Condition 3
                        P4 * P6 * P8 == 0):         # Condition 4
                        changing1.append((x, y))
            for x, y in changing1:
                image_thinned[x][y] = 0
            # Step 2
            changing2 = []
            for x in range(1, rows - 1):
                for y in range(1, columns - 1):
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = self.get_neighbours(x, y, image_thinned)
                    if (image_thinned[x][y] == 1 and        # Condition 0
                        2 <= sum(n) <= 6 and       # Condition 1
                        self.transitions(n) == 1 and      # Condition 2
                        P2 * P4 * P8 == 0 and       # Condition 3
                        P2 * P6 * P8 == 0):            # Condition 4
                        changing2.append((x, y))
            for x, y in changing2:
                image_thinned[x][y] = 0
        return image_thinned


class HausdorffDistance:
    def __init__(self, data1, data2):
        """
        the hausdorff distance between two sets is the maximum of the distances from a point in any of the sets to the nearest point in the other set
        :return:
        """
        self.data1 = data1
        self.data2 = data2

        self.min_distances_1 = self.relative_hausdorff_dist(self.data1, self.data2)
        self.min_distances_2 = self.relative_hausdorff_dist(self.data2, self.data1)

        # relatives hausdorff's distances in pixel
        self.h1 = np.max(self.min_distances_1)
        self.h2 = np.max(self.min_distances_2)

        # Hausdorff's distance in pixel
        self.H = max(self.h1, self.h2)

    def relative_hausdorff_dist(self, dat1, dat2):
        h = np.zeros(dat1.shape)
        for x1, y1 in zip(range(dat1.shape[0]), range(dat1.shape[1])):
            if dat1[x1, y1] == 1:
                d_p1_dat2 = []
                p1 = np.asarray([x1, y1])
                for x2, y2 in zip(range(dat2.shape[0]), range(dat2.shape[1])):
                    if dat2[x2, y2] == 1:
                        p2 = np.asarray([x2, y2])
                        d_p1_dat2.append(np.linalg.norm(p1-p2))  # Euclidean distance between p1 and p2
                h[x1, y1] = min(d_p1_dat2)
        return h


class ComputeDistances:
    def __init__(self, im1, im2):
        self.im1 = im1
        self.im2 = im2

        nx1, ny1, nz1, nt1, px1, py1, pz1, pt1 = sct.get_dimension(self.im1.absolutepath)
        nx2, ny2, nz2, nt2, px2, py2, pz2, pt2 = sct.get_dimension(self.im2.absolutepath)

        assert px1 == px2 and py1 == py2 and px1 == py1

        if param.thinning:
            self.thinning1 = Thinning(im1)
            self.thinning2 = Thinning(im2)
            self.thinning1.thinned_image.save()
            self.thinning2.thinned_image.save()

            dat1 = self.thinning1.thinned_image.data
            dat2 = self.thinning2.thinned_image.data
        else:
            dat1 = (self.im1.data > 0).astype(int)
            dat2 = (self.im2.data > 0).astype(int)

        self.distances = HausdorffDistance(dat1, dat2)

        res = '-----------------------------------------------------------------------------\n' \
              'Hausdorff\'s distance : ' + str(self.distances.H*px1) + ' mm\n\n' \
              'First relative Hausdorff\'s distance : ' + str(self.distances.h1*px1) + ' mm\n' \
              'Second Hausdorff\'s distance : ' + str(self.distances.h2*px1) + ' mm'

        sct.printv(res, param.verbose, 'normal')

        # self.distances.h2*px1, self.distances.H*px1



########################################################################################################################
# ------------------------------------------------------  MAIN ------------------------------------------------------- #
########################################################################################################################

if __name__ == "__main__":
    param = Param()
    input_fname = None
    if param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
    else:
        param_default = Param()

        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Compute the Hausdorff\'s distance between two binary images which can be thinned (ie skeletonized)'
                                     'If only one image is inputted, it will be only thinned')
        parser.add_option(name="-i",
                          type_value="file",
                          description="First Image on which you want to find the skeleton",
                          mandatory=True,
                          example='t2star_manual_gmseg.nii.gz')
        parser.add_option(name="-r",
                          type_value="file",
                          description="Second Image on which you want to find the skeleton",
                          mandatory=False,
                          default_value=None,
                          example='t2star_manual_gmseg.nii.gz')
        parser.add_option(name="-t",
                          type_value="multiple_choice",
                          description="Thinning : find the skeleton of the binary images using the Zhang-Suen algorithm (1984)",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1'])
        parser.add_option(name="-v",
                          type_value="int",
                          description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                          mandatory=False,
                          default_value=0,
                          example='1')

        arguments = parser.parse(sys.argv[1:])
        input_fname = arguments["-i"]
        input_second_fname = None

        if "-r" in arguments:
            input_second_fname = arguments["-r"]
        if "-v" in arguments:
            param.verbose = arguments["-v"]
        if "-t" in arguments:
            param.thinning = bool(int(arguments["-t"]))

        tmp_dir = 'tmp_' + time.strftime("%y%m%d%H%M%S")
        sct.run('mkdir ' + tmp_dir)
        im1_name = "im1.nii.gz"
        sct.run('cp ' + input_fname + ' ' + tmp_dir + '/' + im1_name)
        if input_second_fname is not None:
            im2_name = 'im2.nii.gz'
            sct.run('cp ' + input_second_fname + ' ' + tmp_dir + '/' + im2_name)
        else:
            im2_name = ''

        os.chdir(tmp_dir)
        input_im1 = Image(sct_gm.resample_image(im1_name, binary=True, npx=0.3, npy=0.3))
        if input_second_fname is not None:
            input_im2 = Image(sct_gm.resample_image(im2_name, binary=True, npx=0.3, npy=0.3))
        else:
            input_im2 = None
        if input_second_fname is not None:
            computation = ComputeDistances(input_im1, input_im2)
        else:
            thinning = Thinning(input_im1)
            thinning.thinned_image.save()
            sct.run('cp ' + thinning.thinned_image.file_name + thinning.thinned_image.ext + ' ../' + sct.extract_fname(input_fname)[1] + '_thinned' + sct.extract_fname(input_fname)[2])

        os.chdir('..')
