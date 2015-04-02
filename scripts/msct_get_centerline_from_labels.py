#!/usr/bin/env python


import numpy as np
import commands, sys


# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
sys.path.append('/home/tamag/code')

from msct_image import Image
from msct_parser import Parser
import nibabel
import os
import time
import sct_utils as sct
from sct_orientation import get_orientation, set_orientation
from sct_process_segmentation import b_spline_centerline
from scipy import interpolate, ndimage



class ExtractCenterline :
    def __init__(self):
        self.list_image = []
        self.centerline = []
        self.dimension = [0, 0, 0, 0, 0, 0, 0, 0]

    def addfiles(self, file):

        image_input = Image(file)

        #check that files are same size
        if len(self.list_image) > 0 :
            self.dimension = sct.get_dimension(self.list_image[0])
            nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(image_input)
            if self.dimension != [nx, ny, nz, nt, px, py, pz, pt] :
                # Return error and exit programm if not same size
                print('\nError: Files are not of the same size.')
                sys.exit()
        # Add file if same size
        self.list_image = self.list_image.extend(image_input)

    def compute(self):
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(self.list_image[0])

        # Define output image (size matter)
        image_concatenation = self.list_image[0].copy()
        image_concatenation *= 0
        image_output = self.list_image[0].copy()
        image_output *= 0
        # Concatenate all files by addition
        for i in range(0, len(self.list_image)):
            for s in range(0, nz) :
                image_concatenation[:,:,s] = image_concatenation[:,:,s] + self.list_image[i][:,:,s] * (1/len(self.list_image))


        # get center of mass of the centerline/segmentation
        sct.printv('\nGet center of mass of the concatenate file...')
        z_centerline = [iz for iz in range(0, nz, 1) if image_concatenation[:, :, iz].any()]

        nz_nonz = len(z_centerline)
        x_centerline = [0 for iz in range(0, nz_nonz, 1)]
        y_centerline = [0 for iz in range(0, nz_nonz, 1)]


        # Calculate centerline coordinates and create image of the centerline
        for iz in range(0, nz_nonz, 1):
            x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(image_concatenation[:, :, z_centerline[iz]])

        x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)

        for iz in range(0, nz_nonz, 1):
            image_output[x_centerline_fit[iz], y_centerline_fit[iz], z_centerline[iz]] = 1



        return image_output

    def getCenterline(self, type=''):
        # Compute the centerline and save it into a image file of type "type"

        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(self.list_image[0])

        # Define output image (size matter)
        image_concatenation = self.list_image[0].copy()
        image_concatenation *= 0
        image_output = self.list_image[0].copy()
        image_output *= 0
        # Concatenate all files by addition
        for i in range(0, len(self.list_image)):
            for s in range(0, nz) :
                image_concatenation[:,:,s] = image_concatenation[:,:,s] + self.list_image[i][:,:,s] * (1/len(self.list_image))


        # get center of mass of the centerline/segmentation
        sct.printv('\nGet center of mass of the concatenate file...')
        z_centerline = [iz for iz in range(0, nz, 1) if image_concatenation[:, :, iz].any()]
        nz_nonz = len(z_centerline)
        x_centerline = [0 for iz in range(0, nz_nonz, 1)]
        y_centerline = [0 for iz in range(0, nz_nonz, 1)]


        # Calculate centerline coordinates and create image of the centerline
        for iz in range(0, nz_nonz, 1):
            x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(image_concatenation[:, :, z_centerline[iz]])

        x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)

        for iz in range(0, nz_nonz, 1):
            image_output[round(x_centerline_fit[iz]), round(y_centerline_fit[iz]), z_centerline[iz]] = 1

        image_output.save(type)

        #return file  


    def writeCenterline(self, output_file_name=None):
        # Compute the centerline and write the float coordinates into a txt file

        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(self.list_image[0])

        # Define output image (size matter)
        image_concatenation = self.list_image[0].copy()
        image_concatenation *= 0
        image_output = self.list_image[0].copy()
        image_output *= 0
        # Concatenate all files by addition
        for i in range(0, len(self.list_image)):
            for s in range(0, nz) :
                image_concatenation[:,:,s] = image_concatenation[:,:,s] + self.list_image[i][:,:,s] * (1/len(self.list_image))


        # get center of mass of the centerline/segmentation
        sct.printv('\nGet center of mass of the concatenate file...')
        z_centerline = [iz for iz in range(0, nz, 1) if image_concatenation[:, :, iz].any()]
        nz_nonz = len(z_centerline)
        x_centerline = [0 for iz in range(0, nz_nonz, 1)]
        y_centerline = [0 for iz in range(0, nz_nonz, 1)]


        # Calculate centerline coordinates and create image of the centerline
        for iz in range(0, nz_nonz, 1):
            x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(image_concatenation[:, :, z_centerline[iz]])

        x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)



        # Create output text file
        if output_file_name != None :
            file_name = output_file_name
        else: file_name = 'generated'+'_centerline'+'.txt'

        sct.printv('\nWrite text file...')
        #file_results = open("../"+file_name, 'w')
        file_results = open(file_name, 'w')
        for i in range(0, nz_nonz, 1):
            file_results.write(str(int(i)) + ' ' + str(x_centerline_fit[i]) + ' ' + str(y_centerline_fit[i]) + '\n')
        file_results.close()

        return file_name



# =======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":


    parser = Parser(__file__)
    parser.usage.set_description('Class to process centerline extraction from.')
    parser.add_option()
    arguments = parser.parse(sys.argv[1:])

    image = Image(arguments["-i"])
    image.changeType('minimize')
