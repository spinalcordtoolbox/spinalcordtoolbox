#!/usr/bin/env python


import numpy as np
import commands, sys


# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, 'scripts'))
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
from msct_nurbs import NURBS



class ExtractCenterline :
    def __init__(self):
        self.list_image = []
        self.list_file = []
        self.centerline = []
        self.dimension = [0, 0, 0, 0, 0, 0, 0, 0]

    def addfiles(self, file):

        path_data, file_data, ext_data = sct.extract_fname(file)
        #check that files are same size
        if len(self.list_file) > 0 :
            self.dimension = sct.get_dimension(self.list_file[0])
            nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(file)

            #if self.dimension != (nx, ny, nz, nt, px, py, pz, pt) :
            if self.dimension[0:3] != (nx, ny, nz) or self.dimension[4:7] != (px, py, pz) :
                # Return error and exit programm if not same size
                print('\nError: Files are not of the same size.')
                sys.exit()
        # Add file if same size
        self.list_file.append(file)

        image_input = Image(file)
        self.list_image.append(image_input)
        print('\nFile', file_data+ext_data,' added to the list.')

    def compute(self):
        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(self.list_file[0])

        # Define output image (size matter)
        image_concatenation = self.list_image[0].copy()
        image_concatenation.data *= 0
        image_output = self.list_image[0].copy()
        image_output.data *= 0
        # Concatenate all files by addition
        for i in range(0, len(self.list_image)):
            for s in range(0, nz) :
                image_concatenation.data[:,:,s] = image_concatenation.data[:,:,s] + self.list_image[i].data[:,:,s] #* (1/len(self.list_image))


        # get center of mass of the centerline/segmentation
        sct.printv('\nGet center of mass of the concatenate file...')
        z_centerline = [iz for iz in range(0, nz, 1) if image_concatenation.data[:, :, iz].any()]

        nz_nonz = len(z_centerline)
        x_centerline = [0 for iz in range(0, nz_nonz, 1)]
        y_centerline = [0 for iz in range(0, nz_nonz, 1)]


        # Calculate centerline coordinates and create image of the centerline
        for iz in range(0, nz_nonz, 1):
            x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(image_concatenation.data[:, :, z_centerline[iz]])

        points = [[x_centerline[n],y_centerline[n], z_centerline[n]] for n in range(len(z_centerline))]
        nurbs = NURBS(3, 1000, points)
        P = nurbs.getCourbe3D()
        x_centerline_fit = P[0]
        y_centerline_fit = P[1]
        z_centerline_fit = P[2]

        #x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)

        for iz in range(0, z_centerline_fit.shape[0], 1):
            image_output.data[x_centerline_fit[iz], y_centerline_fit[iz], z_centerline_fit[iz]] = 1


        return image_output

    def getCenterline(self, type='', output_file_name=None, verbose=0):
        # Compute the centerline and save it into a image file of type "type"

        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(self.list_file[0])

        # Define output image (size matter)
        image_concatenation = self.list_image[0].copy()
        image_concatenation.data *= 0
        image_output = self.list_image[0].copy()
        image_output.data *= 0
        # Concatenate all files by addition
        for i in range(0, len(self.list_image)):
            for s in range(0, nz) :
                image_concatenation.data[:,:,s] = image_concatenation.data[:,:,s] + self.list_image[i].data[:,:,s] #* (1/len(self.list_image))
        print image_concatenation.data[:,:,414]

        # get center of mass of the centerline/segmentation
        sct.printv('\nGet center of mass of the concatenate file...')
        z_centerline = [iz for iz in range(0, nz, 1) if image_concatenation.data[:, :, iz].any()]
        nz_nonz = len(z_centerline)
        x_centerline = [0 for iz in range(0, nz_nonz, 1)]
        y_centerline = [0 for iz in range(0, nz_nonz, 1)]


        # Calculate centerline coordinates and create image of the centerline
        for iz in range(0, nz_nonz, 1):
            x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(image_concatenation.data[:, :, z_centerline[iz]])
        #x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)


        points = [[x_centerline[n], y_centerline[n], z_centerline[n]] for n in range(nz_nonz)]
        nurbs = NURBS(3, 1000, points, nbControl=None)
        P = nurbs.getCourbe3D()
        x_centerline_fit = P[0]
        y_centerline_fit = P[1]
        z_centerline_fit = P[2]

        if verbose==1 :
                import matplotlib.pyplot as plt

                #Creation of a vector x that takes into account the distance between the labels
                x_display = [0 for i in range(x_centerline_fit.shape[0])]
                y_display = [0 for i in range(y_centerline_fit.shape[0])]
                for i in range(0, nz_nonz, 1):
                    x_display[z_centerline[i]-z_centerline[0]] = x_centerline[i]
                    y_display[z_centerline[i]-z_centerline[0]] = y_centerline[i]

                plt.figure(1)
                plt.subplot(2,1,1)
                #plt.plot(z_centerline,x_centerline, 'ro')
                plt.plot(z_centerline_fit, x_display, 'ro')
                plt.plot(z_centerline_fit, x_centerline_fit)
                plt.xlabel("Z")
                plt.ylabel("X")
                plt.title("x and x_fit coordinates")

                plt.subplot(2,1,2)
                #plt.plot(z_centerline,y_centerline, 'ro')
                plt.plot(z_centerline_fit, y_display, 'ro')
                plt.plot(z_centerline_fit, y_centerline_fit)
                plt.xlabel("Z")
                plt.ylabel("Y")
                plt.title("y and y_fit coordinates")
                plt.show()


        for iz in range(0, z_centerline_fit.shape[0], 1):
            image_output.data[int(round(x_centerline_fit[iz])), int(round(y_centerline_fit[iz])), z_centerline_fit[iz]] = 1

        #image_output.save(type)
        file_load = nibabel.load(self.list_file[0])
        data = file_load.get_data()
        hdr = file_load.get_header()

        print '\nWrite NIFTI volumes...'
        img = nibabel.Nifti1Image(image_output.data, None, hdr)
        if output_file_name != None :
            file_name = output_file_name
        else: file_name = 'generated_centerline.nii.gz'
        nibabel.save(img,file_name)


        # to view results
        print '\nDone !'
        print '\nTo view results, type:'
        print 'fslview '+file_name+' &\n'


    def writeCenterline(self, output_file_name=None):
        # Compute the centerline and write the float coordinates into a txt file

        nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(self.list_file[0])

        # Define output image (size matter)
        image_concatenation = self.list_image[0].copy()
        image_concatenation.data *= 0
        image_output = self.list_image[0].copy()
        image_output.data *= 0
        # Concatenate all files by addition
        for i in range(0, len(self.list_image)):
            for s in range(0, nz) :
                image_concatenation.data[:,:,s] = image_concatenation.data[:,:,s] + self.list_image[i].data[:,:,s] #* (1/len(self.list_image))


        # get center of mass of the centerline/segmentation
        sct.printv('\nGet center of mass of the concatenate file...')
        z_centerline = [iz for iz in range(0, nz, 1) if image_concatenation.data[:, :, iz].any()]
        nz_nonz = len(z_centerline)
        x_centerline = [0 for iz in range(0, nz_nonz, 1)]
        y_centerline = [0 for iz in range(0, nz_nonz, 1)]


        # Calculate centerline coordinates and create image of the centerline
        for iz in range(0, nz_nonz, 1):
            x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(image_concatenation.data[:, :, z_centerline[iz]])

        #x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)

        points = [[x_centerline[n], y_centerline[n], z_centerline[n]] for n in range(nz_nonz)]
        nurbs = NURBS(3, 1000, points)
        P = nurbs.getCourbe3D()
        x_centerline_fit = P[0]
        y_centerline_fit = P[1]
        z_centerline_fit = P[2]

        # Create output text file
        if output_file_name != None :
            file_name = output_file_name
        else: file_name = 'generated_centerline.txt'

        sct.printv('\nWrite text file...')
        #file_results = open("../"+file_name, 'w')
        file_results = open(file_name, 'w')
        for i in range(0, z_centerline_fit.shape[0], 1):
            file_results.write(str(int(z_centerline_fit[i])) + ' ' + str(x_centerline_fit[i]) + ' ' + str(y_centerline_fit[i]) + '\n')
        file_results.close()

        #return file_name



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
