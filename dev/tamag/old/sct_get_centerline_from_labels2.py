#!/usr/bin/env python


import sys, io, shutil, os, time

import numpy as np

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))
sys.path.append('/home/tamag/code')

from msct_image import Image
from msct_parser import Parser
import sct_utils as sct
from sct_orientation import get_orientation, set_orientation
from sct_process_segmentation import b_spline_centerline
from scipy import interpolate



def main(segmentation_file=None, label_file=None, output_file_name=None, parameter = "binary_centerline", remove_temp_files = 1, verbose = 0 ):

#Process for a binary file as output:
    if parameter == "binary_centerline":

        # Binary_centerline: Process for only a segmentation file:
        if "-i" in arguments and "-l" not in arguments:
                    # Extract path, file and extension
            segmentation_file = os.path.abspath(segmentation_file)
            path_data, file_data, ext_data = sct.extract_fname(segmentation_file)

            path_tmp = sct.tmp_create(basename="centerline_from_labels", verbose=verbose)

            # copy files into tmp folder
            sct.run('cp '+segmentation_file+' '+path_tmp)

            # go to tmp folder
            curdir = os.getcwd()
            os.chdir(path_tmp)

            # Change orientation of the input segmentation into RPI
            print '\nOrient segmentation image to RPI orientation...'
            fname_segmentation_orient = 'tmp.segmentation_rpi' + ext_data
            set_orientation(file_data+ext_data, 'RPI', fname_segmentation_orient)

            # Extract orientation of the input segmentation
            orientation = get_orientation(file_data+ext_data)
            print '\nOrientation of segmentation image: ' + orientation

            # Get size of data
            print '\nGet dimensions data...'
            nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_segmentation_orient)
            print '.. '+str(nx)+' x '+str(ny)+' y '+str(nz)+' z '+str(nt)

            print '\nOpen segmentation volume...'
            file = nibabel.load(fname_segmentation_orient)
            data = file.get_data()
            hdr = file.get_header()

            # Extract min and max index in Z direction
            X, Y, Z = (data>0).nonzero()
            min_z_index, max_z_index = min(Z), max(Z)
            x_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
            y_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
            z_centerline = [iz for iz in range(min_z_index, max_z_index+1)]
            # Extract segmentation points and average per slice
            for iz in range(min_z_index, max_z_index+1):
                x_seg, y_seg = (data[:,:,iz]>0).nonzero()
                x_centerline[iz-min_z_index] = np.mean(x_seg)
                y_centerline[iz-min_z_index] = np.mean(y_seg)

            #ne sert a rien
            for k in range(len(X)):
                data[X[k],Y[k],Z[k]] = 0

            print len(x_centerline)
            # Fit the centerline points with splines and return the new fitted coordinates
                    #done with nurbs for now
            x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)
                        # Create an image with the centerline
            for iz in range(min_z_index, max_z_index+1):
                data[round(x_centerline_fit[iz-min_z_index]), round(y_centerline_fit[iz-min_z_index]), iz] = 1    #with nurbs fitting
                #data[round(x_centerline[iz-min_z_index]), round(y_centerline[iz-min_z_index]), iz] = 1             #without nurbs fitting


            # Write the centerline image in RPI orientation
            hdr.set_data_dtype('uint8') # set imagetype to uint8
            print '\nWrite NIFTI volumes...'
            img = nibabel.Nifti1Image(data, None, hdr)
            if output_file_name != None :
                file_name = output_file_name
            else: file_name = file_data+'_centerline'+ext_data
            nibabel.save(img,'tmp.centerline.nii')
            sct.generate_output_file('tmp.centerline.nii',file_name)

            del data

            # come back
            os.chdir(curdir)

            # Change orientation of the output centerline into input orientation
            print '\nOrient centerline image to input orientation: ' + orientation
            set_orientation(path_tmp+'/'+file_name, orientation, file_name)

           # Remove temporary files
            if remove_temp_files:
                print('\nRemove temporary files...')
                sct.run('rm -rf '+path_tmp)

            return file_name


        # Binary_centerline: Process for only a label file:
        if "-l" in arguments and "-i" not in arguments:
            file = os.path.abspath(label_file)
            path_data, file_data, ext_data = sct.extract_fname(file)

            file = nibabel.load(label_file)
            data = file.get_data()
            hdr = file.get_header()

            X,Y,Z = (data>0).nonzero()
            Z_new = np.linspace(min(Z),max(Z),(max(Z)-min(Z)+1))

            # sort X and Y arrays using Z
            X = [X[i] for i in Z[:].argsort()]
            Y = [Y[i] for i in Z[:].argsort()]
            Z = [Z[i] for i in Z[:].argsort()]

            #print X, Y, Z

            f1 = interpolate.UnivariateSpline(Z, X)
            f2 = interpolate.UnivariateSpline(Z, Y)

            X_fit = f1(Z_new)
            Y_fit = f2(Z_new)

            #print X_fit
            #print Y_fit

            if verbose==1 :
                import matplotlib.pyplot as plt

                plt.figure()
                plt.plot(Z_new,X_fit)
                plt.plot(Z,X,'o',linestyle = 'None')
                plt.show()

                plt.figure()
                plt.plot(Z_new,Y_fit)
                plt.plot(Z,Y,'o',linestyle = 'None')
                plt.show()

            data =data*0

            for i in xrange(len(X_fit)):
                data[X_fit[i],Y_fit[i],Z_new[i]] = 1


            # Create NIFTI image
            print '\nSave volume ...'
            hdr.set_data_dtype('float32') # set image type to uint8
            img = nibabel.Nifti1Image(data, None, hdr)
            if output_file_name != None :
                file_name = output_file_name
            else: file_name = file_data+'_centerline'+ext_data
            # save volume
            nibabel.save(img,file_name)
            print '\nFile created : ' + file_name

            del data



        #### Binary_centerline: Process for a segmentation file and a label file:
        if "-l" and "-i" in arguments:

            ## Creation of a temporary file that will contain each centerline file of the process
            path_tmp = sct.tmp_create(verbose=verbose)

            ##From label file create centerline image
            print '\nPROCESS PART 1: From label file create centerline image.'
            file_label = os.path.abspath(label_file)
            path_data_label, file_data_label, ext_data_label = sct.extract_fname(file_label)

            file_label = nibabel.load(label_file)

            #Copy label_file into temporary folder
            sct.run('cp '+label_file+' '+path_tmp)

            data_label = file_label.get_data()
            hdr_label = file_label.get_header()

            if verbose == 1:
                from copy import copy
                data_label_to_show = copy(data_label)

            X,Y,Z = (data_label>0).nonzero()
            Z_new = np.linspace(min(Z),max(Z),(max(Z)-min(Z)+1))

            # sort X and Y arrays using Z
            X = [X[i] for i in Z[:].argsort()]
            Y = [Y[i] for i in Z[:].argsort()]
            Z = [Z[i] for i in Z[:].argsort()]

            #print X, Y, Z

            f1 = interpolate.UnivariateSpline(Z, X)
            f2 = interpolate.UnivariateSpline(Z, Y)

            X_fit = f1(Z_new)
            Y_fit = f2(Z_new)

            #print X_fit
            #print Y_fit

            if verbose==1 :
                import matplotlib.pyplot as plt

                plt.figure()
                plt.plot(Z_new,X_fit)
                plt.plot(Z,X,'o',linestyle = 'None')
                plt.show()

                plt.figure()
                plt.plot(Z_new,Y_fit)
                plt.plot(Z,Y,'o',linestyle = 'None')
                plt.show()

            data_label =data_label*0

            for i in xrange(len(X_fit)):
                data_label[X_fit[i],Y_fit[i],Z_new[i]] = 1

            # Create NIFTI image
            print '\nSave volume ...'
            hdr_label.set_data_dtype('float32') # set image type to uint8
            img = nibabel.Nifti1Image(data_label, None, hdr_label)
            # save volume
            file_name_label = file_data_label + '_centerline' + ext_data_label
            nibabel.save(img, file_name_label)
            print '\nFile created : ' + file_name_label

            # copy files into tmp folder
            sct.run('cp '+file_name_label+' '+path_tmp)
            #effacer fichier dans folder parent
            os.remove(file_name_label)
            del data_label


            ##From segmentation file create centerline image
            print '\nPROCESS PART 2: From segmentation file create centerline image.'
            # Extract path, file and extension
            segmentation_file = os.path.abspath(segmentation_file)
            path_data_seg, file_data_seg, ext_data_seg = sct.extract_fname(segmentation_file)

            # copy files into tmp folder
            sct.run('cp '+segmentation_file+' '+path_tmp)

            # go to tmp folder
            curdir = os.getcwd()
            os.chdir(path_tmp)

            # Change orientation of the input segmentation into RPI
            print '\nOrient segmentation image to RPI orientation...'
            fname_segmentation_orient = 'tmp.segmentation_rpi' + ext_data_seg
            set_orientation(file_data_seg+ext_data_seg, 'RPI', fname_segmentation_orient)

            # Extract orientation of the input segmentation
            orientation = get_orientation(file_data_seg+ext_data_seg)
            print '\nOrientation of segmentation image: ' + orientation

            # Get size of data
            print '\nGet dimensions data...'
            nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_segmentation_orient)
            print '.. '+str(nx)+' x '+str(ny)+' y '+str(nz)+' z '+str(nt)

            print '\nOpen segmentation volume...'
            file_seg = nibabel.load(fname_segmentation_orient)
            data_seg = file_seg.get_data()
            hdr_seg = file_seg.get_header()

            if verbose == 1:
                data_seg_to_show = copy(data_seg)

            # Extract min and max index in Z direction
            X, Y, Z = (data_seg>0).nonzero()
            min_z_index, max_z_index = min(Z), max(Z)
            x_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
            y_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
            z_centerline = [iz for iz in range(min_z_index, max_z_index+1)]
            # Extract segmentation points and average per slice
            for iz in range(min_z_index, max_z_index+1):
                x_seg, y_seg = (data_seg[:,:,iz]>0).nonzero()
                x_centerline[iz-min_z_index] = np.mean(x_seg)
                y_centerline[iz-min_z_index] = np.mean(y_seg)
            for k in range(len(X)):
                data_seg[X[k],Y[k],Z[k]] = 0
            # Fit the centerline points with splines and return the new fitted coordinates
                    #done with nurbs for now
            x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)


            # Create an image with the centerline
            for iz in range(min_z_index, max_z_index+1):
                data_seg[round(x_centerline_fit[iz-min_z_index]), round(y_centerline_fit[iz-min_z_index]), iz] = 1
            # Write the centerline image in RPI orientation
            hdr_seg.set_data_dtype('uint8') # set imagetype to uint8
            print '\nWrite NIFTI volumes...'
            img = nibabel.Nifti1Image(data_seg, None, hdr_seg)
            nibabel.save(img,'tmp.centerline.nii')
            file_name_seg = file_data_seg+'_centerline'+ext_data_seg
            sct.generate_output_file('tmp.centerline.nii',file_name_seg)   #pb ici

            # copy files into parent folder
            #sct.run('cp '+file_name_seg+' ../')

            del data_seg

            # Change orientation of the output centerline into input orientation
            print '\nOrient centerline image to input orientation: ' + orientation
            set_orientation(file_name_seg, orientation, file_name_seg)



            print '\nRemoving overlap of the centerline obtain with label file if there are any:'

            ## Remove overlap from centerline file obtain with label file
            remove_overlap(file_name_label, file_name_seg, "generated_centerline_without_overlap.nii.gz")


            ## Concatenation of the two centerline files
            print '\nConcatenation of the two centerline files:'
            if output_file_name != None :
                file_name = output_file_name
            else: file_name = 'centerline_total_from_label_and_seg'

            sct.run('fslmaths generated_centerline_without_overlap.nii.gz -add ' + file_name_seg + ' ' + file_name)



            if verbose == 1 :
                import matplotlib.pyplot as plt
                from scipy import ndimage

                #Get back concatenation of segmentation and labels before any processing
                data_concatenate = data_seg_to_show + data_label_to_show
                z_centerline = [iz for iz in range(0, nz, 1) if data_concatenate[:, :, iz].any()]
                nz_nonz = len(z_centerline)
                x_centerline = [0 for iz in range(0, nz_nonz, 1)]
                y_centerline = [0 for iz in range(0, nz_nonz, 1)]


                # Calculate centerline coordinates and create image of the centerline
                for iz in range(0, nz_nonz, 1):
                    x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(data_concatenate[:, :, z_centerline[iz]])

                #Load file with resulting centerline
                file_centerline_fit = nibabel.load(file_name)
                data_centerline_fit = file_centerline_fit.get_data()

                z_centerline_fit = [iz for iz in range(0, nz, 1) if data_centerline_fit[:, :, iz].any()]
                nz_nonz_fit = len(z_centerline_fit)
                x_centerline_fit_total = [0 for iz in range(0, nz_nonz_fit, 1)]
                y_centerline_fit_total = [0 for iz in range(0, nz_nonz_fit, 1)]

                #Convert to array
                x_centerline_fit_total = np.asarray(x_centerline_fit_total)
                y_centerline_fit_total = np.asarray(y_centerline_fit_total)
                #Calculate overlap between seg and label
                length_overlap = X_fit.shape[0] + x_centerline_fit.shape[0] - x_centerline_fit_total.shape[0]
                # The total fitting is the concatenation of the two fitting (
                for i in range(x_centerline_fit.shape[0]):
                    x_centerline_fit_total[i] = x_centerline_fit[i]
                    y_centerline_fit_total[i] = y_centerline_fit[i]
                for i in range(X_fit.shape[0]-length_overlap):
                    x_centerline_fit_total[x_centerline_fit.shape[0] + i] = X_fit[i+length_overlap]
                    y_centerline_fit_total[x_centerline_fit.shape[0] + i] = Y_fit[i+length_overlap]
                    print x_centerline_fit.shape[0] + i

                #for iz in range(0, nz_nonz_fit, 1):
                #    x_centerline_fit[iz], y_centerline_fit[iz] = ndimage.measurements.center_of_mass(data_centerline_fit[:, :, z_centerline_fit[iz]])

                #Creation of a vector x that takes into account the distance between the labels
                #x_centerline_fit = np.asarray(x_centerline_fit)
                #y_centerline_fit = np.asarray(y_centerline_fit)
                x_display = [0 for i in range(x_centerline_fit_total.shape[0])]
                y_display = [0 for i in range(y_centerline_fit_total.shape[0])]


                for i in range(0, nz_nonz, 1):
                    x_display[z_centerline[i]-z_centerline[0]] = x_centerline[i]
                    y_display[z_centerline[i]-z_centerline[0]] = y_centerline[i]

                plt.figure(1)
                plt.subplot(2,1,1)
                plt.plot(z_centerline_fit, x_display, 'ro')
                plt.plot(z_centerline_fit, x_centerline_fit_total)
                plt.xlabel("Z")
                plt.ylabel("X")
                plt.title("x and x_fit coordinates")

                plt.subplot(2,1,2)
                plt.plot(z_centerline_fit, y_display, 'ro')
                plt.plot(z_centerline_fit, y_centerline_fit_total)
                plt.xlabel("Z")
                plt.ylabel("Y")
                plt.title("y and y_fit coordinates")
                plt.show()

                del data_concatenate, data_label_to_show, data_seg_to_show, data_centerline_fit

            # Retrieve result
            sct.copy(file_name, curdir)

            # come back
            os.chdir(curdir)

            # Remove temporary centerline files
            if remove_temp_files:
                print('\nRemove temporary files...')
                sct.run('rm -rf '+path_tmp)


  #Process for a text file as output:
    if parameter == "text_file" :
        print "\nText file process"
        #Process for only a segmentation file:
        if "-i" in arguments and "-l" not in arguments:

                    # Extract path, file and extension
            segmentation_file = os.path.abspath(segmentation_file)
            path_data, file_data, ext_data = sct.extract_fname(segmentation_file)


            path_tmp = sct.tmp_create()

            # copy files into tmp folder
            sct.run('cp '+segmentation_file+' '+path_tmp)

            # go to tmp folder
            curdir = os.getcwd()
            os.chdir(path_tmp)

            # Change orientation of the input segmentation into RPI
            print '\nOrient segmentation image to RPI orientation...'
            fname_segmentation_orient = 'tmp.segmentation_rpi' + ext_data
            set_orientation(file_data+ext_data, 'RPI', fname_segmentation_orient)

            # Extract orientation of the input segmentation
            orientation = get_orientation(file_data+ext_data)
            print '\nOrientation of segmentation image: ' + orientation

            # Get size of data
            print '\nGet dimensions data...'
            nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_segmentation_orient)
            print '.. '+str(nx)+' x '+str(ny)+' y '+str(nz)+' z '+str(nt)

            print '\nOpen segmentation volume...'
            file = nibabel.load(fname_segmentation_orient)
            data = file.get_data()
            hdr = file.get_header()

            # Extract min and max index in Z direction
            X, Y, Z = (data>0).nonzero()
            min_z_index, max_z_index = min(Z), max(Z)
            x_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
            y_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
            z_centerline = [iz for iz in range(min_z_index, max_z_index+1)]
            # Extract segmentation points and average per slice
            for iz in range(min_z_index, max_z_index+1):
                x_seg, y_seg = (data[:,:,iz]>0).nonzero()
                x_centerline[iz-min_z_index] = np.mean(x_seg)
                y_centerline[iz-min_z_index] = np.mean(y_seg)
            for k in range(len(X)):
                data[X[k],Y[k],Z[k]] = 0
            # Fit the centerline points with splines and return the new fitted coordinates
            x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)

            # Create output text file
            if output_file_name != None :
                file_name = output_file_name
            else: file_name = file_data+'_centerline'+'.txt'

            sct.printv('\nWrite text file...', verbose)
            #file_results = open("../"+file_name, 'w')
            file_results = open(file_name, 'w')
            for i in range(min_z_index, max_z_index+1):
                file_results.write(str(int(i)) + ' ' + str(x_centerline_fit[i-min_z_index]) + ' ' + str(y_centerline_fit[i-min_z_index]) + '\n')
            file_results.close()

            # Retrieve result
            sct.run('cp '+file_name+' ../')

            del data

            # come back
            os.chdir(curdir)


           # Remove temporary files
            if remove_temp_files:
                print('\nRemove temporary files...')
                sct.run('rm -rf '+path_tmp)

            return file_name


        #Process for only a label file:
        if "-l" in arguments and "-i" not in arguments:
            file = os.path.abspath(label_file)
            path_data, file_data, ext_data = sct.extract_fname(file)

            file = nibabel.load(label_file)
            data = file.get_data()
            hdr = file.get_header()

            X,Y,Z = (data>0).nonzero()
            Z_new = np.linspace(min(Z),max(Z),(max(Z)-min(Z)+1))

            # sort X and Y arrays using Z
            X = [X[i] for i in Z[:].argsort()]
            Y = [Y[i] for i in Z[:].argsort()]
            Z = [Z[i] for i in Z[:].argsort()]

            #print X, Y, Z

            f1 = interpolate.UnivariateSpline(Z, X)
            f2 = interpolate.UnivariateSpline(Z, Y)

            X_fit = f1(Z_new)
            Y_fit = f2(Z_new)

            #print X_fit
            #print Y_fit

            if verbose==1 :
                import matplotlib.pyplot as plt

                plt.figure()
                plt.plot(Z_new,X_fit)
                plt.plot(Z,X,'o',linestyle = 'None')
                plt.show()

                plt.figure()
                plt.plot(Z_new,Y_fit)
                plt.plot(Z,Y,'o',linestyle = 'None')
                plt.show()

            data =data*0

            for iz in xrange(len(X_fit)):
                data[X_fit[iz],Y_fit[iz],Z_new[iz]] = 1

            # Create output text file
            sct.printv('\nWrite text file...', verbose)
            if output_file_name != None :
                file_name = output_file_name
            else: file_name = file_data+'_centerline'+ext_data
            file_results = open(file_name, 'w')
            min_z_index, max_z_index = min(Z), max(Z)
            for i in range(min_z_index, max_z_index+1):
                file_results.write(str(int(i)) + ' ' + str(X_fit[i-min_z_index]) + ' ' + str(Y_fit[i-min_z_index]) + '\n')
            file_results.close()

            del data

        #Process for a segmentation file and a label file:
        if "-l" and "-i" in arguments:

            ## Creation of a temporary file that will contain each centerline file of the process
            path_tmp = sct.tmp_create()

            ##From label file create centerline text file
            print '\nPROCESS PART 1: From label file create centerline text file.'
            file_label = os.path.abspath(label_file)
            path_data_label, file_data_label, ext_data_label = sct.extract_fname(file_label)

            file_label = nibabel.load(label_file)

            #Copy label_file into temporary folder
            sct.run('cp '+label_file+' '+path_tmp)

            data_label = file_label.get_data()
            hdr_label = file_label.get_header()

            X,Y,Z = (data_label>0).nonzero()
            Z_new = np.linspace(min(Z),max(Z),(max(Z)-min(Z)+1))

            # sort X and Y arrays using Z
            X = [X[i] for i in Z[:].argsort()]
            Y = [Y[i] for i in Z[:].argsort()]
            Z = [Z[i] for i in Z[:].argsort()]

            #print X, Y, Z

            f1 = interpolate.UnivariateSpline(Z, X)
            f2 = interpolate.UnivariateSpline(Z, Y)

            X_fit = f1(Z_new)
            Y_fit = f2(Z_new)

            #print X_fit
            #print Y_fit

            if verbose==1 :
                import matplotlib.pyplot as plt

                plt.figure()
                plt.plot(Z_new,X_fit)
                plt.plot(Z,X,'o',linestyle = 'None')
                plt.show()

                plt.figure()
                plt.plot(Z_new,Y_fit)
                plt.plot(Z,Y,'o',linestyle = 'None')
                plt.show()

            data_label =data_label*0

            for i in xrange(len(X_fit)):
                data_label[X_fit[i],Y_fit[i],Z_new[i]] = 1

            # Create output text file
            sct.printv('\nWrite text file...', verbose)
            file_name_label = file_data_label+'_centerline'+'.txt'
            file_results = io.open(os.path.join(path_tmp, file_name_label), 'w')
            min_z_index, max_z_index = min(Z), max(Z)
            for i in range(min_z_index, max_z_index+1):
                file_results.write(str(int(i)) + ' ' + str(X_fit[i-min_z_index]) + ' ' + str(Y_fit[i-min_z_index]) + '\n')
            file_results.close()

            # copy files into tmp folder
            #sct.run('cp '+file_name_label+' '+path_tmp)

            del data_label


            ##From segmentation file create centerline text file
            print '\nPROCESS PART 2: From segmentation file create centerline image.'
            # Extract path, file and extension
            segmentation_file = os.path.abspath(segmentation_file)
            path_data_seg, file_data_seg, ext_data_seg = sct.extract_fname(segmentation_file)

            # copy files into tmp folder
            sct.run('cp '+segmentation_file+' '+path_tmp)

            # go to tmp folder
            curdir = os.getcwd()
            os.chdir(path_tmp)

            # Change orientation of the input segmentation into RPI
            print '\nOrient segmentation image to RPI orientation...'
            fname_segmentation_orient = 'tmp.segmentation_rpi' + ext_data_seg
            set_orientation(file_data_seg+ext_data_seg, 'RPI', fname_segmentation_orient)

            # Extract orientation of the input segmentation
            orientation = get_orientation(file_data_seg+ext_data_seg)
            print '\nOrientation of segmentation image: ' + orientation

            # Get size of data
            print '\nGet dimensions data...'
            nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_segmentation_orient)
            print '.. '+str(nx)+' x '+str(ny)+' y '+str(nz)+' z '+str(nt)

            print '\nOpen segmentation volume...'
            file_seg = nibabel.load(fname_segmentation_orient)
            data_seg = file_seg.get_data()
            hdr_seg = file_seg.get_header()

            # Extract min and max index in Z direction
            X, Y, Z = (data_seg>0).nonzero()
            min_z_index, max_z_index = min(Z), max(Z)
            x_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
            y_centerline = [0 for i in range(0,max_z_index-min_z_index+1)]
            z_centerline = [iz for iz in range(min_z_index, max_z_index+1)]
            # Extract segmentation points and average per slice
            for iz in range(min_z_index, max_z_index+1):
                x_seg, y_seg = (data_seg[:,:,iz]>0).nonzero()
                x_centerline[iz-min_z_index] = np.mean(x_seg)
                y_centerline[iz-min_z_index] = np.mean(y_seg)
            for k in range(len(X)):
                data_seg[X[k],Y[k],Z[k]] = 0
            # Fit the centerline points with splines and return the new fitted coordinates
                    #done with nurbs for now
            x_centerline_fit, y_centerline_fit,x_centerline_deriv,y_centerline_deriv,z_centerline_deriv = b_spline_centerline(x_centerline,y_centerline,z_centerline)


             # Create output text file
            file_name_seg = file_data_seg+'_centerline'+'.txt'
            sct.printv('\nWrite text file...', verbose)
            file_results = open(file_name_seg, 'w')
            for i in range(min_z_index, max_z_index+1):
                file_results.write(str(int(i)) + ' ' + str(x_centerline_fit[i-min_z_index]) + ' ' + str(y_centerline_fit[i-min_z_index]) + '\n')
            file_results.close()

            del data_seg


            print '\nRemoving overlap of the centerline obtain with label file if there are any:'

            ## Remove overlap from centerline file obtain with label file
            remove_overlap(file_name_label, file_name_seg, "generated_centerline_without_overlap1.txt", parameter=1)

            ## Concatenation of the two centerline files
            print '\nConcatenation of the two centerline files:'
            if output_file_name != None :
                file_name = output_file_name
            else: file_name = 'centerline_total_from_label_and_seg.txt'

            f_output = open(file_name, "w")
            f_output.close()
            with open(file_name_seg, "r") as f_seg:
                with open("generated_centerline_without_overlap1.txt", "r") as f:
                    with open(file_name, "w") as f_output:
                        data_line_seg = f_seg.readlines()
                        data_line = f.readlines()
                        for line in data_line_seg :
                            f_output.write(line)
                        for line in data_line :
                            f_output.write(line)

            # Retrieve result
            sct.copy(file_name, curdir)

            # Come back
            os.chdir(curdir)

            # Remove temporary centerline files
            if remove_temp_files:
                print('\nRemove temporary files...')
                sct.run('rm -rf '+path_tmp)

########## Function remove overlap of a centerline file  ###########
def remove_overlap(file_centerline_generated_by_labels, file_with_seg_or_centerline, output_file_name, parameter=0):
    # Image file process
    if parameter == 0 :
        image_with_seg_or_centerline = Image(file_with_seg_or_centerline).copy()
        z_test = ComputeZMinMax(image_with_seg_or_centerline)
        zmax = z_test.Zmax
        zmin = z_test.Zmin

        tab1 = Image(file_centerline_generated_by_labels).copy()
        size_x=tab1.data.shape[0]
        size_y=tab1.data.shape[1]

        #X_coor, Y_coor, Z_coor = (tab1.data).nonzero()
        #nb_one = X_coor.shape[0]
        #for i in range(0, nb_one):
        #    tab1.data[X_coor[i], Y_coor[i], X_coor[i]] = 0

        #each slice under zmax is filled with zeros
        print zmax
        for i in range(zmin, zmax):
            tab1.data[:,:,i] = np.zeros((size_x,size_y))


        #Save file
        tab1.setFileName(output_file_name)
        tab1.save('minimize')   #size of image should be minimized

     # Text file process
    if parameter == 1 :
        z_test = ComputeZMinMax(file_with_seg_or_centerline)
        zmax = z_test.Zmax
        zmin = z_test.Zmin
        print zmax
        #create output txt file
        tab2 = open(output_file_name , "w")
        tab2.close()

        #Delete lines under zmax
        with open(file_centerline_generated_by_labels) as f:
            data_line = f.readlines()
            with open(output_file_name, "w") as f1:
                for line in data_line:
                    words = line.split()
                    if int(words [0]) < zmin or int(words [0]) > zmax :
                        f1.write(line)

########## Class to compute Zmin and Zmax of a centerline file (binary or text) ###########
class ComputeZMinMax():
    def __init__(self, file):
        if isinstance(file, Image):
            self.image = file
            self.Zmin, self.Zmax = self.computeZ_for_image()
        elif isinstance(file, str):
            self.text = file
            self.Zmin, self.Zmax = self.computeZ_for_text()
        else:
            print "ERROR: input file is not a text file or an instance Image."


    def computeZ_for_image(self):
        Zmin=0
        Zmax=0
        x=self.image.data.shape
        i=0
        while np.max(self.image.data[:,:,i]) == 0 and i<x[2]:
            i=i+1
        Zmin = i  #Zmin is the last slice, or bottom slice, with a non-zero value

        i=x[2]
        while np.max(self.image.data[:,:,i-1]) == 0 and i>-1:
            i=i-1
        Zmax = i-1  #Zmax is the last slice, top slice, with a non zero value

        return Zmin, Zmax

    def computeZ_for_text(self):
        total=[]
        with open(self.text) as f:
            data_line = f.readlines()   #data_line is a list
            for line in data_line:
                words = line.split()
                total.extend(words)
        total=np.asarray(total)
        Z_min=int(total[0])   #Zmin is the last slice, or bottom slice, with a non-zero value
        Z_max=int(total[-3])  #Zmax is the last slice, top slice, with a non zero value
        return Z_min, Z_max
###############################




#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters


    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Utility function for labels.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Segmentation NIFTI file. Must be 3D.",
                      mandatory=False)
    parser.add_option(name="-l",
                      type_value="file",
                      description="NIFTI file presenting labels. Must be 3D.",
                      mandatory=False)
    parser.add_option(name="-p",
                      type_value="multiple_choice",
                      description="Type of output wanted: \nbinary_centerline: return binary NIFTI file of the centerline \ntext_file: return text file with float coordinates according to z",
                      mandatory=False,
                      example=["binary_centerline","text_file"],
                      default_value="binary_centerline")

    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Name of the output NIFTI image with the centerline, or of the output text file with the coordinates according to z.",
                      mandatory=False,
                      default_value="labels.nii.gz")

    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="Remove temporary files. Specify 0 to get access to temporary files.",
                      mandatory=False,
                      example=['0','1'],
                      default_value="1")
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="Verbose. 0: nothing. 1: basic. 2: extended.",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1', '2'])
    arguments = parser.parse(sys.argv[1:])

    #parameter = "binary_centerline"
    #remove_temp_files = 1
    # verbose = 0
    parameter = arguments["-p"]
    remove_temp_files = int(arguments["-r"])
    verbose = int(arguments["-v"])

    if "-i" in arguments:
        segmentation_file = arguments["-i"]
    else: segmentation_file = None
    if "-l" in arguments:
        label_file = arguments["-l"]
    else: label_file = None
    if "-i" not in arguments and "-l" not in arguments:
        print """\nERROR: Arguments must contain "-i" or "-l". """
        sys.exit()
    if "-o" in arguments:
        output_file_name = arguments["-o"]
    else: output_file_name = None

    #sct_obtain_centerline(segmentation_file="data_RPI_seg.nii.gz", label_file="labels_brainstem_completed.nii.gz", output_file_name="centerline_from_label_and_seg.nii.gz", parameter = "binary_centerline", remove_temp_files = 0, verbose = 0 )
    #sct_obtain_centerline(segmentation_file = "data_RPI_seg.nii.gz", label_file="labels_brainstem_completed.nii.gz", output_file_name="test_label_and_seg.txt", parameter = "text_file", remove_temp_files = 1, verbose = 0 )
    main(segmentation_file, label_file, output_file_name, parameter, remove_temp_files, verbose)

#segmentation_file="data_RPI_seg.nii.gz"
#label_file="labels_brainstem_completed.nii.gz"
#comm for binary seg   -i data_RPI_seg.nii.gz -o seg_centerline.nii.gz -p binary_centerline
