#!/usr/bin/env python
#########################################################################################
#
# Validation of WM atlas
#
# This script generates a synthetic volume from an atlas of white matter
# It then estimates the np.mean value within each tract using different methods.
# The estimation is afterwards substracted to the real value in the tracts to find the absolute error
# For each method, this process is repeated a certain number of iterations(bootstrap_iterations)
# The np.mean of all absolute deviations and the respective np.std is calculated for each method
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Charles Naaman, Julien Cohen-Adad
# Modified: 2014-11-25
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO : Make a function for which the estimation is made from a particular method(remove code duplication in bootstrap loop and parameters initialization)
# TODO : Make a function to print results depending on which methods are used
# TODO : Add usage 
# TODO : Add class initialization of default parameters (not working for some reason)


# Import common Python libraries
import os
import sys
import time
import commands
import shutil
import numpy as np
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
import sct_utils as sct
from generate_phantom import phantom_generation, get_tracts, save_3D_nparray_nifti

def main():
    # Parameters
    bootstrap_iter = 1
    folder_atlas = '../create_atlas/final_results/'  # path to atlas. add / at the end
    mask_folder = ['manual_masks/charles/', 'manual_masks/julien/', 'manual_masks/tanguy/', 'manual_masks/simon/']  # folder of manual masks
    std_noise_list = [0, 20] #[0, 5, 10, 20, 50]  # standard deviation of the noise added to the generated phantom
    range_tract_list = [0] #[0, 5, 10, 20, 50]  # in percent
    fixed_range = 10  # in percent
    fixed_noise = 10  # in percent
    results_folder = 'results/'  # add / at the end

    # create output folder
    create_folder(results_folder)

    # loop across noise levels
    range_tract = fixed_range
    for std_noise in std_noise_list:
        results_file = 'results_noise'+str(std_noise)+'_range'+str(range_tract)+'.txt'
        validate_atlas(folder_atlas, bootstrap_iter, std_noise, range_tract, results_folder+results_file, mask_folder)

    # # loop across tract ranges
    # std_noise = fixed_noise
    # for range_tract in range_tract_list:
    #     results_file = 'results_noise'+str(std_noise)+'_range'+str(range_tract)+'.txt'
    #     validate_atlas(folder_atlas, bootstrap_iter, std_noise, range_tract, results_folder+results_file, mask_folder)


def validate_atlas(folder_atlas, nb_bootstraps, std_noise, range_tract, results_file, mask_folder):
    # Parameters
    folder_cropped_atlas = "cropped_atlas/"
    crop = 0  # crop atlas, default=1. Only need to do it once (saves time).
    zcrop_ind = [10, 110, 210, 310, 410]
    file_phantom = "WM_phantom.nii.gz"
    file_phantom_noise = "WM_phantom_noise.nii.gz"
    file_tract_sum = "tracts_sum.nii.gz"
    true_value = 40
    file_extract_metrics = "metric_label.txt"
    list_tracts = ['2', '17', '0,1,15,16']
    list_tracts_txt = ['csl', 'csr', 'dc']
    index_dorsalcolumn = 2  # index of dorsal column in list_tracts
    list_methods = ['ml', 'wa', 'wath', 'bin', 'man0', 'man1', 'man2', 'man3']
    # dorsal_column_labels = '0,1,15,16'
    # nb_tracts_dorsalcolumn = 4
    zero_last_tract = 1  # if last tract correspond to CSF, zero it

    # Parameters for the manual estimation
    # man_mask_index = 2, 17, 30  # 30 corresponds to the dorsal column mask
    #mask_prefix = 'mask_tract_'
    mask_prefix = 'manual_'
    mask_ext = '.nii.gz'

    # initialization
    start_time = time.time()  # save start time for duration
    folder_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S/")
    nb_methods = len(list_methods)
    nb_tracts = len(list_tracts)
    x_estim = np.zeros(shape=(nb_tracts, nb_methods, nb_bootstraps))
    x_true_i = np.zeros(shape=(nb_tracts))
    fname_phantom = folder_tmp+file_phantom
    fname_phantom_noise = folder_tmp+file_phantom_noise
    fname_tract_sum = folder_tmp+file_tract_sum
    # man_users_number = len(mask_prefix)
    # X_man = np.zeros([man_users_number, nb_tracts+1])
    # X_man_dc = np.zeros([man_users_number, nb_bootstraps])
    # D_man = np.zeros([man_users_number, nb_tracts, nb_bootstraps])
    # D_man_dc = np.zeros([man_users_number, 1, nb_bootstraps])

    # Crop the atlas
    if crop == 1:
        create_folder(folder_cropped_atlas)
        crop_atlas(folder_atlas, folder_cropped_atlas, zcrop_ind)
        #sct.run('./crop_atlas.py -f '+folder_atlas+' -o '+folder_cropped_atlas+' -z '+str(zcrop_ind))
        # Copy the info_label.txt file in the cropped atlas' folder
        # This file needs to be there in order for the sct_extract_metric code to work
        sct.run('cp ../WM_atlas_generation/info_label.txt '+folder_cropped_atlas)

    # Extract the tracts from the atlas' folder
    tracts = get_tracts(folder_cropped_atlas)

    # get file name of the first atlas file
    fname_atlas = folder_cropped_atlas+'WMtract__00.nii.gz'

    # Get ponderation of each tract for dorsal column average ponderation of each tract of the dorsal column
    list_tract_dorsalcolumn = list_tracts[index_dorsalcolumn].split(',')
    nb_tracts_dorsalcolumn = len(list_tract_dorsalcolumn)

    pond_dc = np.zeros(nb_tracts_dorsalcolumn)
    # sum of each 
    pond_sum = 0
    for i in range(nb_tracts_dorsalcolumn):
        # i = int(i)
        # Sum tracts values which are higher than 0 in the tracts
        pond_dc[i] = sum(tracts[int(list_tract_dorsalcolumn[i]), 0][tracts[int(list_tract_dorsalcolumn[i]), 0] > 0])
        pond_sum = pond_sum + pond_dc[i]
    # Normalize the sum of ponderations to 1 
    pond_dc = pond_dc / pond_sum

    # create temporary folder
    sct.run('mkdir '+folder_tmp)

    # loop across bootstrap
    for i_bootstrap in range(0, nb_bootstraps):
        sct.printv('Iteration:  ' + str(i_bootstrap+1) + '/' + str(nb_bootstraps), 1, 'warning')

        # Generate phantom
        [WM_phantom, WM_phantom_noise, values_synthetic_data, tracts_sum] = phantom_generation(tracts, std_noise, range_tract, true_value, folder_tmp, zero_last_tract)
        # Save generated phantoms as nifti image (.nii.gz)
        save_3D_nparray_nifti(WM_phantom, fname_phantom, fname_atlas)
        save_3D_nparray_nifti(WM_phantom_noise, fname_phantom_noise, fname_atlas)
        save_3D_nparray_nifti(tracts_sum, fname_tract_sum, fname_atlas)

        # Get the np.mean of all values in dorsal column in the generated phantom
        dc_val_avg = 0
        for j in range(nb_tracts_dorsalcolumn):
            dc_val_avg = dc_val_avg + values_synthetic_data[int(list_tract_dorsalcolumn[j])] * pond_dc[j]
        dc_val_avg = float(dc_val_avg)

        # build variable with true values (WARNING: HARD-CODED INDICES)
        x_true_i[0] = values_synthetic_data[int(list_tracts[0])]
        x_true_i[1] = values_synthetic_data[int(list_tracts[1])]
        x_true_i[2] = dc_val_avg

        # pause for 0.5 second, otherwise the temporary file names will have the same name and will be removed by the following function
        time.sleep(0.5)

        fname_extract_metrics = folder_tmp + file_extract_metrics

        # loop across tracts
        for i_tract in range(len(list_tracts)):

            # loop across methods
            for i_method in range(len(list_methods)):

                # display stuff
                print 'Tract: '+list_tracts[i_tract]+', Method: '+list_methods[i_method]

                # check if method is manual
                if not list_methods[i_method].find('man') == -1:
                    # find index of manual mask
                    index_manual = int(list_methods[i_method][list_methods[i_method].find('man')+3])
                    fname_mask = mask_folder[index_manual] + mask_prefix + list_tracts_txt[i_tract] + mask_ext
                    # manual extraction
                    status, output = sct.run('sct_average_data_within_mask -i ' + fname_phantom_noise + ' -m ' + fname_mask + ' -v 0')
                    x_estim_i = float(output)
                else:
                    # automatic extraction
                    # estimate metric
                    sct.run('sct_extract_metric -i ' + fname_phantom_noise + ' -f ' + folder_cropped_atlas+ ' -m '+list_methods[i_method]+' -l '+list_tracts[i_tract]+' -a -o '+fname_extract_metrics)
                    # read in txt file
                    x_estim_i = read_results(fname_extract_metrics)

                # Get the percent absolute deviation with the true value
                x_estim[i_tract, i_method, i_bootstrap] = 100 * abs(x_estim_i - x_true_i[i_tract]) / float(x_true_i[i_tract])

    # Calculate elapsed time
    elapsed_time = int(round(time.time() - start_time))

    # Extract time in minutes and seconds
    sec = elapsed_time % 60
    mte = (elapsed_time - sec) / 60

    # Open text file where results are printed
    results_text = open(results_file, 'w+')

    # print header
    print >>results_text, '# Mean(std) percentage of absolute error.'
    print >>results_text, '# Generated on: ' + time.strftime('%Y-%m-%d %H:%M:%S')
    print >>results_text, '# sigma noise: ' + str(std_noise) + '%'
    print >>results_text, '# range tracts: (-' + str(range_tract) + '%:+' + str(range_tract) + '%)'
    print >>results_text, '# true_value: ' + str(true_value)
    print >>results_text, '# number of iterations: ' + str(nb_bootstraps)
    print >>results_text, '# elapsed time: ' + str(mte) + 'min' + str(sec) + 's'
    text_methods = 'Label'
    # loop across methods
    for i_method in range(len(list_methods)):
        text_methods = text_methods + ', ' + list_methods[i_method]
    print >>results_text, text_methods

    # print results
    # loop across tracts
    for i_tract in range(len(list_tracts)):
        text_results = list_tracts_txt[i_tract]
        # loop across methods
        for i_method in range(len(list_methods)):
            text_methods = text_methods + ', ' + list_methods[i_method]
            text_results = text_results + ', ' + str(round(np.mean(x_estim[i_tract, i_method, :]), ndigits=3))+'('+str(round(np.std(x_estim[i_tract, i_method, :]), ndigits=3))+')'
        print >>results_text, text_results

    # close file
    results_text.close()

    # display results
    status, output = sct.run('cat ' + results_file)
    print output


def create_folder(folder):
    """create folder-- delete if already exists"""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)


def read_results(fname_metrics):
    # Read file
    f = open(fname_metrics)

    # Extract all lines in the results file from sct_extract_metric
    # Do not extract lines which start with #
    lines = [lines for lines in f.readlines() if lines.strip() if not lines.startswith("#")]

    # read each line
    metrics_results = []
    for i in range(0, len(lines)):
        line = lines[i].split(',')
        # Get np.mean value of metric from column 2 of the text file
        metrics_results.append(line[2][:-1].replace(" ", ""))
    # Transform the column into a numpy array
    metrics_results = np.array(metrics_results)
    # Change the type of the values in the numpy array to float
    metrics_results = metrics_results.astype(np.float)
    return metrics_results


# def init_values(number_of_tracts, number_of_iterations):
#     # Mean estimation in tract
#     X_ = np.zeros([number_of_tracts])
#     # Mean estimation in dorsal column
#     X_dc = np.zeros([1, 1])
#     # Mean absolute error between np.mean estimation and true value in a tract
#     D_ = np.zeros([number_of_tracts,number_of_iterations])
#     # Mean absolute error between np.mean estimation and true value in dorsal column
#     D_dc = np.zeros([1, number_of_iterations])
#     return [X_, X_dc, D_, D_dc]


def crop_atlas(folder_atlas, folder_out, zind):

    # get atlas files
    status, output = sct.run('ls '+folder_atlas+'*.nii.gz', 1)
    fname_list = output.split()

    # loop across atlas
    for i in xrange(0, len(fname_list)):
        path_list, file_list, ext_list = sct.extract_fname(fname_list[i])
        # crop file and then merge back
        cmd = 'fslmerge -z '+folder_out+file_list
        for iz in zind:
            sct.run('fslroi '+fname_list[i]+' tmpcrop.z'+str(zind.index(iz))+'_'+file_list+' 0 -1 0 -1 '+str(iz)+' 1')
            cmd = cmd+' tmpcrop.z'+str(zind.index(iz))+'_'+file_list
        sct.run(cmd)

    # delete tmp file
    sct.run('rm tmpcrop.*')



if __name__ == "__main__":
    main()
