#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_register_multimodal script
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/09/28
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import commands
import sct_utils as sct
import sct_register_multimodal
from copy import deepcopy
from pandas import DataFrame
import os

def test(path_data='', parameters=''):
    output = ''
    status = 0
    verbose = 0

    if not parameters:
        folder_data = 'mt/'
        file_data = ['mt0.nii.gz', 'mt1.nii.gz', 'mt0_seg.nii.gz', 'mt1_seg.nii.gz']

        parameters = []
        list_validation = []

        # check syn
        algo = 'syn'
        cmd = '-i '  + folder_data + file_data[0] \
              + ' -d ' +  folder_data + file_data[1] \
              + ' -o ' + sct.add_suffix(file_data[0], '_reg_'+algo)  \
              + ' -param step=1,algo='+algo+',type=im,iter=1,smooth=1,shrink=2,metric=MI'  \
              + ' -x linear' \
              + ' -r 0' \
              + ' -v 1'

        parameters.append(cmd)
        list_validation.append((algo, sct.add_suffix(file_data[0], '_reg_'+algo), path_data+folder_data+sct.add_suffix(file_data[0], '_reg_'+algo+'_goldstandard') ))

        # output += '\n====================================================================================================\n'\
        #           +cmd+\
        #           '\n====================================================================================================\n\n'  # copy command
        # s, o = commands.getstatusoutput(cmd)
        # status += s
        # output += o
        # if status == 0:
        #     s, o = check_integrity(algo=algo, fname_result=sct.add_suffix(file_data[0], '_reg_'+algo), fname_goldstandard=path_data+folder_data+sct.add_suffix(file_data[0], '_reg_'+algo+'_goldstandard'))
        #     status += s
        #     output += o

        # check slicereg
        algo = 'slicereg'
        cmd = '-i ' +  folder_data + file_data[0] \
              + ' -d ' +  folder_data + file_data[1] \
              + ' -o ' + sct.add_suffix(file_data[0], '_reg_'+algo)  \
              + ' -param step=1,algo='+algo+',type=im,iter=5,smooth=0,metric=MeanSquares'  \
              + ' -x linear' \
              + ' -r 0' \
              + ' -v 1'

        parameters.append(cmd)
        list_validation.append((algo, sct.add_suffix(file_data[0], '_reg_'+algo), path_data+folder_data+sct.add_suffix(file_data[0], '_reg_'+algo+'_goldstandard') ))

        # output += '\n====================================================================================================\n'\
        #           +cmd+\
        #           '\n====================================================================================================\n\n'  # copy command
        # s, o = commands.getstatusoutput(cmd)
        # status += s
        # output += o
        # if status == 0:
        #     s, o = check_integrity(algo=algo, fname_result=sct.add_suffix(file_data[0], '_reg_'+algo), fname_goldstandard=path_data+folder_data+sct.add_suffix(file_data[0], '_reg_'+algo+'_goldstandard'))
        #     status += s
        #     output += o

        # check centermass
        algo = 'centermass'
        cmd = ' -i ' +  folder_data + file_data[0] \
              + ' -d ' +  folder_data + file_data[1] \
              + ' -iseg ' +  folder_data + file_data[2] \
              + ' -dseg ' +  folder_data + file_data[3] \
              + ' -o ' + sct.add_suffix(file_data[0], '_reg_'+algo)  \
              + ' -param step=1,type=seg,algo='+algo+',smooth=1'  \
              + ' -x linear' \
              + ' -r 0' \
              + ' -v 1'

        parameters.append(cmd)
        list_validation.append(None)

        # output += '\n====================================================================================================\n'\
        #           +cmd+\
        #           '\n====================================================================================================\n\n'  # copy command
        # s, o = commands.getstatusoutput(cmd)
        # status += s
        # output += o

        # check centermassrot
        algo = 'centermassrot'
        cmd = ' -i ' +  folder_data + file_data[0] \
              + ' -d ' +  folder_data + file_data[1] \
              + ' -iseg ' +  folder_data + file_data[2] \
              + ' -dseg ' +  folder_data + file_data[3] \
              + ' -o ' + sct.add_suffix(file_data[0], '_reg_'+algo)  \
              + ' -param step=1,type=seg,algo='+algo+',smooth=1'  \
              + ' -x linear' \
              + ' -r 0' \
              + ' -v 1'

        parameters.append(cmd)
        list_validation.append(None)

        # output += '\n====================================================================================================\n'\
        #           +cmd+\
        #           '\n====================================================================================================\n\n'  # copy command
        # s, o = commands.getstatusoutput(cmd)
        # status += s
        # output += o

        # check columnwise
        algo = 'columnwise'
        cmd = ' -i ' +  folder_data + file_data[0] \
              + ' -d ' +  folder_data + file_data[1] \
              + ' -iseg ' +  folder_data + file_data[2] \
              + ' -dseg ' +  folder_data + file_data[3] \
              + ' -o ' + sct.add_suffix(file_data[0], '_reg_'+algo)  \
              + ' -param step=1,type=seg,algo='+algo+',smooth=1'  \
              + ' -x linear' \
              + ' -r 0' \
              + ' -v 1'

        parameters.append(cmd)
        list_validation.append(None)

        # output += '\n====================================================================================================\n'\
        #           +cmd+\
        #           '\n====================================================================================================\n\n'  # copy command
        # s, o = commands.getstatusoutput(cmd)
        # status += s
        # output += o
    else:
        parameters = [parameters]
        list_validation = [None]

    for param, val in zip(parameters, list_validation):
        parser = sct_register_multimodal.get_parser()
        dict_param = parser.parse(param.split(), check_file_exist=False)
        dict_param_with_path = parser.add_path_to_file(deepcopy(dict_param), path_data, input_file=True)
        param_with_path = parser.dictionary_to_string(dict_param_with_path)

        if not (os.path.isfile(dict_param_with_path['-i']) and os.path.isfile(dict_param_with_path['-d'])):
            status = 200
            output = 'ERROR: the file(s) provided to test function do not exist in folder: ' + path_data
            return status, output, DataFrame(data={'status': int(status), 'output': output}, index=[path_data])

        import time, random
        subject_folder = path_data.split('/')
        if subject_folder[-1] == '' and len(subject_folder) > 1:
            subject_folder = subject_folder[-2]
        else:
            subject_folder = subject_folder[-1]
        path_output = sct.slash_at_the_end(
            'sct_register_multimodal_' + subject_folder + '_' + time.strftime("%y%m%d%H%M%S") + '_' + str(
                random.randint(1, 1000000)), slash=1)
        param_with_path += ' -ofolder ' + path_output

        cmd = 'sct_register_multimodal ' + param_with_path
        output = '\n====================================================================================================\n' + cmd + '\n====================================================================================================\n\n'  # copy command
        time_start = time.time()
        try:
            status, o = sct.run(cmd, verbose)
        except:
            status, o = 1, 'ERROR: Function crashed!'
        output += o
        duration = time.time() - time_start

        if val is not None:
            s, o = check_integrity(val[0], path_output+val[1], val[2])
            status += s
            output += o

    results = DataFrame(data={'status': int(status), 'output': output, 'duration [s]': duration}, index=[path_data])

    return status, output, results


def check_integrity(algo='', fname_result='', fname_goldstandard=''):
    """
    Check integrity between registered image and gold-standard
    :param algo:
    :return:
    """
    status = 0
    output = '\nChecking integrity between: \n  Result: '+fname_result+'\n  Gold-standard: '+fname_goldstandard

    from msct_image import Image
    # compare with gold-standard registration
    im_gold = Image(fname_goldstandard)
    data_gold = im_gold.data
    data_res = Image(fname_result).data
    # get dimensions
    nx, ny, nz, nt, px, py, pz, pt = im_gold.dim
    # set the difference threshold to 1e-3 pe voxel
    threshold = 1e-3 * nx * ny * nz * nt
    # check if non-zero elements are present when computing the difference of the two images
    diff = data_gold - data_res
    # report result
    import numpy as np
    output += '\nDifference between the two images: '+str(abs(np.sum(diff)))
    output += '\nThreshold: '+str(threshold)
    if abs(np.sum(diff)) > threshold:
        Image(param=diff, absolutepath='res_differences_from_gold_standard.nii.gz').save()
        status = 99
        output += '\nWARNING: Difference is higher than threshold.'
    return status, output

if __name__ == "__main__":
    # call main function
    test()