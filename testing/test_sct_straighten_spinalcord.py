#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_sctraighten_spinalcord script
#
#   replace the shell test script in sct 1.0
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# modified: 2014/09/28
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import sct_utils as sct
from msct_parser import Parser
import sct_straighten_spinalcord
from pandas import DataFrame


def test(path_data='', parameters=''):

    if not parameters:
        parameters = '-i t2/t2.nii.gz -c t2/t2_seg.nii.gz'
    # add mandatory verbose to parameters list
    parameters = parameters + ' -v 1'

    parser = sct_straighten_spinalcord.get_parser()
    dict_param = parser.parse(parameters.split(), check_file_exist=False)
    dict_param_with_path = parser.add_path_to_file(dict_param, path_data, input_file=True)
    param_with_path = Parser.dictionary_to_string(dict_param_with_path)

    cmd = 'sct_straighten_spinalcord ' + param_with_path
    status, output = sct.run(cmd, 0)

    # initialization of results: must be NaN if test fails
    result_mse, result_dist_max = float('nan'), float('nan')
    if status == 0:
        # extraction of results
        output_split = output.split('Maximum x-y error = ')[1].split(' mm')
        result_dist_max = float(output_split[0])
        result_mse = float(output_split[1].split('Accuracy of straightening (MSE) = ')[1])

        # integrity testing - straightening has been tested with v2.0.6 on several images.
        # mse is less than 1.5 and dist_max is less than 4
        if result_dist_max > 4.0 or result_mse > 1.5:
            status = 99


    # transform results into Pandas structure
    results = DataFrame(data={'mse': result_mse, 'dist_max': result_dist_max}, index=[path_data])

    return status, output, results

if __name__ == "__main__":
    # call main function
    test()