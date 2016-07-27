#!/usr/bin/env python
#
# Test major functions.
#
# Authors: Julien Cohen-Adad, Benjamin De Leener, Augustin Roux
# Updated: 2014-10-06

# TODO: list functions to test in help (do a search in testing folder)


import sys
import time

import os
from msct_parser import Parser

# get path of the toolbox
# TODO: put it back below when working again (julien 2016-04-04)
# <<<
# OLD
# status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# NEW
path_script = os.path.dirname(__file__)
path_sct = os.path.dirname(path_script)
# >>>
# append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
sys.path.append(path_sct + '/testing')
import sct_utils as sct
import importlib

# define nice colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

# JULIEN: NOW THAT THE USER WILL HAVE ACCESS TO TEST_ALL, WE SHOULD NOT USE $SCT_TESTING_DATA_DIR ANYMORE.
# get path of testing data
# status, path_sct_testing = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')


class param:
    def __init__(self):
        self.download = 0
        self.path_data = 'sct_testing_data/'
        self.function_to_test = None
        # self.function_to_avoid = None
        self.remove_tmp_file = 0
        self.verbose = 1
        # self.url_git = 'https://github.com/neuropoly/sct_testing_data.git'
        self.path_tmp = ""


# START MAIN
# ==========================================================================================
def main():

    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    if '-d' in arguments:
        param.download = int(arguments['-d'])
    if '-p' in arguments:
        param.path_data = arguments['-p']
    if '-f' in arguments:
        param.function_to_test = arguments['-f']
    if '-r' in arguments:
        param.remove_tmp_file = int(arguments['-r'])

    # path_data = param.path_data
    function_to_test = param.function_to_test
    # function_to_avoid = param.function_to_avoid
    remove_tmp_file = param.remove_tmp_file

    start_time = time.time()

    # # download data
    # if param.download:
    #     downloaddata()
    #
    # get absolute path and add slash at the end
    param.path_data = sct.slash_at_the_end(os.path.abspath(param.path_data), 1)

    # check existence of testing data folder
    if not os.path.isdir(param.path_data) or param.download:
        downloaddata()

    # display path to data
    sct.printv('\nPath to testing data: '+param.path_data, param.verbose)

    # create temp folder that will have all results and go in it
    param.path_tmp = sct.tmp_create()
    # param.path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
    # sct.create_folder(param.path_tmp)
    os.chdir(param.path_tmp)

    # get list of all scripts to test
    functions = fill_functions()

    # loop across all functions and test them
    status = []
    [status.append(test_function(f)) for f in functions if function_to_test == f]
    if not status:
        for f in functions:
            status.append(test_function(f))
    print 'status: '+str(status)

    # display elapsed time
    elapsed_time = time.time() - start_time
    print 'Finished! Elapsed time: '+str(int(round(elapsed_time)))+'s\n'

    # remove temp files
    if param.remove_tmp_file:
        sct.printv('\nRemove temporary files...', param.verbose)
        sct.run('rm -rf '+param.path_tmp, param.verbose)

    e = 0
    if sum(status) != 0:
        e = 1
    print e

    sys.exit(e)


def downloaddata():
    sct.printv('\nDownloading testing data...', param.verbose)
    sct.run('sct_download_data -d sct_testing_data')


# list of all functions to test
# ==========================================================================================
def fill_functions():
    functions = []
    functions.append('sct_apply_transfo')
    # functions.append('sct_check_atlas_integrity')
    functions.append('sct_compute_mtr')
    functions.append('sct_concat_transfo')
    functions.append('sct_convert')
    #functions.append('sct_convert_binary_to_trilinear')  # not useful
    functions.append('sct_create_mask')
    functions.append('sct_crop_image')
    functions.append('sct_dmri_compute_dti')
    functions.append('sct_dmri_create_noisemask')
    functions.append('sct_dmri_get_bvalue')
    functions.append('sct_dmri_transpose_bvecs')
    functions.append('sct_dmri_moco')
    functions.append('sct_dmri_separate_b0_and_dwi')
    functions.append('sct_documentation')
    functions.append('sct_extract_metric')
    # functions.append('sct_flatten_sagittal')
    functions.append('sct_fmri_compute_tsnr')
    functions.append('sct_fmri_moco')
    # functions.append('sct_get_centerline')
    functions.append('sct_image')
    functions.append('sct_label_utils')
    functions.append('sct_label_vertebrae')
    functions.append('sct_maths')
    functions.append('sct_process_segmentation')
    functions.append('sct_propseg')
    functions.append('sct_register_graymatter')
    functions.append('sct_register_multimodal')
    functions.append('sct_register_to_template')
    functions.append('sct_resample')
    functions.append('sct_segment_graymatter')
    functions.append('sct_smooth_spinalcord')
    functions.append('sct_straighten_spinalcord')
    functions.append('sct_warp_template')
    return functions


# print without carriage return
# ==========================================================================================
def print_line(string):
    import sys
    sys.stdout.write(string + make_dot_lines(string))
    sys.stdout.flush()


# fill line with dots
# ==========================================================================================
def make_dot_lines(string):
    if len(string) < 52:
        dot_lines = '.'*(52 - len(string))
        return dot_lines
    else:
        return ''


# print in color
# ==========================================================================================
def print_ok():
    print "[" + bcolors.OKGREEN + "OK" + bcolors.ENDC + "]"

def print_warning():
    print "[" + bcolors.WARNING + "WARNING" + bcolors.ENDC + "]"

def print_fail():
    print "[" + bcolors.FAIL + "FAIL" + bcolors.ENDC + "]"


# write to log file
# ==========================================================================================
def write_to_log_file(fname_log, string, mode='w'):
    """
    status, output = sct.run('echo $SCT_DIR', 0)
    path_logs_dir = output + '/testing/logs'

    if not os.path.isdir(path_logs_dir):
        os.makedirs(path_logs_dir)
    """

    string = "test ran at "+time.strftime("%y%m%d%H%M%S")+"\n" \
             + fname_log \
             + string
    # open file
    try:
        f = open(fname_log, mode)
    except Exception as ex:
        raise Exception('WARNING: Cannot open log file.')
    f.write(string+'\n')
    f.close()


# test function
# ==========================================================================================
def test_function(script_name):
    # if script_name == 'test_debug':
    #     return test_debug()  # JULIEN
    # else:
    # build script name
    fname_log = '../' + script_name + ".log"
    tmp_script_name = script_name
    result_folder = "results_"+script_name
    script_name = "test_"+script_name

    sct.create_folder(result_folder)
    os.chdir(result_folder)

    # display script name
    print_line('Checking '+script_name)
    # import function as a module
    script_tested = importlib.import_module(script_name)
    # test function
    result_test = script_tested.test(param.path_data)
    # test functions can return 2 or 3 variables, depending if there is results.
    # In this script, we look only at the first two variables.
    status, output = result_test[0], result_test[1]
    # write log file
    write_to_log_file(fname_log, output, 'w')
    # manage status
    if status == 0:
        print_ok()
    else:
        if status == 99:
            print_warning()
        else:
            print_fail()
        print output
    # go back to parent folder
    os.chdir('..')

    # return
    return status

def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Crash and integrity testing for functions of the Spinal Cord Toolbox. Internet connection is required for downloading testing data.')
    parser.add_option(name="-f",
                      type_value="str",
                      description="Test this specific script (do not add extension).",
                      mandatory=False,
                      example='sct_propseg')
    parser.add_option(name="-d",
                      type_value="multiple_choice",
                      description="Download testing data.",
                      mandatory=False,
                      default_value=param.download,
                      example=['0', '1'])
    parser.add_option(name="-p",
                      type_value="folder",
                      description='Path to testing data. NB: no need to set if using "-d 1"',
                      mandatory=False,
                      default_value=param.path_data)
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    return parser


if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()