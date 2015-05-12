#!/usr/bin/env python
#
# Test major functions.
#
# Authors: Julien Cohen-Adad, Benjamin De Leener, Augustin Roux
# Updated: 2014-10-06

# TODO: list functions to test in help (do a search in testing folder)


import os
import getopt
import sys
import time
import commands
# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
import sct_utils as sct
from os import listdir
from os.path import isfile, join
import importlib

# define nice colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

# get path of testing data
status, path_sct_testing = commands.getstatusoutput('echo $SCT_TESTING_DATA_DIR')


class param:
    def __init__(self):
        self.download = 0
        self.path_data = sct.slash_at_the_end(path_sct_testing, 1)
        self.function_to_test = None
        # self.function_to_avoid = None
        self.remove_tmp_file = 0
        self.verbose = 1
        self.url_git = 'https://github.com/neuropoly/sct_testing_data.git'
        self.path_tmp = ""


# START MAIN
# ==========================================================================================
def main():
    path_data = param.path_data
    function_to_test = param.function_to_test
    # function_to_avoid = param.function_to_avoid
    remove_tmp_file = param.remove_tmp_file

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'h:d:p:f:r:a:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit(0)
        if opt == '-d':
            param.download = int(arg)
        if opt == '-p':
            param.path_data = arg
        if opt == '-f':
            function_to_test = arg
        # if opt == '-a':
        #     function_to_avoid = arg
        if opt == '-r':
            remove_tmp_file = int(arg)

    start_time = time.time()

    # if function_to_avoid:
    #     try:
    #         functions.remove(function_to_avoid)
    #     except ValueError:
    #         print 'The function you want to avoid does not figure in the functions to test list'

    # download data
    if param.download:
        downloaddata()

    param.path_data = 'sct_testing_data/data'
    # get absolute path and add slash at the end
    param.path_data = sct.slash_at_the_end(os.path.abspath(param.path_data), 1)

    # check existence of testing data folder
    if not sct.return_folder_exist(param.path_data):
        downloaddata()

    # display path to data
    sct.printv('\nCheck path to testing data: \n\tOK... '+param.path_data, param.verbose)

    # create temp folder that will have all results and go in it
    param.path_tmp = sct.slash_at_the_end('tmp.'+time.strftime("%y%m%d%H%M%S"), 1)
    sct.create_folder(param.path_tmp)
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
    # remove data folder if exist
    if os.path.exists('sct_testing_data'):
        sct.printv('WARNING: sct_testing_data already exists. Removing it...', param.verbose, 'warning')
        sct.run('rm -rf sct_testing_data')
    # clone git repos
    sct.run('git clone '+param.url_git)
    # update path_data field


# list of all functions to test
# ==========================================================================================
def fill_functions():
    functions = []
    functions.append('test_debug')
    functions.append('sct_apply_transfo')
    functions.append('sct_check_atlas_integrity')
    functions.append('sct_compute_mtr')
    functions.append('sct_concat_transfo')
    #functions.append('sct_convert')
    #functions.append('sct_convert_binary_to_trilinear')  # not useful
    functions.append('sct_create_mask')
    functions.append('sct_crop_image')
    functions.append('sct_dmri_get_bvalue')
    functions.append('sct_dmri_transpose_bvecs')
    functions.append('sct_dmri_moco')
    functions.append('sct_dmri_separate_b0_and_dwi')
    functions.append('sct_extract_metric')
    functions.append('sct_flatten_sagittal')
    functions.append('sct_fmri_compute_tsnr')
    functions.append('sct_fmri_moco')
    functions.append('sct_get_centerline_automatic')
    functions.append('sct_get_centerline_from_labels')
    functions.append('sct_label_utils')
    functions.append('sct_orientation')
    functions.append('sct_process_segmentation')
    functions.append('sct_propseg')
    functions.append('sct_register_multimodal')
    functions.append('sct_register_to_template')
    functions.append('sct_resample')
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
    else: return ''


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

    '''
    status, output = sct.run('echo $SCT_DIR', 0)
    path_logs_dir = output + '/testing/logs'

    if not os.path.isdir(path_logs_dir):
        os.makedirs(path_logs_dir)
    '''

    string = "test ran at "+time.strftime("%y%m%d%H%M%S")+"\n" \
             + fname_log \
             + string

    f = open('../' + fname_log, mode)
    f.write(string+'\n')
    f.close()


# test function
# ==========================================================================================
def test_function(script_name):
    if script_name == 'test_debug':
        return test_debug()  # JULIEN
    else:
        # Using the retest variable to recheck if we can perform tests after we downloaded the data
        retest = 1
        # while condition values are arbitrary and are present to prevent infinite loop
        while 0 < retest < 3:
            # build script name
            fname_log = script_name + ".log"
            tmp_script_name = script_name
            result_folder = "results_"+script_name
            script_name = "test_"+script_name

            if retest == 1:
                # create folder and go in it
                sct.create_folder(result_folder)

            os.chdir(result_folder)

            # display script name
            print_line('Checking '+script_name)
            # import function as a module
            script_tested = importlib.import_module(script_name)
            # test function
            status, output = script_tested.test(param.path_data)
            # returning script_name to its original name
            script_name = tmp_script_name
            # manage status
            if status == 0:
                print_ok()
                retest = 0
            else:
                print_fail()
                print output
                print "\nTest files missing, downloading them now \n"

                os.chdir('../..')
                downloaddata()

                param.path_data = sct.slash_at_the_end(os.path.abspath(param.path_data), 1)

                # check existence of testing data folder
                sct.check_folder_exist(param.path_data)
                os.chdir(param.path_tmp + result_folder)

                retest += 1

            # log file
            write_to_log_file(fname_log, output, 'w')

            # go back to parent folder
            os.chdir('..')
            # end while loop

        # return
        return status


# def old_test_function(folder_test):
#     fname_log = folder_test + ".log"
#     print_line('Checking '+folder_test)
#     os.chdir(folder_test)
#     status, output = commands.getstatusoutput('./test_'+folder_test+'.sh')
#     if status == 0:
#         print_ok()
#     else:
#         print_fail()
#     shutil.rmtree('./results')
#     os.chdir('../')
#     write_to_log_file(fname_log,output)
#     return status


def test_debug():
    print_line ('Checking if debug mode is on .......................')
    path_sct_testing = path_sct + '/testing'
    path_sct_scripts = path_sct + '/scripts'
    debug = []
    files = [f for f in listdir(path_sct_scripts) if isfile(join(path_sct_scripts, f))]
    #print(files)
    for file in files:
        file_fname, ext_fname = os.path.splitext(file)
        if ext_fname == '.py':
            cmd = 'python ' + path_sct_testing + '/test_debug_off.py -i ' + file_fname
            status, output = commands.getstatusoutput(cmd)
            if status != 0:
                print cmd
                debug.append(output)
    if debug == []:
        print_ok()
        return 0  # JULIEN
    else:
        print_fail()
        for string in debug: print string
        return 1  # JULIEN


# Print usage
# ==========================================================================================
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
  Crash test for functions of the Spinal Cord Toolbox.

USAGE
  python """+os.path.basename(__file__)+"""

OPTIONAL ARGUMENTS
  -f <script_name>      test this specific script (do not add extension).
  -d {0,1}              download testing data. Default="""+str(param.download)+"""
  -p <path_data>        path to testing data. Default="""+str(param.path_data)+"""
                        NB: no need to set if using "-d 1"
  -r {0,1}              remove temp files. Default="""+str(param.remove_tmp_file)+"""
  -h                    help. Show this message

EXAMPLE
  python """+os.path.basename(__file__)+""" \n"""

    # exit program
    sys.exit(2)


if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()