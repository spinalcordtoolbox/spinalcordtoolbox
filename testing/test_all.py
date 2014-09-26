#!/usr/bin/env python
#
# Test major functions.
#
# Authors: Julien Cohen-Adad, Benjamin De Leener, Augustin Roux
# Updated: 2014-09-26


import os
import shutil
import getopt
import sys
import time
from numpy import loadtxt
import commands
# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
import sct_utils as sct
from os import listdir
from os.path import isfile, join


# define nice colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


class param:
    def __init__(self):
        self.download = 0
        self.path_data = 0
        self.function_to_test = None
        self.function_to_avoid = None
        self.remove_tmp_file = 0


# START MAIN
# ==========================================================================================
def main():
    path_data = param.path_data
    function_to_test = param.function_to_test
    function_to_avoid = param.function_to_avoid
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
            param.download = arg
        if opt == '-p':
            path_data = arg
        if opt == '-f':
            function_to_test = arg
        if opt == '-a':
            function_to_avoid = arg
        if opt == '-r':
            remove_tmp_file = arg

    functions = fill_functions()
    start_time = time.time()

    if function_to_avoid:
        try:
            functions.remove(function_to_avoid)
        except ValueError:
            print 'The function you want to avoid does not figure in the functions to test list'

    status = []
    [status.append(test_function(f, download)) for f in functions if function_to_test == f]

    if not status:
        for f in functions:
            status.append(test_function(f, download))

    print 'status: '+str(status)

    elapsed_time = time.time() - start_time
    print 'Finished! Elapsed time: '+str(int(round(elapsed_time)))+'s\n'

    e = 0
    if sum(status) != 0:
        e = 1

    sys.exit(e)


# Print without new carriage return
# ==========================================================================================
def fill_functions():
    functions = []
    functions.append('test_debug')
    functions.append('sct_convert_binary_to_trilinear')
    functions.append('sct_detect_spinalcord')
    functions.append('sct_dmri_moco')
    functions.append('sct_dmri_separate_b0_and_dwi')
    functions.append('sct_extract_metric')
    functions.append('sct_get_centerline')
    functions.append('sct_process_segmentation')
    functions.append('sct_propseg')
    functions.append('sct_register_multimodal')
    functions.append('sct_register_to_template')
    functions.append('sct_smooth_spinalcord')
    functions.append('sct_straighten_spinalcord')
    functions.append('sct_warp_template')
    return functions


def print_line(string):
    import sys
    sys.stdout.write(string + make_dot_lines(string))
    sys.stdout.flush()


def make_dot_lines(string):
    if len(string) < 52:
        dot_lines = '.'*(52 - len(string))
        return dot_lines
    else: return ''


def print_ok():
    print "[" + bcolors.OKGREEN + "OK" + bcolors.ENDC + "]"


def print_warning():
    print "[" + bcolors.WARNING + "WARNING" + bcolors.ENDC + "]"


def print_fail():
    print "[" + bcolors.FAIL + "FAIL" + bcolors.ENDC + "]"


def write_to_log_file(fname_log, string):
    f = open(fname_log, 'w')
    f.write(string+'\n')
    f.close()


def test_function(script_tested, d):
    if script_tested == 'test_debug':
        sys.exit(test_debug())

    if param.download:
        sct.run('git clone https://github.com/neuropoly/sct_testing_data.git')
        os.chdir('sct_testing_data')

    import script_tested

    script_tested.test()



def old_test_function(folder_test):
    fname_log = folder_test + ".log"
    print_line('Checking '+folder_test)
    os.chdir(folder_test)
    status, output = commands.getstatusoutput('./test_'+folder_test+'.sh')
    if status == 0:
        print_ok()
    else:
        print_fail()
    shutil.rmtree('./results')
    os.chdir('../')
    write_to_log_file(fname_log,output)
    return status


def test_debug():
    print_line ('Test if debug mode is on ........................... ')
    debug = []
    files = [f for f in listdir('../scripts') if isfile(join('../scripts',f))]
    for file in files:
        #print (file)
        file_fname, ext_fname = os.path.splitext(file)
        if ext_fname == '.py':
            status, output = commands.getstatusoutput('python ../scripts/test_debug_off.py -i '+file_fname)
            if status != 0:
                debug.append(output)
    if debug == []:
        print_ok()
    else:
        print_fail()
        for string in debug: print string


# Print usage
# ==========================================================================================
def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n\n' \
        'Test following majors scripts of the Toolbox:\n\n' \
        '   test_debug\n' \
        '   sct_convert_binary_to_trilinear\n' \
        '   sct_detect_spinalcord\n' \
        '   sct_dmri_moco\n' \
        '   sct_dmri_separate_b0_and_dwi\n' \
        '   sct_extract_metric\n' \
        '   sct_get_centerline\n' \
        '   sct_process_segmentation\n' \
        '   sct_propseg\n' \
        '   sct_register_multimodal\n' \
        '   sct_register_to_template\n' \
        '   sct_smooth_spinalcord\n' \
        '   sct_straighten_spinalcord\n' \
        '   sct_warp_template\n' \
        'OPTIONAL ARGUMENTS:\n' \
        '   -h                      help, show this message' \
        '   -f <function>           Only test <function>\n' \
        '   -a <function>           Avoid the <function> test\n'


if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()