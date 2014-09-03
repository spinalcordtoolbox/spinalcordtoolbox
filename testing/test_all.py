#!/usr/bin/env python
#
# Test major functions.
#
# Authors: Julien Cohen-Adad, Benjamin De Leener
# Updated: 2014-08-12


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


# Print without new carriage return
# ==========================================================================================
def print_line(string):
    import sys
    sys.stdout.write(string)
    sys.stdout.flush()


def print_ok():
    print "[" + bcolors.OKGREEN + "OK" + bcolors.ENDC + "]"


def print_warning():
    print "[" + bcolors.WARNING + "WARNING" + bcolors.ENDC + "]"


def print_fail():
    print "[" + bcolors.FAIL + "FAIL" + bcolors.ENDC + "]"


def write_to_log_file(fname_log,string):
    f = open(fname_log, 'w')
    f.write(string+'\n')
    f.close()


def test_function(folder_test,dot_lines):
    fname_log = folder_test + ".log" 
    print_line('Checking '+folder_test+dot_lines)
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

    
# START MAIN
# ==========================================================================================

start_time = time.time()
print
test_debug()
status = []
status.append( test_function('sct_convert_binary_to_trilinear',' ........... ') )
status.append( test_function('sct_detect_spinalcord',' ..................... ') )
status.append( test_function('sct_dmri_moco',' ............................. ') )
status.append( test_function('sct_dmri_separate_b0_and_dwi',' .............. ') )
status.append( test_function('sct_extract_metric',' ........................ ') )
status.append( test_function('sct_get_centerline',' ........................ ') )
status.append( test_function('sct_process_segmentation',' .................. ') )
status.append( test_function('sct_register_multimodal',' ................... ') )
status.append( test_function('sct_register_to_template',' .................. ') )
status.append( test_function('sct_segmentation_propagation',' .............. ') )
status.append( test_function('sct_smooth_spinalcord',' ..................... ') )
status.append( test_function('sct_straighten_spinalcord',' ................. ') )
status.append( test_function('sct_warp_template',' ......................... ') )


print 'status: '+str(status)
elapsed_time = time.time() - start_time
print 'Finished! Elapsed time: '+str(int(round(elapsed_time)))+'s\n'

sys.exit(sum(status))
