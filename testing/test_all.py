#!/usr/bin/env python
#
# Launch all testing scripts.
#
# Author: Julien Cohen-Adad, Benjamin De Leener
# Last Modif: 2014-06-11


import os
import getopt
import sys
from numpy import loadtxt
import commands
# get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
import sct_utils as sct


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
    os.chdir('../')
    write_to_log_file(fname_log,output)
    return status

    
# START MAIN
# ==========================================================================================

status = []
status.append( test_function('sct_segmentation_propagation',' .............. ') )
status.append( test_function('sct_register_to_template',' .................. ') )
status.append( test_function('sct_register_multimodal',' ................... ') )
status.append( test_function('sct_warp_atlas2metric',' ..................... ') )
status.append( test_function('sct_estimate_MAP_tracts',' ................... ') )
status.append( test_function('sct_dmri_moco',' ............................. ') )

print str(status)

print "done!\n"
