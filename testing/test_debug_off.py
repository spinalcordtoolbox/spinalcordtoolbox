#!/usr/bin/env python
#########################################################################################
#
# Module containing several useful functions.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Augustin Roux
# Modified: 2014-08-29
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys, io, os, getopt, importlib

# get path of SCT
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(path_sct, 'scripts'))

import sct_utils



def main(script_name = ''):
    script_name = ''

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hi:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in '-i':
            script_name = arg

    try:
        script_tested = importlib.import_module(script_name)
        print script_tested
    except IOError:
        sct_utils.printv("\nException caught: IOerror, can not import "+script_name+'\n', 1, 'warning')
        sys.exit(2)
    except ImportError, e:
        sct_utils.printv("\nException caught: ImportError in "+script_name+'\n', 1, 'warning')
        print e
        sys.exit(2)
    else:
        try:
            sct = script_tested.Param()
        except AttributeError:
        #except IOError:
            print ('\nno class param found in script '+script_name+'\n')
            sys.exit(0)
        else:
            if hasattr(sct, 'debug'):
                if sct.debug == 1:
                    print ('\nWarning debug mode on in script '+script_name+'\n')
                    sys.exit(2)


def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Test if the input python script debug mode is on.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <script>\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -i <script>                  python file you want to test (do not add the .py)\n' \


    # exit program
    sys.exit(2)

if __name__ == "__main__":
    main()
