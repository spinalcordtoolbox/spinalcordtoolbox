#!/usr/bin/env python

import sys
import nibabel as nib
import os
import numpy as np
import getopt
import sct_utils as sct

# DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1


#main
# ======================================================================================================================
def main():

    #init
    input_name = ''
    output_name = 'converted_file'
    verbose = param.verbose

    # Check input parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:o:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt == '-i':
            input_name = arg
        elif opt == '-o':
            output_name = arg

    if input_name == '':
        usage()

    # Check file existence
    sct.printv('\nCheck file existence...', verbose)
    sct.check_file_exist(input_name)

    # Load input file
    input_file = nib.load(input_name)
    hdr = input_file.get_header()
    #convert data type
    data = input_file.get_data()



    #hdr.set_data_dtype('float32') # set imagetype to uint8

    data_out = data.astype(np.int8, copy=False)
    img = nib.Nifti1Image(data_out, None)
    #img = nib.Nifti1Image(data, None, hdr)
    nib.save(img, output_name)
    print '.. File created:' + output_name




 #       data_seg = data_seg.astype(np.float32, copy =False)
 #       img = nibabel.Nifti1Image(data_seg, None, hdr_seg)
 #       file_name = path_tmp+'/'+file_data_seg+'_CSA_slices_rpi'+ext_data_seg
 #       nibabel.save(img,file_name)
 #       print '.. File created:' + file_name


# Print usage
# ======================================================================================================================
def usage():
    print """
    """+os.path.basename(__file__)+"""
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>
    DESCRIPTION
      This function performs various types of processing from the spinal cord segmentation:

    USAGE
      """+os.path.basename(__file__)+"""  -i <segmentation> -p <process>
    """
    # exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()