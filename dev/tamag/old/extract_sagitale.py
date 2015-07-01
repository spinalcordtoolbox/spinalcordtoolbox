#!/usr/bin/env python


import commands, sys, os


# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')

import nibabel
#from Pil import Image
from scipy.misc import imsave
from sct_utils import extract_fname

os.chdir('/Users/tamag/data/original_data')

dirpath = [x[0] for x in os.walk('C1-T3')]
dirnames = [x[1] for x in os.walk('C1-T3')]
filenames = [x[2] for x in os.walk('C1-T3')]

L = len(dirpath)
dirpath_simplified = []
for i in range(0, L):
    if dirpath[i].find('t2star') == -1:
        dirpath_simplified.append(dirpath[i])



for i in range(3, L, 4):
    files = os.listdir(dirpath[i])
    os.chdir(dirpath[i])
    for file in files:
        if file == 'data_RPI_straight.nii.gz' or file == 'data_RPI_straight_normalized.nii.gz':
            path, name, ext = extract_fname(file)
            uploaded_file = nibabel.load(file)
            data = uploaded_file.get_data()
            middle_sag = round(data.shape[1]/2)
            #Select slice of interest
            data_sagitale = data[:, middle_sag, :]

            #Save data into image jpg (scipy)
            imsave('sagital_slice_' + name + '.png', data_sagitale)
    #Get back to parent folder
    os.chdir('../../..')







