#!/usr/bin/env python
# change type of template data

import os
import commands
import sys
from shutil import move
# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))
from msct_image import Image
import sct_utils as sct

path_template = '/Users/julien/data/PAM50/template'
folder_PAM50 = 'PAM50/template/'
os.chdir(path_template)
sct.create_folder(folder_PAM50)

for file_template in ['MNI-Poly-AMU_T1.nii.gz', 'MNI-Poly-AMU_T2.nii.gz', 'MNI-Poly-AMU_T2star.nii.gz']:
    im = Image(file_template)
    # remove negative values
    data = im.data
    data[data<0] = 0
    im.data = data
    im.changeType('uint16')
    file_new = file_template.replace('MNI-Poly-AMU', 'PAM50')
    im.setFileName(file_new)
    im.save()
    # move to folder
    move(file_new, os.path.join(folder_PAM50, file_new))