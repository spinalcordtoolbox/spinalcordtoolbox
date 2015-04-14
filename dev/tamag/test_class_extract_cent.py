#!/usr/bin/env python


import commands, sys
import os

# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
sys.path.append('/home/tamag/code')

from msct_get_centerline_from_labels import ExtractCenterline
import sct_utils as sct
from msct_image import Image
import nibabel

test_class = ExtractCenterline()

os.chdir("/Volumes/Usagers/Etudiants/tamag/data/template/errsm_35/t2/experiment")

#add seg
test_class.addfiles("/Volumes/Usagers/Etudiants/tamag/data/template/errsm_35/t2/experiment/data_RPI_seg_crop.nii.gz")
file_name = nibabel.load("data_RPI_seg_crop.nii.gz")
print type(file_name)
#add label file
#test_class.addfiles("/Volumes/Usagers/Etudiants/tamag/data/template/errsm_35/t2/experiment/labels_brainstem_completed_new.nii.gz")
test_class.addfiles("/Volumes/Usagers/Etudiants/tamag/data/template/errsm_35/t2/experiment/labels_brainstem.nii.gz")

#get centerline smooth
#test_class.getCenterline(type='', output_file_name = "generated_centerline_new_labels.nii.gz", verbose=1)


#image_output= test_class.compute()
#image_output.save()

#test_class.writeCenterline("test_writeCenterline.txt")



