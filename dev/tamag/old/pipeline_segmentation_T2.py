#!/usr/bin/env python


import commands, sys, os


# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))
sys.path.append(path_sct + '/dev/tamag')

import sct_utils as sct
import nibabel
from glob import glob
import time

path = '/Users/tamag/data/data_for_template/marseille'
os.chdir(path)
list_dir = os.listdir(path)

for i in range(-2, len(list_dir)):
    if os.path.isdir(list_dir[i]):
        list_dir_2 = os.listdir(path+'/'+list_dir[i])
        for j in range(len(list_dir_2)):
            if list_dir_2[j] == 'T2':
                print '\nExtract segmentation from: '+ list_dir[i]+ '/' + list_dir_2[j] + ' ...'
                os.chdir(list_dir[i]+ '/' + list_dir_2[j])

                path_tmp = sct.tmp_create(basename="segmentation_T2")

                # copy files into tmp folder
                sct.printv('\nCopy files into tmp folder...')
                name_anatomy_file = glob('*t2_crop.nii.gz')[0]
                # name_label_down = glob('*down.nii.gz')[0]
                # name_label_up = glob('*up.nii.gz')[0]
                path_anatomy_file = os.path.abspath(name_anatomy_file)
                path_label_down = os.path.abspath('down.nii.gz')
                path_label_up = os.path.abspath('up.nii.gz')
                sct.printv('cp '+path_anatomy_file+' '+path_tmp)
                os.system('cp '+path_anatomy_file+' '+path_tmp)
                sct.printv('cp '+path_label_down+' '+path_tmp)
                os.system('cp '+path_label_down+' '+path_tmp)
                sct.printv('cp '+path_label_up+' '+path_tmp)
                os.system('cp '+path_label_up+' '+path_tmp)

                # Go to temp folder
                os.chdir(path_tmp)

                # Create spline-centerline to guide propseg
                #sct.run("""matlab_batcher.sh sct_get_centerline "ls('*_t2_crop.nii.gz')" """)
                #sct.run("""matlab_batcher.sh sct_get_centerline "'mar_1_t2_crop.nii.gz'" """)
                sct.printv("matlab_batcher.sh sct_get_centerline \"\'" + name_anatomy_file + "\'\"")
                os.system("matlab_batcher.sh sct_get_centerline \"\'" + name_anatomy_file + "\'\"")

                # Extract segmentation using propseg with spline-centerline
                sct.printv('sct_propseg -i *_t2_crop.nii.gz -t t2 -init-centerline *t2_crop_centerline.nii')
                os.system('sct_propseg -i *_t2_crop.nii.gz -t t2 -init-centerline *t2_crop_centerline.nii')

                # Erase 3 top and 3 bottom slices of the segmentation to avoid edge effects
                name_seg = glob('*t2_crop_seg.nii.gz')[0]
                path_seg, file_seg, ext_seg = sct.extract_fname(name_seg)
                image_seg = nibabel.load(name_seg)
                nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(name_seg)
                data_seg = image_seg.get_data()
                hdr_seg = image_seg.get_header()
                   # List slices that contain non zero values
                z_centerline = [iz for iz in range(0, nz, 1) if data_seg[:,:,iz].any() ]

                for k in range(0,3):
                    data_seg[:,:,z_centerline[-1]-k] = 0
                    if z_centerline[0]+k < nz:
                        data_seg[:,:,z_centerline[0]+k] = 0
                img_seg = nibabel.Nifti1Image(data_seg, None, hdr_seg)
                nibabel.save(img_seg, file_seg + '_mod' + ext_seg)

                os.chdir('../../..')

