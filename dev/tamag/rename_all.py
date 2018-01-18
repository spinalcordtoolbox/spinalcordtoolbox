#!/usr/bin/env python


import sys, os
from glob import glob
import nibabel

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))
sys.path.append(path_sct + '/dev/tamag')

from numpy import mean, append, isnan, array
import sct_utils as sct
from scipy import ndimage
from msct_image import Image

path = '/Users/tamag/data/data_template/test_new_pipeline/subjects'
path_out = '/Users/tamag/Desktop/bug_straightening/all_data'
os.chdir(path)
#
# # Create txt file to list the straightening accuracy per file
# file_name = 'Results_straightening.txt'
# #file_results = open(file_name, 'w')
# #file_results.close()
#
list_dir = os.listdir(path)

for i in range(1,len(list_dir)):
    if os.path.isdir(list_dir[i]):
        list_dir_2 = os.listdir(path + '/' + list_dir[i])
        for j in range(len(list_dir_2)):
            if list_dir_2[j] == 'T2':
                os.chdir(path + '/' + list_dir[i]+ '/' + list_dir_2[j])
                # sct.run('cp data_RPI_crop_normalized.nii.gz ' + path_out + '/T2/' + list_dir[i] + '.nii.gz')
                sct.run('cp generated_centerline.nii.gz ' + path_out + '/T2/' + list_dir[i] + '_centerline.nii.gz')
            if list_dir_2[j] == 'T1':
                os.chdir(path + '/' + list_dir[i]+ '/' + list_dir_2[j])
                # sct.run('cp data_RPI_crop_normalized.nii.gz ' + path_out + '/T1/' + list_dir[i] + '.nii.gz')
                sct.run('cp generated_centerline.nii.gz ' + path_out + '/T1/' + list_dir[i] + '_centerline.nii.gz')
        os.chdir('../..')
#                 # Going into last tmp folder
#                 # list_dir_3 = os.listdir(path + '/' + list_dir[i] + '/' + list_dir_2[j])
#                 # list_tmp_folder = [file for file in list_dir_3 if file.startswith('tmp')]
#                 # last_tmp_folder_name = list_tmp_folder[-1]
#
#                 # os.chdir(list_dir[i]+ '/' + list_dir_2[j])
#                 # os.rename(list_tmp_folder[-1], 'template_creation')
#                 last_tmp_folder_name = 'template_creation'
#                 #os.chdir(list_dir[i]+ '/' + list_dir_2[j]+'/'+last_tmp_folder_name)
#                 print('\n',list_dir[i]+ '/' + list_dir_2[j],'\n')
#                 os.chdir(list_dir[i]+ '/' + list_dir_2[j])
#
#                 # os.chdir('../..')
#                 #old_name = glob(listdir[i])
#                 #os.rename(listdir[i], 't2_'+listdir[i])
#                 #sct.run('cp *_t2_crop_seg_mod.nii.gz t2_crop_seg_mod_crop.nii.gz')
#
#                 # sct.printv(list_dir[i] +': sct_label_utils -i down.nii.gz -t display-voxel')
#                 # os.system('sct_label_utils -i down.nii.gz -t display-voxel')
#                 # sct.printv(list_dir[i] +': sct_label_utils -i up.nii.gz -t display-voxel')
#                 # os.system('sct_label_utils -i up.nii.gz -t display-voxel')
#
#
#                 # #Create cross
#                 # sct.run('sct_detect_extrema.py -i generated_centerline_reg.nii.gz')
#
#                 # Create folder in /Users/tamag/code/spinalcordtoolbox/data with subject name (=list_dir[i])
#                 # if not os.path.isdir('/Users/tamag/code/spinalcordtoolbox/data/template_subjects' + '/'+list_dir[i]):
#                 #     os.makedirs('/Users/tamag/code/spinalcordtoolbox/data/template_subjects' + '/'+list_dir[i])
#                 # Copy files into new folder (files: labels_updown, centerline_propseg_RPI)
#                 # list_dir_3 = os.listdir(path + '/' + list_dir[i] + '/' + list_dir_2[j])
#                 # for k in range(len(list_dir_3)):
#                 #     if list_dir_3[k] == glob('*_t2_crop_centerline.nii'):
#                 os.system('sct_orientation -i *_t2.nii.gz')
#                 sct.run('cp *_t2.nii.gz /Users/tamag/code/spinalcordtoolbox/data/template_subjects' + '/'+list_dir[i]+'/'+'image_RPI.nii.gz')
#
#                 # sct.run('fslmaths down.nii.gz -add up.nii.gz labels_updown.nii.gz')
#                 # sct.run('cp labels_updown.nii.gz /Users/tamag/code/spinalcordtoolbox/data/template_subjects' + '/'+list_dir[i]+'/labels_updown.nii.gz')
#
#
#
#
#                 #os.system('fslview *_t2_crop_straight_normalized.nii.gz *_t2_crop_straight.nii.gz &')
#
#
#                 # Write results into txt file
#                 # nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension('generated_centerline_reg.nii.gz')
#                 #
#                 # from msct_image import Image
#                 # file_centerline_straight = Image('generated_centerline_reg.nii.gz')
#                 #
#                 # data = nibabel.load('generated_centerline_reg.nii.gz').get_data()
#                 # z_centerline = [iz for iz in range(0, nz, 1) if data[:,:,iz].any() ]
#                 # nz_nonz = len(z_centerline)
#                 #
#                 # if nz_nonz==0 :
#                 #    print '\nERROR: Centerline is empty'
#                 #    sys.exit()
#                 #
#                 # x_centerline = [0 for iz in range(0, nz_nonz, 1)]
#                 # y_centerline = [0 for iz in range(0, nz_nonz, 1)]
#                 # print '\nGet center of mass of the centerline ...'
#                 # for iz in xrange(len(z_centerline)):
#                 #    x_centerline[iz], y_centerline[iz] = ndimage.measurements.center_of_mass(array(data[:,:,z_centerline[iz]]))
#                 #
#                 #
#                 # mean_coord = []
#                 # for iz in range(nz_nonz):
#                 #     mean_coord.append([x_centerline[iz], y_centerline[iz]])
#                 #
#                 # # compute error between the input data and the nurbs
#                 # from math import sqrt
#                 # mse_curve = 0.0
#                 # max_dist = 0.0
#                 # x0 = int(round(file_centerline_straight.data.shape[0]/2.0))
#                 # y0 = int(round(file_centerline_straight.data.shape[1]/2.0))
#                 # count_mean = 0
#                 # mse_curve = 0
#                 # for coord_z in mean_coord:
#                 #     if not isnan(sum(coord_z)):
#                 #         dist = ((x0-coord_z[0])*px)**2 + ((y0-coord_z[1])*py)**2
#                 #         mse_curve += dist
#                 #         dist = sqrt(dist)
#                 #         if dist > max_dist:
#                 #             max_dist = dist
#                 #         count_mean += 1
#                 # mse_curve = mse_curve/float(count_mean)
#                 #
#                 #
#                 # sct.printv(list_dir[i] + ': Maximum x-y error = '+str(round(max_dist,2))+' mm')
#                 # sct.printv(list_dir[i] + ': Accuracy of straightening (MSE) = '+str(round(mse_curve,2))+' mm')
#                 #
#                 # # create a txt file with the centerline
#                 # sct.printv('\nWrite text file...')
#                 # file_results = open('../../../' + file_name, 'a')
#                 # file_results.write(list_dir[i] + ': Maximum x-y error = '+str(round(max_dist,2))+' mm' + '\n' +
#                 # list_dir[i] + ': Accuracy of straightening (MSE) = '+str(round(mse_curve,2))+' mm'+ '\n\n' )
#                 # file_results.close()
#
#                 #os.chdir('../../..')
#                 os.chdir('../..')
#
# PATH = '/Users/tamag/code/spinalcordtoolbox/data/template_subjects'
# list_dir_1 = os.listdir(PATH)
# for i in range(-4,len(list_dir_1)):
#     if os.path.isdir(list_dir_1[i]):
#         os.chdir(PATH+'/'+list_dir_1[i])
#         #list_dir_2 = os.listdir(path + '/' + list_dir[i])
#         #os.system('sct_label_utils -i labels_vertebral.nii.gz -t display-voxel')
#         # os.rename('labels_vertebral.nii.gz', 'labels_vertebral_19.nii.gz')
#         # os.rename('labels_vertebral_20.nii.gz','labels_vertebral.nii.gz')
#         #os.system('cp data_RPI_crop_straight_normalized.nii.gz data_RPI_crop_denoised_straight_normalized.nii.gz')
#         os.rename('centerline_propseg_RPI.nii.gz','centerline_propseg_RPI.nii')
#         os.system('gzip centerline_propseg_RPI.nii')
#
#         os.chdir('../')
#
# path_s = '/Users/tamag/data/data_template/Results_template/labels_vertebral'
# os.chdir(path_s)
# list_d = os.listdir(path_s)
# for i in range(1, len(list_d)):
#     os.system('sct_label_utils -i '+list_d[i]+' -t display-voxel')
#
# path = '/Users/tamag/data/data_template/Results_template/to_add/subjects'
# path_info = '/Users/tamag/data/data_template/info/template_subjects/T1'
# os.chdir(path)
# list_d = os.listdir(path)
# for i in range(0, len(list_d)):
#     if os.path.isdir(path+'/'+list_d[i]):
#         os.chdir(path_info)
#         print list_d[i]
#         sct.run('mkdir '+list_d[i])
#         os.chdir(path_info+'/'+list_d[i])
#         file_txt = open('crop.txt', 'w')
#         file_txt.close()
#         sct.run('cp '+path+'/'+list_d[i]+'/T1/data_RPI.nii.gz data_RPI.nii.gz')
#
#         # # Get info from txt file
#         # print '\nRecover infos from text file'
#         # file_name = 'crop.txt'
#         # file_results = open(file_name, 'r')
#         # ymin_anatomic = None
#         # ymax_anatomic = None
#         # for line in file_results:
#         #     line_list = line.split(',')
#         #     zmin_anatomic = line.split(',')[0]
#         #     zmax_anatomic = line.split(',')[1]
#         #     zmin_seg = line.split(',')[2]
#         #     zmax_seg = line.split(',')[3]
#         #     if len(line_list)==6:
#         #         ymin_anatomic = line.split(',')[4]
#         #         ymax_anatomic = line.split(',')[5]
#         # file_results.close()
#         #
#         # # Crop image
#         # if ymin_anatomic == None and ymax_anatomic == None:
#         #     sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start ' + zmin_anatomic + ' -end ' + zmax_anatomic )
#         # else: sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 1,2 -start ' + ymin_anatomic +','+zmin_anatomic+ ' -end ' + ymax_anatomic+','+zmax_anatomic )
#
#
#
#
#         os.chdir('../..')


# path_i = '/Users/tamag/data/data_template/Results_template/to_add/subjects'
# path = '/Users/tamag/data/data_template/info/template_subjects/T1'
# list_d = os.listdir(path_i)
# os.chdir(path)
# for i in range(0, len(list_d)):
#     if os.path.isdir(list_d[i]):
#         print list_d[i]
#         os.chdir(path+'/'+list_d[i])
#
#
#         # # Get info from txt file
#         # print '\nRecover infos from text file'
#         # file_name = 'crop.txt'
#         # file_results = open(file_name, 'r')
#         # ymin_anatomic = None
#         # ymax_anatomic = None
#         # for line in file_results:
#         #     line_list = line.split(',')
#         #     zmin_anatomic = line.split(',')[0]
#         #     zmax_anatomic = line.split(',')[1]
#         #     if len(line_list)==4:
#         #         ymin_anatomic = line.split(',')[2]
#         #         ymax_anatomic = line.split(',')[3]
#         # file_results.close()
#         #
#         # # Crop image
#         # if ymin_anatomic == None and ymax_anatomic == None:
#         #     sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start ' + zmin_anatomic + ' -end ' + zmax_anatomic )
#         # else: sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 1,2 -start ' + ymin_anatomic +','+zmin_anatomic+ ' -end ' + ymax_anatomic+','+zmax_anatomic )
#
#         os.chdir('..')

# path_i = '/Users/tamag/data/data_template/info/template_subjects/T1'
# list_d = os.listdir(path_i)
# os.chdir(path_i)
# for i in range(0, len(list_d)):
#     if os.path.isdir(path_2+'/'+list_d[i]):
#         print list_d[i]
#         if os.path.isdir(path_2+'/'+list_d[i]):
#             os.chdir(path_2+'/'+list_d[i])
#             if not os.path.isdir(path_i_2+'/'+list_d[i]):
#                 os.makedirs(path_i_2+'/'+list_d[i])
#             sct.run('cp crop.txt '+path_i_2+'/'+list_d[i]+'/crop.txt')
#             sct.run('cp labels_vertebral.nii.gz '+path_i_2+'/'+list_d[i]+'/labels_vertebral.nii.gz')
#             if os.path.isfile('centerline_propseg_RPI.nii.gz'):
#                 sct.run('cp centerline_propseg_RPI.nii.gz '+path_i_2+'/'+list_d[i]+'/centerline_propseg_RPI.nii.gz')
#             if os.path.isfile('labels_updown.nii.gz'):
#                 sct.run('cp labels_updown.nii.gz '+path_i_2+'/'+list_d[i]+'/labels_updown.nii.gz')
#
#             # sct.run('cp labels_vertebral.nii.gz '+path_i+'/'+list_d[i]+'/T2/labels_vertebral.nii.gz')

