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

#path = '/Users/tamag/data/data_template/montreal'
# path = '/Users/tamag/code/spinalcordtoolbox/data/template_subjects'
# os.chdir(path)
#
# # Create txt file to list the straightening accuracy per file
# file_name = 'Results_straightening.txt'
# #file_results = open(file_name, 'w')
# #file_results.close()
#
# list_dir = os.listdir(path)
#
# for i in range(1,len(list_dir)):
#     if os.path.isdir(list_dir[i]):
#         list_dir_2 = os.listdir(path + '/' + list_dir[i])
#         for j in range(len(list_dir_2)):
#             if list_dir_2[j] == 'T2':
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

SUBJECTS_LIST_total = [['errsm_02', '/Volumes/data_shared/montreal_criugm/errsm_02/22-SPINE_T1', '/Volumes/data_shared/montreal_criugm/errsm_02/28-SPINE_T2'],['errsm_04', '/Volumes/data_shared/montreal_criugm/errsm_04/16-SPINE_memprage/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_04/18-SPINE_space'],\
                       ['errsm_05', '/Volumes/data_shared/montreal_criugm/errsm_05/23-SPINE_MEMPRAGE/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_05/24-SPINE_SPACE'],['errsm_09', '/Volumes/data_shared/montreal_criugm/errsm_09/34-SPINE_MEMPRAGE2/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_09/33-SPINE_SPACE'],\
                       ['errsm_10', '/Volumes/data_shared/montreal_criugm/errsm_10/13-SPINE_MEMPRAGE/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_10/20-SPINE_SPACE'], ['errsm_11', '/Volumes/data_shared/montreal_criugm/errsm_11/24-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_11/09-SPINE_T2'],\
                       ['errsm_12', '/Volumes/data_shared/montreal_criugm/errsm_12/19-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_12/18-SPINE_T2'],['errsm_13', '/Volumes/data_shared/montreal_criugm/errsm_13/33-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_13/34-SPINE_T2'],\
                       ['errsm_14', '/Volumes/data_shared/montreal_criugm/errsm_14/5002-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_14/5003-SPINE_T2'], ['errsm_16', '/Volumes/data_shared/montreal_criugm/errsm_16/23-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_16/39-SPINE_T2'],\
                       ['errsm_17', '/Volumes/data_shared/montreal_criugm/errsm_17/41-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_17/42-SPINE_T2'], ['errsm_18', '/Volumes/data_shared/montreal_criugm/errsm_18/36-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_18/33-SPINE_T2'],\
                       ['errsm_21', '/Volumes/data_shared/montreal_criugm/errsm_21/27-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_21/30-SPINE_T2'],['errsm_22', '/Volumes/data_shared/montreal_criugm/errsm_22/29-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_22/25-SPINE_T2'],\
                       ['errsm_23', '/Volumes/data_shared/montreal_criugm/errsm_23/29-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_23/28-SPINE_T2'],['errsm_24', '/Volumes/data_shared/montreal_criugm/errsm_24/20-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_24/24-SPINE_T2'],\
                       ['errsm_25', '/Volumes/data_shared/montreal_criugm/errsm_25/25-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_25/26-SPINE_T2'],['errsm_30', '/Volumes/data_shared/montreal_criugm/errsm_30/51-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_30/50-SPINE_T2'],\
                       ['errsm_31', '/Volumes/data_shared/montreal_criugm/errsm_31/31-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_31/32-SPINE_T2'],['errsm_32', '/Volumes/data_shared/montreal_criugm/errsm_32/16-SPINE_T1/echo_2.09 ', '/Volumes/data_shared/montreal_criugm/errsm_32/19-SPINE_T2'],\
                       ['errsm_33', '/Volumes/data_shared/montreal_criugm/errsm_33/30-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_33/31-SPINE_T2'],['1','/Volumes/data_shared/marseille/ED/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-101','/Volumes/data_shared/marseille/ED/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-65'],\
                       ['ALT','/Volumes/data_shared/marseille/ALT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15','/Volumes/data_shared/marseille/ALT/01_0100_space-composing'],['JD','/Volumes/data_shared/marseille/JD/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/JD/01_0100_compo-space'],\
                       ['JW','/Volumes/data_shared/marseille/JW/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/JW/01_0100_compo-space'],['MLL','/Volumes/data_shared/marseille/MLL_1016/01_0008_sc-mprage-1mm-2palliers-fov384-comp-sp-7','/Volumes/data_shared/marseille/MLL_1016/01_0100_t2-compo'],\
                       ['MT','/Volumes/data_shared/marseille/MT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/MT/01_0100_t2composing'],['TR', '/Volumes/data_shared/marseille/TR_T076/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/TR_T076/01_0016_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-19'],\
                       ['T047','/Volumes/data_shared/marseille/T047/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/T047/01_0100_t2-3d-composing'],['VC','/Volumes/data_shared/marseille/VC/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/VC/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-113'],\
                       ['VG','/Volumes/data_shared/marseille/VG/T1/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15','/Volumes/data_shared/marseille/VG/T2/01_0024_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-11'],['VP','/Volumes/data_shared/marseille/VP/01_0011_sc-mprage-1mm-2palliers-fov384-comp-sp-25','/Volumes/data_shared/marseille/VP/01_0100_space-compo'],\
                       ['pain_pilot_1','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/24-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/25-SPINE'],['pain_pilot_2','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/13-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/30-SPINE_T2'],\
                       ['pain_pilot_4','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/33-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/32-SPINE_T2'],['TM', '/Volumes/data_shared/marseille/TM_T057c/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/TM_T057c/01_0105_t2-composing'],\
                       ['errsm_20', '/Volumes/data_shared/montreal_criugm/errsm_20/12-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_20/34-SPINE_T2'],['pain_pilot_3','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/16-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/31-SPINE_T2'],\
                       ['errsm_34','/Volumes/data_shared/montreal_criugm/errsm_34/41-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/errsm_34/40-SPINE_T2'],['errsm_35','/Volumes/data_shared/montreal_criugm/errsm_35/37-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/errsm_35/38-SPINE_T2'],\
                       ['pain_pilot_7', '/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot7/32-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot7/33-SPINE_T2'], ['errsm_03', '/Volumes/data_shared/montreal_criugm/errsm_03/32-SPINE_all/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_03/38-SPINE_all_space'],\
                       ['FR', '/Volumes/data_shared/marseille/FR_T080/01_0039_sc-mprage-1mm-3palliers-fov384-comp-sp-13', '/Volumes/data_shared/marseille/FR_T080/01_0104_spine2'],['GB', '/Volumes/data_shared/marseille/GB_T083/01_0029_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/GB_T083/01_0033_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7'],\
                       ['T045', '/Volumes/data_shared/marseille/T045/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/T045/01_0101_t2-3d-composing']]


SUBJECTS_LIST = SUBJECTS_LIST_total
path_i = '/Users/tamag/data/data_template/info/template_subjects'
contrast ='T1'
path_result = ''
os.chdir(path_i)

f_name='all_crop_'+contrast+'.txt'
f = open(path_i+'/'+f_name, 'w')
f.write("""##Commands to generate label and txt files\n\npath_results =''\n\nif not os.path.isdir(path_results +'/' + 'T1'):\n\tos.makedirs(path_results +'/' + 'T1')\n""")
f.write("""\nif not os.path.isdir(path_results +'/' + 'T2'):\n\tos.makedirs(path_results +'/' + 'T2')\n""")
f.close()

list_d = os.listdir(path_i+'/'+contrast)
for i in range(0,len(SUBJECTS_LIST)):
    subject = SUBJECTS_LIST[i][0]
# for i in range(0, len(list_d)):
    if os.path.isdir(path_i+'/'+contrast+'/'+subject):
        print '\n'+subject
        # subject = list_d[i]
        list_d_2 = os.listdir(path_i + '/' + contrast + '/' +subject)

        # Get info from txt file
        print '\nRecover infos from text file' + path_i + '/' + contrast + '/' + subject+ '/' + 'crop.txt'
        file_name = 'crop.txt'
        os.chdir(path_i + '/' + contrast + '/' + subject)

        file = open(path_i + '/' + contrast + '/' +subject+ '/' +file_name, 'r')
        ymin_anatomic = None
        ymax_anatomic = None
        for line in file:
            line_list = line.split(',')
            zmin_anatomic = line.split(',')[0]
            zmax_anatomic = line.split(',')[1]
            zmin_seg = line.split(',')[2]
            zmax_seg = line.split(',')[3]
            if len(line_list)==6:
                ymin_anatomic = line.split(',')[4]
                ymax_anatomic = line.split(',')[5]
        file.close()

        os.chdir('../..')
        f = open(path_i+'/'+f_name, 'a')
        f.write('\n\n#Preprocessing for subject ' + subject + '\n')
        f.write("""os.makedirs(path_results + '/"""+contrast+"""/""" + subject+"""')\nos.chdir(path_results + '/"""+contrast+"""/""" + subject+"""')\n""")
        f.write("""sct.run('dcm2nii -o . -r N """+ SUBJECTS_LIST[i][1] + """/*.dcm')\nsct.run('mv *.nii.gz data.nii.gz')\nsct.run('sct_orientation -i data.nii.gz -s RPI')\n""")
        if 'labels_vertebral.nii.gz' in list_d_2:
            f.write("""sct.run('sct_crop_image -i data_RPI.nii.gz -dim 2 -start 2 -end 1 -b 0 -o labels_vertebral.nii.gz')\n""")
            status, output = sct.run('sct_label_utils -i ' + path_i+'/' + contrast+ '/' + subject + '/labels_vertebral.nii.gz -t display-voxel')
            nb = output.find('notation')
            int_nb = nb + 10
            labels = output[int_nb:]
            f.write("""sct.run('sct_label_utils -i labels_vertebral.nii.gz -o labels_vertebral.nii.gz -t create -x """+labels+"""')\n""")
        if ymin_anatomic == None and ymax_anatomic == None:
            f.write("""sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  """ +zmin_anatomic+""" -end """+zmax_anatomic+""" ')\nf_crop = open('crop.txt', 'w')\nf_crop.write('"""+zmin_anatomic+','+zmax_anatomic+','+zmin_seg+','+zmax_seg+"""')\nf_crop.close()\n""")
        else: f.write("""sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 1,2 -start """ +ymin_anatomic +','+zmin_anatomic+ ' -end ' + ymax_anatomic+','+zmax_anatomic+ """')\nf_crop = open('crop.txt', 'w')\nf_crop.write('"""+zmin_anatomic+','+zmax_anatomic+','+zmin_seg+','+zmax_seg+','+ymin_anatomic+','+ymax_anatomic+"""')\nf_crop.close()\n""")

        if 'centerline_propseg_RPI.nii.gz' in list_d_2:
            status_1, output_1 = sct.run('sct_label_utils -i ' + path_i+'/' + contrast+ '/' + subject + '/centerline_propseg_RPI.nii.gz -t display-voxel')
            nb_1 = output_1.find('notation')
            int_nb_1 = nb_1 + 10
            labels_1 = output_1[int_nb_1:]
            f.write("""sct.run('sct_crop_image -i data_RPI_crop.nii.gz -dim 2 -start 2 -end 1 -b 0 -o centerline_propseg_RPI.nii.gz')\nsct.run('sct_label_utils -i centerline_propseg_RPI.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x """+labels_1+"""')\n""")
        if 'labels_updown.nii.gz' in list_d_2:
            status_2, output_2 = sct.run('sct_label_utils -i ' + path_i+'/' + contrast+ '/' + subject + '/labels_updown.nii.gz -t display-voxel')
            nb_2 = output_2.find('notation')
            int_nb_2 = nb_2 + 10
            labels_2 = output_2[int_nb_2:]
            f.write("""sct.run('sct_crop_image -i data_RPI_crop.nii.gz -dim 2 -start 2 -end 1 -b 0 -o labels_updown.nii.gz')\nsct.run('sct_label_utils -i labels_updown.nii.gz -o labels_updown.nii.gz -t create -x """+labels_2+"""')\n""")
        f.write("""os.remove('data.nii.gz')\nos.remove('data_RPI.nii.gz')\nos.remove('data_RPI_crop.nii.gz')\nos.chdir('../..')\n""")

f.close()
