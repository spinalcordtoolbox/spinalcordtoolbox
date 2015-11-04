#!/usr/bin/env python

#########################################################################################
# This batch prepare all data for each subject, before preprocessing.
# N.B.: Specify an output path in variable 'path_results'. E.g.:
# path_results ='/Users/julien/data/test_template'
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Tanguy Magnan
# Modified: 2015-07-01
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import commands, sys, os

# Get path of the toolbox
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')

# create symbolic link to /Volumes/data_shared
# ln -s /Volumes/data_raid/data_shared /Volumes/data_shared 

import sct_utils as sct

##Commands to generate label and txt files for T1 and T2 (first is T1 then T2)

# Specify the output folder (no slash at the end)
path_results = '/Users/benjamindeleener/data/template_preprocessing'

if not os.path.isdir(path_results + '/' + 'T1'):
    os.makedirs(path_results + '/' + 'T1')

if not os.path.isdir(path_results + '/' + 'T2'):
    os.makedirs(path_results + '/' + 'T2')


# folder to dataset
folder_data_errsm = '/Volumes/data_shared/montreal_criugm/errsm'
folder_data_marseille = '/Volumes/data_shared/marseille'
folder_data_pain = '/Volumes/data_shared/montreal_criugm/simon'

# Creation of necessary files for T2 preprocessing

"""
# Preprocessing for subject errsm_36
if not os.path.isdir(path_results + '/T2/errsm_36'):
    os.makedirs(path_results + '/T2/errsm_36')
os.chdir(path_results + '/T2/errsm_36')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_36/31-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 23,140,71,20:22,136,105,19:23,130,137,18:23,125,166,17:23,121,194,16:24,120,224,15:24,121,249,14:24,125,275,13:24,131,300,12:24,140,323,11:24,152,345,10:23,165,365,9:22,175,384,8:21,184,404,7:21,188,422,6:21,190,441,5:21,191,461,4:21,191,481,3:23,184,552,2:25,188,577,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  0 -end 590')
f_crop = open('crop.txt', 'w')
f_crop.write('0,590,75,523')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 19,140,0,1:20,139,17,1:20,137,32,1:21,134,46,1:21,131,61,1:22,129,74,1:22,127,89,1:22,123,103,1:22,122,118,1:22,120,131,1:23,119,144,1:22,117,161,1:23,116,178,1:23,115,191,1:23,115,205,1:24,115,218,1:24,115,233,1:24,116,247,1:24,118,262,1:25,121,275,1:25,124,291,1:25,129,307,1:25,135,323,1:24,143,340,1:24,150,356,1:23,157,370,1:22,164,383,1:22,170,397,1:22,175,411,1:21,179,428,1:21,180,440,1:21,181,453,1:21,181,466,1:21,180,480,1:22,179,492,1:22,178,503,1:21,178,514,1:21,178,525,1:23,179,537,1:25,182,552,1:25,183,566,1:25,184,575,1:25,186,584,1:25,189,590,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 19,140,0,1:19,140,1,1:20,139,15,1:21,136,28,1:21,134,40,1:21,133,51,1:21,130,63,1:22,128,74,1:22,178,524,1:22,178,532,1:23,179,541,1:23,181,548,1:25,183,556,1:25,185,565,1:25,185,574,1:25,186,583,1:25,190,590,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_37
if not os.path.isdir(path_results + '/T2/errsm_37'):
    os.makedirs(path_results + '/T2/errsm_37')
os.chdir(path_results + '/T2/errsm_37')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_37/20-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 21,158,170,20:23,152,198,19:23,147,225,18:23,143,251,17:23,140,274,16:23,138,297,15:23,138,321,14:24,142,343,13:23,146,365,12:23,152,385,11:24,159,403,10:24,166,421,9:24,172,438,8:25,173,451,7:25,174,466,6:26,174,480,5:27,175,495,4:27,177,511,3:29,178,562,2:29,186,584,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 597')
f_crop = open('crop.txt', 'w')
f_crop.write('0,597,0,545')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 26,146,0,1:25,156,10,1:24,159,22,1:22,164,42,1:20,166,62,1:19,164,79,1:19,162,96,1:18,158,121,1:19,154,137,1:20,151,154,1:21,148,173,1:22,145,190,1:23,142,211,1:23,139,231,1:23,137,250,1:23,135,267,1:23,134,286,1:23,134,304,1:23,135,324,1:23,137,342,1:24,140,356,1:23,144,374,1:23,148,389,1:24,153,404,1:24,157,418,1:25,162,433,1:25,165,445,1:25,166,460,1:25,167,476,1:26,168,493,1:27,168,511,1:27,168,524,1:28,167,540,1:28,170,552,1:29,174,560,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 28,169,546,1:29,172,553,1:29,173,558,1:30,178,563,1:29,179,568,1:29,182,576,1:29,182,585,1:30,188,597,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_43
if not os.path.isdir(path_results + '/T2/errsm_43'):
    os.makedirs(path_results + '/T2/errsm_43')
os.chdir(path_results + '/T2/errsm_43')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_43/18-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 19,162,73,20:19,156,108,19:18,151,139,18:18,148,167,17:19,144,193,16:18,142,218,15:19,142,243,14:19,144,267,13:20,149,290,12:21,155,312,11:22,164,334,10:24,174,354,9:25,183,372,8:27,190,390,7:28,193,406,6:31,196,423,5:31,198,440,4:32,199,458,3:31,206,520,2:29,212,543,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 557')
f_crop = open('crop.txt', 'w')
f_crop.write('0,557,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 18,156,0,1:18,157,19,1:18,155,36,1:19,155,52,1:18,153,66,1:18,151,79,1:18,150,93,1:18,148,110,1:18,146,125,1:18,145,140,1:18,142,154,1:18,141,170,1:18,140,182,1:19,139,196,1:19,138,211,1:19,137,223,1:19,137,236,1:19,139,254,1:19,141,269,1:19,143,283,1:20,146,297,1:21,150,311,1:22,155,327,1:23,162,343,1:24,167,356,1:25,172,367,1:26,177,380,1:27,182,394,1:29,185,408,1:30,187,419,1:31,189,433,1:32,191,449,1:33,191,463,1:33,191,478,1:34,192,490,1:33,194,500,1:32,200,513,1:31,204,527,1:30,207,536,1:28,210,547,1:28,216,557,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 18,156,0,1:19,157,15,1:19,156,29,1:19,155,41,1:19,156,52,1:19,153,64,1:18,152,73,1:34,196,503,1:32,201,514,1:31,203,522,1:30,207,533,1:29,207,544,1:29,217,557,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_44
if not os.path.isdir(path_results + '/T2/errsm_44'):
    os.makedirs(path_results + '/T2/errsm_44')
os.chdir(path_results + '/T2/errsm_44')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_44/19-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 17,144,83,20:18,141,113,19:18,139,142,18:18,136,170,17:17,133,195,16:18,131,221,15:19,131,245,14:22,134,269,13:24,140,290,12:25,149,310,11:27,162,328,10:28,175,346,9:28,190,362,8:28,201,378,7:28,209,394,6:28,213,411,5:27,212,429,4:26,211,445,3:24,207,517,2:25,212,540,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 557')
f_crop = open('crop.txt', 'w')
f_crop.write('0,557,17,498')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 18,144,0,1:18,143,12,1:18,140,28,1:18,136,46,1:18,134,64,1:18,132,89,1:19,131,111,1:19,130,131,1:18,129,150,1:17,128,168,1:17,127,189,1:17,125,211,1:18,125,232,1:20,127,254,1:22,131,277,1:24,137,297,1:26,147,315,1:27,160,334,1:28,171,350,1:28,184,367,1:28,197,388,1:28,202,405,1:27,203,424,1:26,202,446,1:25,198,470,1:25,198,491,1:25,201,505,1:25,204,520,1:25,206,534,1:25,210,547,1:25,216,557,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 19,145,0,1:18,142,8,1:18,141,16,1:25,198,499,1:25,201,511,1:25,203,519,1:25,205,528,1:25,207,538,1:25,210,547,1:25,215,557,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_02
if not os.path.isdir(path_results + '/T2/errsm_02'):
    os.makedirs(path_results + '/T2/errsm_02')
os.chdir(path_results + '/T2/errsm_02')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_02/28-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 25,63,169,20:26,58,200,19:25,54,229,18:25,53,256,17:25,52,282,16:25,52,307,15:24,55,333,14:23,58,356,13:24,63,379,12:27,69,400,11:29,76,421,10:29,86,440,9:30,96,458,8:31,101,474,7:31,103,489,6:31,105,505,5:29,107,521,4:26,108,538,3:26,114,594,2:22,122,617,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 100 -end 620')
f_crop = open('crop.txt', 'w')
f_crop.write('100,620,0,485')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 28,66,0,1:27,60,24,1:26,55,45,1:26,52,69,1:27,51,89,1:27,47,112,1:26,46,134,1:26,46,153,1:25,46,176,1:25,47,203,1:24,49,223,1:24,51,240,1:24,54,259,1:26,57,278,1:26,63,298,1:27,69,315,1:29,76,334,1:30,84,354,1:31,90,372,1:31,94,392,1:31,97,413,1:30,98,432,1:29,99,448,1:28,99,461,1:27,100,474,1:26,104,483,1:25,109,493,1:23,112,503,1:24,116,511,1:22,118,520,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 25,107,486,1:25,111,496,1:24,116,509,1:23,121,520,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_04
if not os.path.isdir(path_results + '/T2/errsm_04'):
    os.makedirs(path_results + '/T2/errsm_04')
os.chdir(path_results + '/T2/errsm_04')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_04/18-SPINE_space/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 34,144,77,20:33,140,108,19:34,137,138,18:33,136,166,17:33,135,192,16:33,135,218,15:34,137,244,14:33,143,267,13:34,150,290,12:35,160,311,11:37,169,331,10:38,178,350,9:38,187,368,8:38,192,385,7:37,192,402,6:35,193,418,5:35,195,435,4:35,198,452,3:35,202,508,2:36,208,531,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 73 -end 549')
f_crop = open('crop.txt', 'w')
f_crop.write('73,549,0,414')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 33,133,0,1:33,131,22,1:33,129,42,1:34,128,67,1:34,128,91,1:33,128,114,1:33,128,135,1:33,130,155,1:33,133,176,1:33,138,199,1:33,145,218,1:35,152,236,1:36,161,258,1:36,169,277,1:37,174,292,1:37,179,310,1:37,184,329,1:36,186,348,1:36,187,367,1:36,189,386,1:36,190,406,1:36,193,423,1:35,198,436,1:36,201,449,1:36,206,463,1:36,213,476,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 36,191,415,1:36,194,426,1:36,199,437,1:36,202,448,1:36,205,459,1:36,210,469,1:36,214,476,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_05
if not os.path.isdir(path_results + '/T2/errsm_05'):
    os.makedirs(path_results + '/T2/errsm_05')
os.chdir(path_results + '/T2/errsm_05')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_05/24-SPINE_SPACE/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 24,140,105,20:24,133,135,19:25,126,161,18:25,120,184,17:23,116,206,16:25,112,229,15:24,113,250,14:24,116,272,13:24,121,292,12:25,128,312,11:27,139,329,10:26,151,344,9:28,162,360,8:28,170,375,7:28,176,388,6:28,181,402,5:27,184,416,4:27,186,431,3:24,186,488,2:24,190,514,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 35 -end 528')
f_crop = open('crop.txt', 'w')
f_crop.write('35,528,0,434')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 22,137,0,1:21,135,20,1:21,131,41,1:22,128,60,1:22,125,82,1:23,122,105,1:24,118,125,1:24,113,147,1:24,109,168,1:24,107,187,1:24,107,210,1:24,110,232,1:24,114,254,1:24,121,273,1:26,131,292,1:27,143,313,1:28,155,331,1:29,165,350,1:28,172,370,1:27,175,387,1:26,176,405,1:25,176,423,1:24,179,439,1:23,183,454,1:23,186,467,1:23,187,481,1:22,195,493,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 24,178,435,1:24,181,445,1:23,185,455,1:23,186,466,1:23,187,478,1:23,190,486,1:23,194,493,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_09
if not os.path.isdir(path_results + '/T2/errsm_09'):
    os.makedirs(path_results + '/T2/errsm_09')
os.chdir(path_results + '/T2/errsm_09')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_09/33-SPINE_SPACE/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 22,139,117,20:22,136,144,19:23,134,169,18:23,134,191,17:24,133,213,16:24,134,235,15:25,136,256,14:25,139,276,13:25,144,295,12:26,151,313,11:27,158,331,10:27,167,348,9:26,175,364,8:26,180,381,7:26,182,398,6:27,183,413,5:27,185,430,4:28,188,446,3:29,192,503,2:29,202,525,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  52 -end 540 ')
f_crop = open('crop.txt', 'w')
f_crop.write('52,540,0,432')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 18,136,0,1:18,135,16,1:18,131,31,1:19,130,48,1:20,128,69,1:21,127,89,1:21,126,107,1:22,126,127,1:23,126,149,1:23,126,169,1:24,127,188,1:24,130,208,1:25,133,222,1:25,136,239,1:25,141,255,1:26,146,269,1:27,153,286,1:27,160,301,1:27,165,314,1:27,171,331,1:27,174,347,1:27,177,366,1:27,179,385,1:27,179,402,1:28,179,418,1:28,181,431,1:28,185,440,1:28,190,454,1:28,194,464,1:28,197,476,1:28,206,488,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 28,182,433,1:28,187,443,1:28,190,452,1:28,193,461,1:28,196,468,1:28,201,478,1:28,205,488,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_10
if not os.path.isdir(path_results + '/T2/errsm_10'):
    os.makedirs(path_results + '/T2/errsm_10')
os.chdir(path_results + '/T2/errsm_10')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_10/20-SPINE_SPACE/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 24,176,99,20:24,167,130,19:24,159,161,18:24,156,188,17:24,154,214,16:23,154,238,15:24,158,262,14:24,164,285,13:23,171,307,12:23,180,326,11:23,189,345,10:24,198,364,9:24,207,383,8:27,213,401,7:28,216,418,6:28,218,435,5:28,221,452,4:28,222,470,3:28,222,533,2:28,228,554,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 23 -end 568')
f_crop = open('crop.txt', 'w')
f_crop.write('23,568,0,487')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 32,178,0,1:31,176,19,1:29,171,41,1:26,166,64,1:25,161,87,1:25,156,110,1:24,153,132,1:24,151,151,1:24,149,174,1:23,148,196,1:23,149,217,1:24,151,234,1:24,155,251,1:24,160,271,1:24,167,289,1:24,174,306,1:24,181,322,1:24,188,337,1:24,195,353,1:25,202,370,1:26,206,390,1:27,210,409,1:28,212,428,1:28,213,446,1:28,212,466,1:28,213,484,1:27,217,496,1:27,221,508,1:27,223,520,1:27,226,532,1:27,232,545,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 32,178,0,1:31,176,10,1:30,175,18,1:30,174,26,1:28,214,488,1:28,217,499,1:28,220,510,1:28,223,520,1:28,226,531,1:28,232,545,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_11
if not os.path.isdir(path_results + '/T2/errsm_11'):
    os.makedirs(path_results + '/T2/errsm_11')
os.chdir(path_results + '/T2/errsm_11')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_11/09-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 28,137,90,20:26,130,120,19:22,123,151,18:21,119,179,17:19,117,205,16:19,118,231,15:19,118,257,14:20,121,282,13:20,130,304,12:20,140,326,11:18,154,345,10:18,170,363,9:22,184,378,8:21,193,393,7:22,198,408,6:22,201,424,5:23,202,441,4:22,202,459,3:23,194,522,2:23,199,543,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 17 -end 557')
f_crop = open('crop.txt', 'w')
f_crop.write('17,557,0,486')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 31,137,0,1:30,134,25,1:29,128,47,1:28,126,69,1:27,123,86,1:27,120,108,1:26,117,126,1:24,114,150,1:22,112,171,1:20,111,193,1:19,111,214,1:19,112,234,1:19,114,252,1:19,120,274,1:19,127,294,1:19,136,311,1:18,147,327,1:17,158,343,1:18,169,357,1:18,180,373,1:20,188,389,1:21,192,408,1:22,193,425,1:23,192,444,1:23,191,461,1:23,190,477,1:23,191,491,1:23,192,504,1:23,193,516,1:23,195,527,1:23,200,540,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 23,190,487,1:23,191,501,1:23,192,511,1:23,194,520,1:23,196,529,1:23,200,540,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_12
if not os.path.isdir(path_results + '/T2/errsm_12'):
    os.makedirs(path_results + '/T2/errsm_12')
os.chdir(path_results + '/T2/errsm_12')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_12/18-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 26,136,181,20:28,128,210,19:28,123,237,18:28,120,261,17:28,119,284,16:28,120,306,15:28,123,329,14:27,127,350,13:27,133,370,12:24,140,388,11:24,147,406,10:22,155,423,9:22,159,441,8:22,163,458,7:22,165,473,6:22,168,488,5:22,171,504,4:21,176,521,3:23,191,575,2:23,195,597,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 115 -end 610')
f_crop = open('crop.txt', 'w')
f_crop.write('115,610,0,441')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 23,140,0,1:23,137,15,1:24,132,31,1:25,127,49,1:25,122,68,1:26,118,88,1:27,116,105,1:27,114,125,1:27,113,141,1:27,112,160,1:27,112,180,1:27,114,200,1:27,117,219,1:26,121,239,1:26,126,257,1:25,131,275,1:24,137,290,1:24,142,306,1:23,149,326,1:22,153,340,1:22,156,354,1:22,160,371,1:21,164,388,1:21,169,406,1:22,173,426,1:22,177,442,1:22,182,456,1:23,188,471,1:23,193,484,1:22,198,495,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 22,177,442,1:22,181,452,1:22,186,461,1:22,189,470,1:22,192,480,1:22,195,488,1:22,199,495,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_13
if not os.path.isdir(path_results + '/T2/errsm_13'):
    os.makedirs(path_results + '/T2/errsm_13')
os.chdir(path_results + '/T2/errsm_13')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_13/34-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 36,126,102,20:32,122,135,19:27,120,166,18:24,120,195,17:24,121,222,16:24,123,249,15:24,129,276,14:24,135,301,13:25,142,326,12:26,149,349,11:26,158,372,10:26,166,393,9:24,173,413,8:24,177,432,7:25,180,451,6:26,184,470,5:26,187,488,4:27,187,509,3:27,187,578,2:29,192,603,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 21 -end 618')
f_crop = open('crop.txt', 'w')
f_crop.write('21,618,82,527')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 39,123,0,1:38,122,6,1:38,121,14,1:37,119,24,1:36,118,39,1:36,117,47,1:35,117,56,1:35,116,63,1:35,115,71,1:35,115,77,1:34,115,83,1:34,115,87,1:26,175,520,1:26,176,526,1:26,177,533,1:26,178,539,1:26,181,548,1:26,184,557,1:27,190,570,1:27,190,578,1:28,188,585,1:27,192,592,1:27,191,597,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_14
if not os.path.isdir(path_results + '/T2/errsm_14'):
    os.makedirs(path_results + '/T2/errsm_14')
os.chdir(path_results + '/T2/errsm_14')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_14/5003-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 25,159,102,20:25,150,127,19:25,142,151,18:24,140,174,17:24,139,196,16:23,138,218,15:23,138,238,14:23,139,258,13:23,141,278,12:23,144,296,11:23,147,314,10:23,152,331,9:23,157,348,8:23,161,364,7:24,165,378,6:24,169,392,5:24,171,407,4:23,173,422,3:26,187,477,2:27,195,500,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 42 -end 515')
f_crop = open('crop.txt', 'w')
f_crop.write('42,515,0,420')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 28,160,0,1:28,157,24,1:26,151,45,1:26,146,62,1:25,141,82,1:25,137,98,1:24,134,116,1:24,133,133,1:24,132,148,1:24,131,166,1:23,131,187,1:23,132,205,1:23,134,226,1:23,136,246,1:23,140,268,1:23,145,289,1:23,149,305,1:23,154,326,1:23,158,345,1:23,162,365,1:23,165,381,1:24,167,397,1:25,170,411,1:26,175,422,1:26,180,432,1:27,185,442,1:27,189,452,1:27,194,460,1:28,200,473,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 26,174,421,1:26,180,431,1:26,185,441,1:26,188,450,1:27,194,461,1:27,200,473,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_16
if not os.path.isdir(path_results + '/T2/errsm_16'):
    os.makedirs(path_results + '/T2/errsm_16')
os.chdir(path_results + '/T2/errsm_16')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_16/39-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 28,135,179,20:25,127,206,19:23,120,232,18:21,118,255,17:20,117,279,16:18,117,301,15:18,118,324,14:18,121,345,13:19,126,365,12:21,130,385,11:22,138,403,10:23,145,421,9:23,151,439,8:23,154,456,7:24,156,472,6:24,159,490,5:25,163,506,4:25,168,524,3:26,175,583,2:25,183,605,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 114 -end 618')
f_crop = open('crop.txt', 'w')
f_crop.write('114,618,0,450')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 33,138,0,1:34,136,13,1:33,132,29,1:31,129,48,1:29,124,68,1:26,120,88,1:24,117,106,1:22,114,125,1:19,112,144,1:18,111,167,1:17,111,188,1:17,113,212,1:18,116,231,1:20,121,255,1:21,127,277,1:22,132,295,1:23,137,311,1:23,142,328,1:24,146,345,1:24,150,362,1:24,154,383,1:25,158,399,1:25,160,416,1:25,161,432,1:26,164,448,1:25,167,456,1:25,174,473,1:24,181,492,1:24,187,504,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 25,165,451,1:25,169,460,1:25,172,468,1:25,175,476,1:25,178,484,1:25,181,492,1:25,184,499,1:25,187,504,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_17
if not os.path.isdir(path_results + '/T2/errsm_17'):
    os.makedirs(path_results + '/T2/errsm_17')
os.chdir(path_results + '/T2/errsm_17')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_17/42-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 22,140,156,20:25,134,183,19:26,126,211,18:28,121,236,17:28,119,259,16:28,118,282,15:29,120,305,14:30,123,326,13:31,127,347,12:31,133,367,11:31,141,385,10:32,151,402,9:32,158,419,8:31,163,435,7:31,166,451,6:30,168,467,5:30,170,484,4:29,172,500,3:24,172,558,2:23,181,581,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 86 -end 595')
f_crop = open('crop.txt', 'w')
f_crop.write('86,595,0,454')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 23,146,0,1:23,142,17,1:23,136,38,1:24,132,59,1:24,128,80,1:25,124,99,1:26,121,116,1:27,118,133,1:27,116,154,1:28,114,175,1:29,114,189,1:29,114,206,1:30,116,222,1:30,118,241,1:31,121,254,1:31,124,268,1:31,129,283,1:32,135,298,1:32,142,316,1:32,149,334,1:32,154,351,1:31,159,371,1:30,162,392,1:29,163,412,1:27,163,428,1:26,163,445,1:25,164,458,1:24,171,474,1:23,175,487,1:22,179,497,1:22,186,509,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 25,163,455,1:24,167,466,1:24,171,476,1:24,176,486,1:23,180,496,1:22,187,509,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_18
if not os.path.isdir(path_results + '/T2/errsm_18'):
    os.makedirs(path_results + '/T2/errsm_18')
os.chdir(path_results + '/T2/errsm_18')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_18/33-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 24,151,88,20:24,145,119,19:24,141,150,18:24,140,176,17:23,137,203,16:23,135,230,15:22,135,254,14:24,137,278,13:24,142,301,12:25,148,324,11:26,155,344,10:24,163,363,9:23,171,383,8:22,175,401,7:23,178,417,6:22,182,434,5:23,187,453,4:24,192,472,3:23,199,533,2:24,204,555,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 15 -end 568')
f_crop = open('crop.txt', 'w')
f_crop.write('15,568,0,496')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 23,157,0,1:24,153,14,1:24,145,42,1:24,140,69,1:24,137,92,1:24,135,115,1:24,133,137,1:23,132,153,1:23,130,173,1:23,130,192,1:23,129,212,1:23,129,232,1:23,130,250,1:24,132,273,1:24,137,292,1:24,142,312,1:24,148,329,1:24,153,347,1:23,158,362,1:23,163,378,1:23,168,395,1:23,172,412,1:23,175,427,1:23,178,445,1:24,181,459,1:25,184,473,1:24,187,489,1:25,190,499,1:24,194,510,1:23,196,522,1:23,199,533,1:23,203,543,1:23,207,553,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 24,189,497,1:24,193,505,1:24,197,514,1:24,199,523,1:24,200,531,1:24,202,540,1:24,205,547,1:24,208,553,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_21
if not os.path.isdir(path_results + '/T2/errsm_21'):
    os.makedirs(path_results + '/T2/errsm_21')
os.chdir(path_results + '/T2/errsm_21')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_21/30-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 21,127,121,20:19,121,151,19:19,118,181,18:18,117,207,17:18,118,233,16:18,119,259,15:18,122,284,14:18,127,307,13:18,134,328,12:19,142,349,11:19,152,368,10:22,164,385,9:24,175,403,8:27,181,421,7:28,181,438,6:28,179,454,5:28,176,472,4:27,176,492,3:24,176,551,2:23,175,577,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 48 -end 591')
f_crop = open('crop.txt', 'w')
f_crop.write('48,591,0,485')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 22,127,0,1:21,123,19,1:20,119,40,1:20,117,61,1:20,114,83,1:19,112,104,1:19,111,129,1:19,110,150,1:18,110,174,1:18,111,194,1:17,112,212,1:17,115,233,1:18,119,252,1:18,123,270,1:19,130,290,1:20,138,307,1:21,146,324,1:22,156,342,1:24,162,354,1:25,167,364,1:26,171,377,1:26,172,389,1:27,171,402,1:27,169,421,1:27,168,438,1:27,167,457,1:26,166,472,1:25,167,487,1:24,170,502,1:23,171,515,1:23,173,528,1:22,178,543,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 25,167,486,1:24,172,496,1:24,173,508,1:23,174,519,1:23,174,528,1:22,176,536,1:22,179,543,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_22
if not os.path.isdir(path_results + '/T2/errsm_22'):
    os.makedirs(path_results + '/T2/errsm_22')
os.chdir(path_results + '/T2/errsm_22')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_22/25-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 23,123,140,20:23,121,171,19:23,120,198,18:23,118,224,17:22,116,247,16:22,115,269,15:22,115,292,14:22,117,313,13:23,121,334,12:23,128,354,11:23,135,373,10:23,143,390,9:24,150,407,8:24,150,424,7:24,149,441,6:24,151,458,5:23,153,474,4:21,157,492,3:23,162,547,2:25,171,569,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 65 -end 582')
f_crop = open('crop.txt', 'w')
f_crop.write('65,582,0,464')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 20,122,0,1:20,118,25,1:21,114,54,1:22,112,86,1:23,111,112,1:23,110,133,1:23,110,154,1:23,109,176,1:22,109,194,1:22,109,213,1:22,111,236,1:22,114,257,1:23,118,276,1:23,125,296,1:23,131,313,1:23,136,329,1:23,141,347,1:24,143,362,1:23,143,376,1:23,144,390,1:22,145,407,1:21,148,423,1:20,149,436,1:20,150,450,1:20,152,465,1:21,154,474,1:23,158,483,1:24,161,491,1:24,163,499,1:24,168,508,1:25,174,517,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 20,151,465,1:21,157,475,1:23,159,483,1:23,163,492,1:24,166,499,1:24,169,507,1:25,176,517,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_23
if not os.path.isdir(path_results + '/T2/errsm_23'):
    os.makedirs(path_results + '/T2/errsm_23')
os.chdir(path_results + '/T2/errsm_23')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_23/28-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 17,143,136,20:19,140,165,19:19,138,192,18:19,136,216,17:22,135,239,16:22,134,262,15:23,135,286,14:25,137,307,13:25,141,327,12:22,145,346,11:20,149,365,10:20,154,384,9:20,158,402,8:20,159,419,7:21,157,436,6:21,157,453,5:22,161,470,4:22,166,486,3:21,173,542,2:23,182,563,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 62 -end 582')
f_crop = open('crop.txt', 'w')
f_crop.write('62,582,0,463')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 16,141,0,1:16,138,22,1:16,134,41,1:16,133,66,1:17,132,89,1:18,131,113,1:19,130,136,1:19,129,161,1:20,129,182,1:21,128,201,1:21,129,217,1:22,131,237,1:23,134,258,1:23,137,274,1:22,140,293,1:21,143,310,1:21,147,330,1:20,149,346,1:20,150,363,1:20,151,377,1:21,152,396,1:21,155,413,1:21,157,428,1:21,159,447,1:21,162,461,1:22,166,470,1:22,168,480,1:22,174,492,1:22,180,503,1:22,188,520,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 21,163,464,1:21,168,477,1:22,173,487,1:22,177,495,1:22,182,505,1:23,188,513,1:22,192,520,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_24
if not os.path.isdir(path_results + '/T2/errsm_24'):
    os.makedirs(path_results + '/T2/errsm_24')
os.chdir(path_results + '/T2/errsm_24')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_24/24-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 14,133,106,20:14,131,138,19:14,128,168,18:15,126,196,17:16,124,223,16:16,122,250,15:17,125,274,14:17,129,300,13:20,136,323,12:20,146,347,11:21,157,367,10:21,172,383,9:21,183,399,8:23,186,415,7:23,186,432,6:23,186,448,5:23,187,467,4:23,188,483,3:23,190,541,2:25,195,564,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  26 -end 579 ')
f_crop = open('crop.txt', 'w')
f_crop.write('26,579,0,496')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 15,131,0,1:14,128,21,1:13,126,41,1:12,125,63,1:12,123,83,1:12,122,108,1:13,120,129,1:13,120,144,1:14,119,162,1:15,118,187,1:15,117,214,1:16,118,235,1:16,121,257,1:16,125,277,1:18,131,297,1:19,138,317,1:20,146,333,1:21,156,349,1:21,165,363,1:22,174,379,1:22,178,393,1:22,179,407,1:22,179,424,1:22,180,439,1:22,180,454,1:22,179,471,1:23,179,487,1:23,181,497,1:23,185,509,1:24,189,523,1:24,192,537,1:25,198,553,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 23,181,497,1:24,185,507,1:25,188,517,1:24,190,527,1:25,191,537,1:25,195,546,1:25,198,553,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_25
if not os.path.isdir(path_results + '/T2/errsm_25'):
    os.makedirs(path_results + '/T2/errsm_25')
os.chdir(path_results + '/T2/errsm_25')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_25/26-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 26,132,173,20:24,128,204,19:24,124,232,18:23,122,257,17:23,121,279,16:24,121,302,15:24,124,324,14:24,128,345,13:24,133,365,12:24,138,384,11:24,145,401,10:24,153,419,9:24,162,438,8:24,168,455,7:25,172,469,6:24,178,484,5:23,183,500,4:23,186,517,3:24,197,577,2:27,210,600,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 105 -end 614')
f_crop = open('crop.txt', 'w')
f_crop.write('105,614,0,453')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 29,138,0,1:28,132,28,1:27,125,50,1:26,122,68,1:26,120,86,1:25,117,106,1:24,116,130,1:22,115,149,1:22,115,169,1:22,115,188,1:23,116,209,1:24,119,227,1:24,122,244,1:24,127,265,1:25,132,282,1:25,138,297,1:24,146,316,1:24,153,330,1:23,159,345,1:23,164,362,1:23,171,381,1:23,176,399,1:23,178,411,1:23,180,428,1:24,182,445,1:25,185,456,1:24,190,466,1:26,197,479,1:26,204,493,1:26,214,509,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 25,184,454,1:24,189,463,1:25,194,473,1:25,199,482,1:25,204,491,1:26,210,500,1:26,215,509,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_30
if not os.path.isdir(path_results + '/T2/errsm_30'):
    os.makedirs(path_results + '/T2/errsm_30')
os.chdir(path_results + '/T2/errsm_30')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_30/50-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 15,134,115,20:15,126,145,19:15,121,174,18:17,117,199,17:18,115,224,16:19,114,247,15:19,116,271,14:19,121,294,13:20,127,317,12:20,135,337,11:22,145,357,10:24,157,374,9:26,168,391,8:27,175,407,7:26,178,422,6:26,179,438,5:26,180,456,4:24,181,476,3:22,181,536,2:22,184,561,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 37 -end 576')
f_crop = open('crop.txt', 'w')
f_crop.write('37,576,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 39,135,0,1:39,135,1,1:39,135,2,1:39,135,3,1:39,135,4,1:39,135,5,1:39,134,6,1:39,134,7,1:39,134,8,1:39,134,9,1:39,134,10,1:39,134,11,1:39,134,12,1:39,133,13,1:39,133,14,1:39,133,15,1:39,133,16,1:39,133,17,1:39,133,18,1:39,132,19,1:39,132,20,1:39,132,21,1:39,132,22,1:39,132,23,1:39,132,24,1:39,132,25,1:39,131,26,1:39,131,27,1:39,131,28,1:39,131,29,1:39,131,30,1:39,131,31,1:39,130,32,1:39,130,33,1:39,130,34,1:39,130,35,1:39,130,36,1:39,130,37,1:39,130,38,1:39,129,39,1:39,129,40,1:38,129,41,1:38,129,42,1:38,129,43,1:38,129,44,1:38,128,45,1:38,128,46,1:38,128,47,1:38,128,48,1:38,128,49,1:38,128,50,1:38,128,51,1:38,127,52,1:38,127,53,1:38,127,54,1:38,127,55,1:38,127,56,1:38,127,57,1:38,126,58,1:38,126,59,1:38,126,60,1:38,126,61,1:38,126,62,1:37,126,63,1:37,126,64,1:37,125,65,1:37,125,66,1:37,125,67,1:37,125,68,1:37,125,69,1:37,125,70,1:37,124,71,1:37,124,72,1:37,124,73,1:37,124,74,1:37,124,75,1:37,124,76,1:37,124,77,1:37,123,78,1:37,123,79,1:37,123,80,1:37,123,81,1:37,123,82,1:37,123,83,1:36,122,84,1:36,122,85,1:36,122,86,1:36,122,87,1:36,122,88,1:36,122,89,1:36,122,90,1:36,121,91,1:36,121,92,1:36,121,93,1:36,121,94,1:36,121,95,1:36,121,96,1:36,120,97,1:36,120,98,1:36,120,99,1:36,120,100,1:36,120,101,1:36,120,102,1:36,120,103,1:36,119,104,1:36,119,105,1:36,119,106,1:36,119,107,1:36,119,108,1:36,119,109,1:36,119,110,1:36,118,111,1:36,118,112,1:36,118,113,1:36,118,114,1:36,118,115,1:36,118,116,1:36,118,117,1:36,117,118,1:36,117,119,1:36,117,120,1:36,117,121,1:36,117,122,1:36,117,123,1:36,117,124,1:36,116,125,1:36,116,126,1:36,116,127,1:36,116,128,1:36,116,129,1:36,116,130,1:36,116,131,1:36,115,132,1:36,115,133,1:36,115,134,1:36,115,135,1:36,115,136,1:36,115,137,1:36,115,138,1:36,115,139,1:36,114,140,1:36,114,141,1:36,114,142,1:36,114,143,1:36,114,144,1:36,114,145,1:36,114,146,1:36,114,147,1:36,113,148,1:36,113,149,1:36,113,150,1:36,113,151,1:36,113,152,1:36,113,153,1:36,113,154,1:36,113,155,1:36,113,156,1:36,113,157,1:36,112,158,1:36,112,159,1:36,112,160,1:36,112,161,1:36,112,162,1:36,112,163,1:36,112,164,1:36,112,165,1:36,112,166,1:36,112,167,1:36,112,168,1:36,112,169,1:36,111,170,1:36,111,171,1:36,111,172,1:36,111,173,1:36,111,174,1:36,111,175,1:36,111,176,1:36,111,177,1:36,111,178,1:36,111,179,1:36,111,180,1:36,111,181,1:36,111,182,1:36,111,183,1:36,111,184,1:36,111,185,1:36,111,186,1:36,110,187,1:36,110,188,1:36,110,189,1:36,110,190,1:36,110,191,1:36,110,192,1:36,110,193,1:36,110,194,1:36,110,195,1:36,110,196,1:36,110,197,1:36,110,198,1:36,110,199,1:36,110,200,1:36,110,201,1:36,110,202,1:36,110,203,1:36,110,204,1:36,110,205,1:36,110,206,1:36,110,207,1:36,110,208,1:36,110,209,1:36,110,210,1:35,110,211,1:35,110,212,1:35,110,213,1:35,110,214,1:35,111,215,1:35,111,216,1:35,111,217,1:35,111,218,1:35,111,219,1:35,111,220,1:35,111,221,1:35,111,222,1:35,111,223,1:35,111,224,1:35,111,225,1:35,111,226,1:35,111,227,1:35,111,228,1:35,111,229,1:35,112,230,1:35,112,231,1:35,112,232,1:34,112,233,1:34,112,234,1:34,112,235,1:34,112,236,1:34,112,237,1:34,112,238,1:34,112,239,1:34,113,240,1:34,113,241,1:34,113,242,1:34,113,243,1:34,113,244,1:34,113,245,1:34,113,246,1:34,114,247,1:34,114,248,1:33,114,249,1:33,114,250,1:33,114,251,1:33,114,252,1:33,115,253,1:33,115,254,1:33,115,255,1:33,115,256,1:33,115,257,1:33,116,258,1:33,116,259,1:33,116,260,1:33,116,261,1:33,116,262,1:32,117,263,1:32,117,264,1:32,117,265,1:32,117,266,1:32,118,267,1:32,118,268,1:32,118,269,1:32,118,270,1:32,119,271,1:32,119,272,1:32,119,273,1:32,120,274,1:32,120,275,1:31,120,276,1:31,120,277,1:31,121,278,1:31,121,279,1:31,121,280,1:31,122,281,1:31,122,282,1:31,122,283,1:31,123,284,1:31,123,285,1:31,123,286,1:31,124,287,1:30,124,288,1:30,124,289,1:30,125,290,1:30,125,291,1:30,126,292,1:30,126,293,1:30,126,294,1:30,127,295,1:30,127,296,1:30,128,297,1:30,128,298,1:30,128,299,1:30,129,300,1:29,129,301,1:29,130,302,1:29,130,303,1:29,131,304,1:29,131,305,1:29,131,306,1:29,132,307,1:29,132,308,1:29,133,309,1:29,133,310,1:29,134,311,1:29,134,312,1:29,135,313,1:28,135,314,1:28,136,315,1:28,136,316,1:28,137,317,1:28,137,318,1:28,138,319,1:28,138,320,1:28,139,321,1:28,139,322,1:28,140,323,1:28,141,324,1:28,141,325,1:28,142,326,1:28,142,327,1:28,143,328,1:28,143,329,1:27,144,330,1:27,145,331,1:27,145,332,1:27,146,333,1:27,146,334,1:27,147,335,1:27,147,336,1:27,148,337,1:27,149,338,1:27,149,339,1:27,150,340,1:27,150,341,1:27,151,342,1:27,152,343,1:27,152,344,1:27,153,345,1:27,153,346,1:27,154,347,1:27,155,348,1:26,155,349,1:26,156,350,1:26,156,351,1:26,157,352,1:26,157,353,1:26,158,354,1:26,158,355,1:26,159,356,1:26,160,357,1:26,160,358,1:26,161,359,1:26,161,360,1:26,162,361,1:26,162,362,1:26,163,363,1:26,163,364,1:26,164,365,1:26,164,366,1:26,164,367,1:26,165,368,1:26,165,369,1:26,166,370,1:26,166,371,1:26,167,372,1:26,167,373,1:26,167,374,1:26,168,375,1:26,168,376,1:26,168,377,1:26,169,378,1:26,169,379,1:26,169,380,1:25,170,381,1:25,170,382,1:25,170,383,1:25,170,384,1:25,170,385,1:25,171,386,1:25,171,387,1:25,171,388,1:25,171,389,1:25,171,390,1:25,172,391,1:25,172,392,1:25,172,393,1:25,172,394,1:25,172,395,1:25,172,396,1:25,172,397,1:25,172,398,1:25,172,399,1:25,172,400,1:25,173,401,1:25,173,402,1:25,173,403,1:25,173,404,1:25,173,405,1:25,173,406,1:25,173,407,1:25,173,408,1:25,173,409,1:25,173,410,1:25,173,411,1:25,173,412,1:25,173,413,1:25,173,414,1:25,173,415,1:25,173,416,1:25,173,417,1:25,173,418,1:25,173,419,1:25,173,420,1:25,173,421,1:25,173,422,1:25,173,423,1:25,173,424,1:25,173,425,1:25,173,426,1:25,172,427,1:25,172,428,1:25,172,429,1:26,172,430,1:26,172,431,1:26,172,432,1:26,172,433,1:26,172,434,1:26,172,435,1:26,172,436,1:26,172,437,1:26,172,438,1:26,172,439,1:26,172,440,1:26,172,441,1:26,172,442,1:26,172,443,1:26,172,444,1:26,172,445,1:26,172,446,1:26,172,447,1:26,172,448,1:26,172,449,1:26,173,450,1:26,173,451,1:26,173,452,1:26,173,453,1:26,173,454,1:26,173,455,1:26,173,456,1:27,173,457,1:27,173,458,1:27,173,459,1:27,173,460,1:27,173,461,1:27,173,462,1:27,173,463,1:27,173,464,1:27,173,465,1:27,173,466,1:27,173,467,1:27,173,468,1:27,173,469,1:27,173,470,1:27,173,471,1:27,173,472,1:27,173,473,1:27,173,474,1:27,173,475,1:27,173,476,1:27,174,477,1:27,174,478,1:28,174,479,1:28,174,480,1:28,174,481,1:28,174,482,1:28,174,483,1:28,174,484,1:28,174,485,1:28,174,486,1:28,174,487,1:28,174,488,1:28,174,489,1:28,174,490,1:28,174,491,1:28,174,492,1:28,174,493,1:28,174,494,1:28,174,495,1:28,174,496,1:28,174,497,1:28,174,498,1:28,174,499,1:28,174,500,1:28,174,501,1:28,174,502,1:28,174,503,1:28,174,504,1:28,175,505,1:28,175,506,1:28,175,507,1:28,175,508,1:28,175,509,1:29,175,510,1:29,175,511,1:29,175,512,1:29,175,513,1:29,175,514,1:29,175,515,1:29,175,516,1:29,175,517,1:29,175,518,1:29,175,519,1:29,175,520,1:29,175,521,1:29,175,522,1:29,175,523,1:29,175,524,1:29,175,525,1:29,175,526,1:29,174,527,1:29,174,528,1:28,174,529,1:28,174,530,1:28,174,531,1:28,174,532,1:28,174,533,1:28,174,534,1:28,174,535,1:28,174,536,1:28,174,537,1:28,174,538,1:28,174,539,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 12,135,0,1:11,133,12,1:11,132,17,1:13,129,35,1:13,126,54,1:25,173,409,1:25,173,422,1:25,173,435,1:24,173,447,1:23,173,463,1:23,173,476,1:22,176,490,1:23,180,504,1:22,183,519,1:22,180,531,1:22,184,539,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_31
if not os.path.isdir(path_results + '/T2/errsm_31'):
    os.makedirs(path_results + '/T2/errsm_31')
os.chdir(path_results + '/T2/errsm_31')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_31/32-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 34,139,75,20:32,137,108,19:30,131,140,18:27,128,170,17:27,124,199,16:26,122,228,15:25,123,254,14:25,127,280,13:24,132,305,12:24,139,329,11:24,146,351,10:24,154,373,9:24,160,395,8:26,164,414,7:26,167,433,6:27,171,453,5:28,173,472,4:28,173,492,3:22,166,555,2:22,171,578,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 593')
f_crop = open('crop.txt', 'w')
f_crop.write('0,593,0,535')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 35,135,0,1:35,134,22,1:35,131,43,1:34,128,64,1:33,126,86,1:32,124,110,1:32,123,131,1:31,122,151,1:29,120,170,1:28,118,192,1:27,118,209,1:26,117,233,1:25,119,253,1:25,121,274,1:25,124,294,1:24,129,315,1:24,134,334,1:24,138,349,1:24,143,366,1:24,148,387,1:24,153,405,1:24,157,421,1:24,160,439,1:24,163,459,1:24,163,477,1:24,162,494,1:23,160,509,1:22,159,526,1:22,160,539,1:21,164,553,1:21,166,567,1:21,167,580,1:21,176,593,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 22,160,536,1:22,163,547,1:22,165,558,1:21,166,567,1:21,168,578,1:21,171,586,1:21,175,593,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_32
if not os.path.isdir(path_results + '/T2/errsm_32'):
    os.makedirs(path_results + '/T2/errsm_32')
os.chdir(path_results + '/T2/errsm_32')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_32/19-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 16,125,75,20:17,121,107,19:19,121,137,18:19,120,165,17:20,118,190,16:21,117,216,15:22,118,240,14:22,122,264,13:24,128,286,12:24,136,307,11:24,146,326,10:24,156,345,9:24,166,362,8:25,171,380,7:25,169,396,6:25,168,413,5:24,168,431,4:23,167,451,3:21,162,511,2:20,167,538,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 552')
f_crop = open('crop.txt', 'w')
f_crop.write('0,552,0,493')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 12,127,0,1:13,121,27,1:13,118,51,1:14,116,73,1:15,114,98,1:16,113,120,1:17,112,139,1:18,112,157,1:19,111,180,1:20,111,199,1:21,111,220,1:22,113,244,1:23,117,264,1:23,122,287,1:24,128,303,1:24,135,319,1:24,143,337,1:24,150,353,1:24,156,368,1:24,161,389,1:24,160,409,1:23,160,426,1:23,159,444,1:23,158,462,1:23,156,478,1:22,155,493,1:21,158,505,1:20,159,517,1:20,160,528,1:19,164,539,1:18,171,552,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 22,156,494,1:21,158,503,1:20,161,514,1:20,162,523,1:20,162,531,1:19,165,541,1:18,172,552,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_33
if not os.path.isdir(path_results + '/T2/errsm_33'):
    os.makedirs(path_results + '/T2/errsm_33')
os.chdir(path_results + '/T2/errsm_33')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_33/31-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 19,126,71,20:19,121,101,19:20,119,135,18:21,121,162,17:22,121,189,16:22,121,215,15:22,123,243,14:23,126,267,13:24,133,290,12:25,143,313,11:26,154,331,10:26,166,350,9:26,177,367,8:25,183,385,7:25,186,401,6:25,186,417,5:25,186,435,4:25,185,452,3:25,181,512,2:25,187,536,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 548')
f_crop = open('crop.txt', 'w')
f_crop.write('0,548,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 19,128,0,1:19,122,22,1:19,119,45,1:19,116,67,1:19,114,92,1:20,113,114,1:20,112,134,1:21,113,155,1:21,114,176,1:22,115,197,1:22,115,215,1:22,117,240,1:22,118,243,1:22,120,259,1:22,121,264,1:23,126,283,1:24,132,301,1:25,141,322,1:26,150,339,1:27,158,355,1:27,166,372,1:26,173,392,1:26,175,408,1:26,176,426,1:25,175,447,1:25,173,463,1:25,171,480,1:24,172,494,1:25,177,509,1:23,179,522,1:23,183,536,1:23,188,548,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 25,171,491,1:25,173,499,1:24,175,509,1:24,178,518,1:24,180,529,1:23,184,541,1:23,186,548,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')

"""
# Preprocessing for subject errsm_34
if not os.path.isdir(path_results + '/T2/errsm_34'):
    os.makedirs(path_results + '/T2/errsm_34')
os.chdir(path_results + '/T2/errsm_34')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_34/40-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 19,131,107,20:21,127,139,19:23,125,170,18:25,123,197,17:25,120,223,16:27,117,247,15:27,117,271,14:28,120,295,13:28,124,318,12:28,131,341,11:27,138,363,10:27,151,381,9:27,161,400,8:26,167,416,7:26,170,434,6:24,173,452,5:23,178,471,4:21,183,490,3:20,187,556,2:19,190,581,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 613')
f_crop = open('crop.txt', 'w')
f_crop.write('0,613,0,544')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 16,135,0,1:16,132,17,1:17,127,35,1:17,127,54,1:18,123,72,1:19,121,107,1:20,121,122,1:21,120,138,1:23,118,170,1:24,116,198,1:26,114,223,1:26,112,247,1:27,112,271,1:28,115,296,1:28,119,319,1:28,125,342,1:27,135,366,1:27,144,385,1:27,152,404,1:26,158,420,1:25,162,437,1:24,166,454,1:23,170,470,1:22,172,485,1:21,174,501,1:21,175,519,1:22,178,534,1:21,181,546,1:20,185,557,1:20,187,570,1:19,189,583,1:18,192,600,1:19,192,613,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 21,181,545,1:21,182,548,1:21,184,556,1:20,187,564,1:20,188,574,1:20,188,586,1:20,190,596,1:19,192,608,1:19,192,613,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')
"""

# Preprocessing for subject ALT
if not os.path.isdir(path_results + '/T2/ALT'):
    os.makedirs(path_results + '/T2/ALT')
os.chdir(path_results + '/T2/ALT')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/ALT/01_0100_space-composing/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 15,82,53,20:16,76,84,19:15,71,113,18:16,67,140,17:16,65,165,16:16,62,190,15:18,64,214,14:19,69,238,13:21,77,260,12:21,85,282,11:22,96,301,10:23,108,320,9:26,120,337,8:27,128,355,7:27,133,373,6:29,137,391,5:32,138,410,4:33,137,430,3:33,121,492,2:32,120,516,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 40 -end 533')
f_crop = open('crop.txt', 'w')
f_crop.write('40,533,0,451')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 14,72,0,1:15,66,40,1:15,62,80,1:15,58,120,1:16,57,160,1:18,64,200,1:20,79,240,1:23,100,280,1:25,121,320,1:29,130,360,1:33,126,400,1:34,120,440,1:32,121,470,1:30,121,493,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 33,119,433,1:33,119,434,1:33,120,435,1:33,120,436,1:33,120,437,1:33,120,438,1:33,121,439,1:33,121,440,1:33,121,441,1:33,121,442,1:33,121,443,1:33,121,444,1:33,121,445,1:33,121,446,1:33,121,447,1:33,121,448,1:32,121,449,1:32,121,450,1:32,121,451,1:32,121,452,1:32,121,453,1:32,120,454,1:32,120,455,1:32,120,456,1:32,120,457,1:32,119,458,1:32,119,459,1:32,119,460,1:32,119,461,1:32,119,462,1:31,118,463,1:31,118,464,1:31,118,465,1:31,118,466,1:31,117,467,1:31,117,468,1:31,117,469,1:31,117,470,1:31,117,471,1:31,116,472,1:31,116,473,1:31,116,474,1:31,116,475,1:31,116,476,1:30,116,477,1:30,116,478,1:30,116,479,1:30,116,480,1:30,116,481,1:30,116,482,1:30,116,483,1:30,116,484,1:30,116,485,1:30,116,486,1:30,116,487,1:30,117,488,1:30,117,489,1:30,117,490,1:30,118,491,1:30,118,492,1:30,119,493,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject JD
if not os.path.isdir(path_results + '/T2/JD'):
    os.makedirs(path_results + '/T2/JD')
os.chdir(path_results + '/T2/JD')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/JD/01_0100_compo-space/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 24,82,7,20:23,79,38,19:22,77,70,18:22,77,102,17:23,76,130,16:23,76,160,15:24,78,187,14:24,82,213,13:25,89,241,12:26,96,264,11:27,106,289,10:27,118,311,9:27,129,331,8:27,134,351,7:27,134,371,6:25,133,392,5:24,132,413,4:24,132,435,3:27,117,500,2:30,123,529,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 3 -end 545')
f_crop = open('crop.txt', 'w')
f_crop.write('3,545,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 24,70,0,1:23,69,40,1:22,69,80,1:22,69,120,1:22,70,160,1:23,75,200,1:25,84,240,1:26,97,280,1:27,115,320,1:26,127,360,1:25,126,400,1:23,120,440,1:29,118,500,1:32,123,542,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 23,69,4,1:23,69,13,1:23,69,21,1:23,69,37,1:27,127,368,1:26,127,384,1:25,126,396,1:25,125,410,1:24,122,427,1:24,120,438,1:24,116,451,1:24,113,464,1:25,113,481,1:27,113,491,1:28,118,504,1:29,121,516,1:30,119,528,1:32,125,542,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject JW
if not os.path.isdir(path_results + '/T2/JW'):
    os.makedirs(path_results + '/T2/JW')
os.chdir(path_results + '/T2/JW')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/JW/01_0100_compo-space/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 36,109,29,20:35,105,57,19:32,99,87,18:31,97,116,17:29,96,143,16:27,97,170,15:24,100,196,14:23,104,219,13:26,110,242,12:26,118,263,11:26,127,284,10:27,136,303,9:27,143,323,8:27,146,342,7:27,146,359,6:26,146,376,5:24,145,394,4:24,144,413,3:25,134,475,2:25,131,501,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  20 -end 516 ')
f_crop = open('crop.txt', 'w')
f_crop.write('20,516,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 36,98,0,1:35,95,40,1:33,92,80,1:30,90,120,1:27,92,160,1:25,98,200,1:26,110,240,1:26,124,280,1:27,136,320,1:25,138,360,1:23,135,400,1:24,132,440,1:24,130,480,1:25,131,496,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 36,97,8,1:35,96,23,1:35,95,39,1:26,118,264,1:26,124,282,1:26,131,302,1:27,135,319,1:26,137,337,1:26,138,355,1:25,137,372,1:24,136,389,1:23,133,407,1:24,131,425,1:24,131,443,1:25,132,459,1:25,131,474,1:24,126,486,1:25,131,496,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject MLL
if not os.path.isdir(path_results + '/T2/MLL'):
    os.makedirs(path_results + '/T2/MLL')
os.chdir(path_results + '/T2/MLL')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/MLL_1016/01_0100_t2-compo/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 34,98,85,20:32,93,116,19:31,88,145,18:30,86,170,17:30,84,195,16:28,84,218,15:27,85,242,14:26,87,265,13:26,92,287,12:26,99,308,11:26,107,329,10:26,117,346,9:24,126,363,8:24,131,380,7:23,133,397,6:22,135,415,5:22,134,433,4:22,132,451,3:17,120,507,2:16,125,530,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 14 -end 542')
f_crop = open('crop.txt', 'w')
f_crop.write('14,542,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 35,101,0,1:34,92,40,1:33,85,80,1:32,81,120,1:31,79,160,1:28,78,200,1:27,81,240,1:26,89,280,1:25,104,320,1:24,119,360,1:23,126,400,1:20,119,440,1:17,116,480,1:14,126,528,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 34,100,5,1:34,97,21,1:34,92,39,1:34,89,57,1:33,86,78,1:32,83,96,1:32,81,113,1:20,116,451,1:19,113,463,1:17,114,476,1:16,116,491,1:16,123,504,1:15,122,518,1:14,128,528,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject MT
if not os.path.isdir(path_results + '/T2/MT'):
    os.makedirs(path_results + '/T2/MT')
os.chdir(path_results + '/T2/MT')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/MT/01_0100_t2composing/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 32,72,41,20:29,66,72,19:28,65,102,18:27,63,129,17:28,64,156,16:26,65,182,15:26,69,206,14:26,74,232,13:25,83,254,12:28,93,276,11:28,103,296,10:28,115,314,9:26,126,332,8:26,133,348,7:25,134,366,6:25,133,382,5:25,131,399,4:25,130,417,3:25,114,474,2:27,111,499,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  8 -end 515 ')
f_crop = open('crop.txt', 'w')
f_crop.write('8,515,0,449')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 31,63,0,1:30,60,40,1:29,59,80,1:28,57,120,1:27,59,160,1:26,64,200,1:26,75,240,1:28,92,280,1:28,114,320,1:25,126,360,1:25,123,400,1:25,114,440,1:26,114,470,1:27,115,507,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 31,61,8,1:31,60,27,1:30,59,48,1:29,58,69,1:28,57,90,1:28,57,110,1:28,57,129,1:28,58,146,1:27,59,160,1:27,59,177,1:26,63,194,1:26,67,212,1:25,113,447,1:26,111,463,1:27,113,476,1:27,106,492,1:28,112,507,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject T045
if not os.path.isdir(path_results + '/T2/T045'):
    os.makedirs(path_results + '/T2/T045')
os.chdir(path_results + '/T2/T045')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/T045/01_0101_t2-3d-composing/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 23,87,98,20:23,84,128,19:24,80,156,18:25,77,183,17:26,76,207,16:26,73,230,15:27,74,254,14:27,75,277,13:28,80,300,12:28,87,323,11:28,93,342,10:28,102,363,9:28,109,381,8:27,114,400,7:26,117,418,6:25,118,435,5:24,119,453,4:22,118,472,3:18,118,537,2:18,119,563,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 29 -end 580')
f_crop = open('crop.txt', 'w')
f_crop.write('29,580,0,10')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 23,78,0,1:23,77,22,1:22,77,40,1:22,77,58,1:22,77,74,1:22,76,89,1:23,73,113,1:24,71,133,1:24,70,146,1:25,68,164,1:26,67,182,1:26,66,202,1:27,67,217,1:27,68,236,1:27,70,253,1:27,74,275,1:27,79,292,1:27,84,308,1:27,91,327,1:27,98,346,1:27,105,366,1:26,110,384,1:26,111,403,1:24,112,421,1:22,111,438,1:21,110,455,1:20,110,469,1:19,109,482,1:19,111,493,1:19,113,504,1:18,113,519,1:18,114,533,1:18,117,539,1:18,124,551,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 23,78,0,1:23,78,15,1:22,77,33,1:22,77,53,1:22,76,72,1:22,75,86,1:23,74,106,1:24,72,126,1:25,70,144,1:25,68,161,1:26,67,175,1:26,67,187,1:26,66,199,1:26,111,398,1:25,112,407,1:24,112,420,1:23,111,434,1:21,111,449,1:20,110,467,1:20,109,483,1:19,111,494,1:19,112,504,1:19,113,514,1:19,113,523,1:19,114,533,1:19,118,541,1:19,123,551,1')
# os.remove('data.nii.gz')
# os.remove('data_RPI.nii.gz')
# os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject T047
if not os.path.isdir(path_results + '/T2/T047'):
    os.makedirs(path_results + '/T2/T047')
os.chdir(path_results + '/T2/T047')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/T047/01_0100_t2-3d-composing/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 22,96,74,20:22,91,105,19:22,87,135,18:22,83,161,17:22,81,187,16:22,81,210,15:22,82,235,14:22,86,259,13:22,89,281,12:22,96,302,11:22,102,321,10:22,110,340,9:22,118,358,8:23,123,375,7:24,124,392,6:24,124,410,5:25,123,428,4:25,123,445,3:23,120,513,2:23,123,538,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 6 -end 553')
f_crop = open('crop.txt', 'w')
f_crop.write('6,553,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 23,91,0,1:23,85,40,1:22,81,80,1:22,78,120,1:22,76,160,1:22,75,200,1:22,78,240,1:22,86,280,1:22,98,320,1:23,109,360,1:24,116,400,1:24,115,440,1:23,114,480,1:23,123,520,1:23,124,547,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 24,91,0,1:24,90,17,1:24,85,37,1:24,83,54,1:23,81,73,1:23,80,91,1:23,78,109,1:23,77,126,1:22,77,145,1:23,75,165,1:22,75,189,1:23,76,214,1:22,77,230,1:23,117,498,1:23,121,513,1:23,121,525,1:23,119,537,1:23,124,547,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject pain_pilot_1
if not os.path.isdir(path_results + '/T2/pain_pilot_1'):
    os.makedirs(path_results + '/T2/pain_pilot_1')
os.chdir(path_results + '/T2/pain_pilot_1')
sct.run('dcm2nii -o . -r N '+folder_data_pain+'/d_sp_pain_pilot1/25-SPINE/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 24,157,95,20:24,153,127,19:24,149,157,18:23,146,184,17:24,145,210,16:24,144,235,15:23,145,259,14:23,148,282,13:24,152,305,12:25,159,326,11:25,165,349,10:25,171,368,9:25,174,389,8:25,175,406,7:23,175,422,6:23,175,439,5:23,176,456,4:22,176,472,3:23,170,527,2:22,173,551,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 17 -end 585')
f_crop = open('crop.txt', 'w')
f_crop.write('17,585,101,478')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 21,148,0,1:21,148,10,1:22,148,20,1:23,146,32,1:23,145,42,1:23,145,54,1:23,144,64,1:23,144,75,1:23,143,88,1:23,142,98,1:23,141,107,1:23,162,472,1:24,162,477,1:24,161,487,1:24,162,494,1:23,167,505,1:23,170,512,1:23,173,521,1:23,175,529,1:24,177,539,1:23,182,568,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject pain_pilot_2
if not os.path.isdir(path_results + '/T2/pain_pilot_2'):
    os.makedirs(path_results + '/T2/pain_pilot_2')
os.chdir(path_results + '/T2/pain_pilot_2')
sct.run('dcm2nii -o . -r N '+folder_data_pain+'/d_sp_pain_pilot2/30-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 23,143,53,20:20,139,85,19:20,140,113,18:19,138,140,17:19,137,164,16:19,137,189,15:19,138,214,14:19,139,238,13:19,143,262,12:19,149,284,11:19,155,305,10:21,163,325,9:22,171,345,8:23,175,365,7:22,177,384,6:21,175,402,5:19,176,420,4:16,176,440,3:16,172,502,2:20,182,524,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 562')
f_crop = open('crop.txt', 'w')
f_crop.write('0,562,77,477')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 26,136,0,1:26,136,9,1:26,136,15,1:24,135,24,1:24,136,32,1:23,134,43,1:22,134,53,1:22,133,60,1:22,133,68,1:21,133,74,1:21,133,80,1:15,164,473,1:15,164,477,1:14,165,484,1:14,167,491,1:15,169,499,1:16,175,508,1:18,179,520,1:19,179,528,1:18,187,540,1:18,193,557,1:22,197,562,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject pain_pilot_4
if not os.path.isdir(path_results + '/T2/pain_pilot_4'):
    os.makedirs(path_results + '/T2/pain_pilot_4')
os.chdir(path_results + '/T2/pain_pilot_4')
sct.run('dcm2nii -o . -r N '+folder_data_pain+'/d_sp_pain_pilot4/32-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 18,132,94,20:18,129,125,19:19,127,154,18:20,125,182,17:21,124,209,16:21,123,234,15:21,124,259,14:21,127,283,13:24,132,305,12:28,140,327,11:28,146,348,10:25,155,368,9:24,162,387,8:21,166,405,7:21,168,422,6:22,169,440,5:21,170,458,4:21,171,477,3:21,180,546,2:22,187,567,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 16 -end 585')
f_crop = open('crop.txt', 'w')
f_crop.write('16,585,97,509')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 15,130,0,1:15,129,8,1:16,127,18,1:16,127,25,1:16,125,35,1:17,123,45,1:16,122,53,1:17,122,63,1:17,121,74,1:17,121,82,1:18,120,91,1:18,120,99,1:21,167,499,1:21,169,506,1:21,171,513,1:21,174,522,1:21,180,531,1:22,183,540,1:21,184,548,1:23,188,560,1:21,189,569,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject TM
if not os.path.isdir(path_results + '/T2/TM'):
    os.makedirs(path_results + '/T2/TM')
os.chdir(path_results + '/T2/TM')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/TM_T057c/01_0105_t2-composing/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 32,101,20,20:31,94,56,19:31,88,88,18:30,84,117,17:28,82,145,16:29,80,171,15:28,81,197,14:28,85,222,13:28,91,247,12:29,98,270,11:29,109,291,10:28,122,310,9:28,134,329,8:28,141,347,7:27,143,365,6:27,143,385,5:25,142,403,4:24,139,422,3:22,119,484,2:22,118,511,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 1 -end 532')
f_crop = open('crop.txt', 'w')
f_crop.write('1,532,0,1')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 33,93,0,1:33,91,17,1:33,87,29,1:32,84,48,1:32,84,50,1:31,83,58,1:31,82,65,1:30,81,72,1:30,80,79,1:31,80,83,1:31,79,88,1:30,78,89,1:31,78,93,1:30,78,99,1:29,76,119,1:29,76,123,1:29,75,126,1:28,75,132,1:29,74,151,1:29,74,160,1:29,74,172,1:29,74,182,1:28,75,198,1:28,78,219,1:28,80,224,1:28,80,227,1:28,84,244,1:28,92,267,1:28,94,271,1:28,97,278,1:28,99,282,1:28,109,300,1:28,109,301,1:27,123,326,1:27,126,332,1:27,127,336,1:27,133,353,1:27,134,360,1:26,134,377,1:25,133,398,1:25,132,403,1:24,127,423,1:24,122,439,1:24,118,452,1:23,115,466,1:23,116,477,1:22,117,484,1:22,119,501,1:21,120,531,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_20
if not os.path.isdir(path_results + '/T2/errsm_20'):
    os.makedirs(path_results + '/T2/errsm_20')
os.chdir(path_results + '/T2/errsm_20')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_20/34-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 24,134,118,20:25,124,151,19:25,116,182,18:24,113,211,17:23,111,240,16:22,112,267,15:21,117,296,14:20,125,321,13:20,136,345,12:20,150,364,11:20,164,382,10:21,182,398,9:22,195,414,8:23,203,430,7:24,206,446,6:24,206,463,5:24,202,480,4:23,199,498,3:23,189,562,2:23,195,590,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 39 -end 604')
f_crop = open('crop.txt', 'w')
f_crop.write('39,604,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 26,145,0,1:25,136,32,1:25,128,51,1:25,123,71,1:25,119,95,1:24,113,120,1:24,111,140,1:24,107,163,1:24,106,184,1:23,106,203,1:23,106,220,1:22,108,239,1:22,111,256,1:21,118,280,1:21,127,300,1:21,139,318,1:21,151,337,1:21,168,356,1:21,181,372,1:22,191,387,1:22,196,404,1:23,195,425,1:24,193,442,1:25,188,461,1:25,185,479,1:25,182,498,1:24,183,512,1:24,187,530,1:24,191,541,1:23,195,553,1:23,200,565,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 25,182,499,1:25,183,510,1:24,185,518,1:24,187,527,1:24,189,537,1:24,190,547,1:23,194,555,1:23,200,565,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject pain_pilot_3
if not os.path.isdir(path_results + '/T2/pain_pilot_3'):
    os.makedirs(path_results + '/T2/pain_pilot_3')
os.chdir(path_results + '/T2/pain_pilot_3')
sct.run('dcm2nii -o . -r N '+folder_data_pain+'/d_sp_pain_pilot3/31-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 15,119,95,20:16,118,128,19:16,119,159,18:19,121,186,17:21,122,212,16:22,123,237,15:23,125,261,14:25,129,286,13:26,135,308,12:27,141,330,11:27,149,349,10:27,161,367,9:26,169,385,8:26,176,402,7:25,181,418,6:25,184,434,5:25,189,450,4:26,189,468,3:30,185,527,2:33,191,554,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 18 -end 605')
f_crop = open('crop.txt', 'w')
f_crop.write('18,605,69,379')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 13,118,0,1:13,115,19,1:14,111,36,1:14,110,55,1:15,108,68,1:27,166,380,1:27,172,397,1:27,175,408,1:27,177,419,1:27,179,430,1:27,179,442,1:27,179,453,1:27,177,464,1:27,177,475,1:28,178,488,1:29,180,501,1:31,182,511,1:31,185,519,1:31,187,532,1:33,187,542,1:34,193,553,1:34,197,561,1:35,199,572,1:36,202,580,1:35,204,587,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_35
if not os.path.isdir(path_results + '/T2/errsm_35'):
    os.makedirs(path_results + '/T2/errsm_35')
os.chdir(path_results + '/T2/errsm_35')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_35/38-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 24,128,83,20:22,122,114,19:22,118,144,18:21,115,172,17:20,113,197,16:20,113,223,15:19,114,247,14:18,117,271,13:18,122,293,12:18,127,314,11:19,132,333,10:20,141,351,9:18,150,369,8:17,154,384,7:18,157,399,6:18,159,415,5:19,161,431,4:19,164,448,3:22,172,508,2:23,182,530,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 16 -end 550')
f_crop = open('crop.txt', 'w')
f_crop.write('16,550,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 25,122,0,1:25,116,34,1:22,112,77,1:22,110,116,1:21,107,149,1:19,107,185,1:19,107,214,1:18,111,250,1:18,119,290,1:19,128,323,1:18,138,352,1:18,147,381,1:18,153,418,1:20,154,450,1:21,160,474,1:22,167,489,1:23,178,502,1:24,182,513,1:24,194,534,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 25,123,0,1:24,122,9,1:25,120,18,1:24,118,27,1:24,116,35,1:24,115,42,1:21,159,473,1:22,162,479,1:22,165,485,1:22,168,491,1:22,171,499,1:22,173,506,1:22,176,513,1:22,180,519,1:22,184,526,1:22,188,532,1:22,189,534,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject pain_pilot_7
if not os.path.isdir(path_results + '/T2/pain_pilot_7'):
    os.makedirs(path_results + '/T2/pain_pilot_7')
os.chdir(path_results + '/T2/pain_pilot_7')
sct.run('dcm2nii -o . -r N '+folder_data_pain+'/d_sp_pain_pilot7/33-SPINE_T2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 34,143,82,20:32,135,115,19:31,129,143,18:29,123,169,17:27,118,195,16:26,115,221,15:25,113,246,14:22,116,271,13:21,121,295,12:21,130,317,11:21,140,339,10:19,151,359,9:19,163,379,8:19,171,397,7:20,177,416,6:22,182,434,5:23,187,453,4:25,190,473,3:25,185,535,2:25,191,558,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 14 -end 575')
f_crop = open('crop.txt', 'w')
f_crop.write('14,575,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 35,142,0,1:34,136,34,1:33,129,75,1:32,122,116,1:29,115,157,1:25,111,200,1:23,111,241,1:21,118,279,1:20,134,322,1:19,153,364,1:20,170,406,1:24,180,455,1:26,180,494,1:25,182,516,1:25,189,538,1:25,196,561,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 26,180,504,1:26,182,514,1:26,183,523,1:26,185,532,1:26,186,539,1:26,188,546,1:26,192,553,1:26,195,561,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_03
if not os.path.isdir(path_results + '/T2/errsm_03'):
    os.makedirs(path_results + '/T2/errsm_03')
os.chdir(path_results + '/T2/errsm_03')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_03/38-SPINE_all_space/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 25,144,101,20:26,142,133,19:26,139,162,18:27,136,189,17:28,136,214,16:28,137,239,15:29,139,264,14:30,141,287,13:30,145,310,12:30,151,332,11:30,160,352,10:31,173,370,9:31,186,388,8:30,193,404,7:29,194,419,6:29,192,434,5:29,192,451,4:29,193,468,3:28,194,530,2:28,200,554,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 30 -end 572')
f_crop = open('crop.txt', 'w')
f_crop.write('30,572,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 25,135,0,1:25,132,43,1:25,133,73,1:27,132,103,1:27,131,133,1:28,129,179,1:29,131,218,1:29,134,250,1:30,141,286,1:30,149,309,1:30,163,337,1:30,172,352,1:30,181,372,1:29,185,411,1:29,184,444,1:28,184,473,1:29,192,504,1:27,199,516,1:27,204,531,1:28,207,542,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 28,185,477,1:28,186,484,1:28,189,492,1:27,190,500,1:27,192,510,1:27,193,518,1:27,195,525,1:27,199,531,1:27,203,537,1:27,205,542,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject FR
if not os.path.isdir(path_results + '/T2/FR'):
    os.makedirs(path_results + '/T2/FR')
os.chdir(path_results + '/T2/FR')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/FR_T080/01_0104_spine2/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 30,96,60,20:30,91,95,19:30,86,128,18:29,81,157,17:28,76,186,16:27,73,215,15:25,72,249,14:24,71,275,13:23,72,298,12:23,77,324,11:24,83,347,10:24,94,369,9:23,103,389,8:22,112,409,7:22,115,429,6:21,115,449,5:21,116,471,4:21,116,491,3:22,108,558,2:22,111,584,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 605')
f_crop = open('crop.txt', 'w')
f_crop.write('0,605,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 32,96,0,1:31,91,23,1:30,85,45,1:30,81,69,1:30,78,91,1:30,76,112,1:30,73,130,1:29,71,151,1:28,69,170,1:28,67,189,1:27,66,204,1:26,65,223,1:25,65,240,1:24,64,253,1:24,64,269,1:23,65,284,1:23,67,298,1:23,70,318,1:23,74,333,1:24,79,350,1:23,85,367,1:23,91,382,1:22,97,398,1:22,102,414,1:21,105,427,1:21,106,438,1:21,107,456,1:21,107,470,1:21,106,484,1:21,105,495,1:21,103,508,1:21,101,522,1:22,101,531,1:22,103,546,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 32,96,0,1:31,90,29,1:30,84,50,1:30,80,72,1:30,78,95,1:30,75,117,1:29,72,138,1:28,70,156,1:28,68,177,1:27,66,198,1:26,65,216,1:26,65,234,1:22,102,540,1:22,103,546,1:22,104,554,1:22,105,562,1:22,106,570,1:22,106,579,1:22,108,586,1:22,110,592,1:22,112,599,1:22,114,605,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject GB
if not os.path.isdir(path_results + '/T2/GB'):
    os.makedirs(path_results + '/T2/GB')
os.chdir(path_results + '/T2/GB')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/GB_T083/01_0033_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 31,99,63,20:31,99,95,19:32,96,125,18:32,92,151,17:32,90,176,16:32,88,199,15:32,89,223,14:32,92,245,13:32,97,268,12:32,105,289,11:32,114,309,10:30,125,328,9:29,134,347,8:28,138,364,7:28,140,381,6:28,140,398,5:28,137,415,4:28,135,433,3:26,128,495,2:24,130,522,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 1 -end 539')
f_crop = open('crop.txt', 'w')
f_crop.write('1,539,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 32,88,0,1:32,87,5,1:32,87,18,1:32,86,28,1:32,85,39,1:32,85,45,1:32,84,54,1:32,85,104,1:32,84,137,1:33,82,175,1:32,83,225,1:32,91,266,1:31,106,307,1:29,122,346,1:29,131,387,1:28,126,434,1:27,124,474,1:26,126,492,1:24,129,511,1:22,134,538,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 27,123,478,1:26,125,483,1:26,125,491,1:25,126,499,1:24,126,508,1:24,126,516,1:24,127,523,1:24,129,529,1:24,131,538,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject 1
if not os.path.isdir(path_results + '/T2/ED'):
    os.makedirs(path_results + '/T2/ED')
os.chdir(path_results + '/T2/ED')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/ED/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-65/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 14,88,100,20:15,83,132,19:15,77,160,18:15,75,186,17:16,73,210,16:16,73,235,15:16,75,260,14:18,79,283,13:20,86,305,12:21,93,326,11:22,101,345,10:21,111,362,9:21,119,380,8:21,123,397,7:20,123,414,6:20,124,431,5:20,126,449,4:22,126,466,3:19,118,524,2:19,124,544,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 27 -end 559')
f_crop = open('crop.txt', 'w')
f_crop.write('27,559,0,480')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 8,87,0,1:9,84,22,1:10,81,47,1:11,77,68,1:13,74,89,1:14,73,107,1:14,71,128,1:15,70,150,1:15,68,170,1:16,68,189,1:16,67,207,1:17,69,226,1:17,70,241,1:18,74,258,1:19,78,274,1:20,82,286,1:20,89,305,1:21,95,319,1:21,101,334,1:21,106,347,1:22,111,362,1:22,115,381,1:21,116,397,1:22,116,410,1:22,116,423,1:22,115,439,1:21,113,453,1:21,111,466,1:20,112,478,1:19,115,488,1:19,118,498,1:19,119,507,1:18,121,514,1:18,124,523,1:17,128,532,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 20,112,481,1:20,114,485,1:19,115,492,1:18,118,500,1:18,120,507,1:18,121,515,1:18,123,521,1:18,125,527,1:18,127,532,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject VC
if not os.path.isdir(path_results + '/T2/VC'):
    os.makedirs(path_results + '/T2/VC')
os.chdir(path_results + '/T2/VC')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/VC/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-113/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 25,103,137,20:24,96,169,19:24,90,197,18:24,88,223,17:24,83,247,16:24,78,270,15:24,78,293,14:24,79,315,13:24,86,337,12:24,93,357,11:25,103,376,10:26,113,395,9:24,123,411,8:24,128,429,7:23,131,444,6:23,132,459,5:23,133,474,4:23,132,490,3:25,127,550,2:25,130,575,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 1,2 -start 11,67 -end 245,589')
f_crop = open('crop.txt', 'w')
f_crop.write('67,589,0,max,11,245')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 27,97,0,1:27,96,1,1:27,96,2,1:27,96,3,1:26,96,4,1:26,96,5,1:26,95,6,1:26,95,7,1:26,95,8,1:26,95,9,1:26,94,10,1:26,94,11,1:26,94,12,1:26,94,13,1:26,93,14,1:26,93,15,1:26,93,16,1:26,93,17,1:26,93,18,1:26,92,19,1:26,92,20,1:26,92,21,1:26,92,22,1:26,91,23,1:26,91,24,1:26,91,25,1:26,91,26,1:26,90,27,1:26,90,28,1:25,90,29,1:25,90,30,1:25,90,31,1:25,89,32,1:25,89,33,1:25,89,34,1:25,89,35,1:25,88,36,1:25,88,37,1:25,88,38,1:25,88,39,1:25,87,40,1:25,87,41,1:25,87,42,1:25,87,43,1:25,87,44,1:25,86,45,1:25,86,46,1:25,86,47,1:25,86,48,1:25,85,49,1:25,85,50,1:25,85,51,1:25,85,52,1:25,84,53,1:25,84,54,1:25,84,55,1:25,84,56,1:25,84,57,1:25,83,58,1:25,83,59,1:25,83,60,1:25,83,61,1:25,82,62,1:25,82,63,1:25,82,64,1:25,82,65,1:25,82,66,1:25,81,67,1:25,81,68,1:25,81,69,1:25,81,70,1:25,81,71,1:25,80,72,1:25,80,73,1:25,80,74,1:25,80,75,1:25,79,76,1:25,79,77,1:25,79,78,1:25,79,79,1:25,79,80,1:25,79,81,1:25,78,82,1:25,78,83,1:25,78,84,1:25,78,85,1:25,78,86,1:25,77,87,1:25,77,88,1:25,77,89,1:25,77,90,1:25,77,91,1:25,76,92,1:25,76,93,1:25,76,94,1:25,76,95,1:25,76,96,1:25,76,97,1:25,76,98,1:25,75,99,1:26,75,100,1:26,75,101,1:26,75,102,1:26,75,103,1:26,75,104,1:26,74,105,1:26,74,106,1:26,74,107,1:26,74,108,1:26,74,109,1:26,74,110,1:26,74,111,1:26,73,112,1:26,73,113,1:26,73,114,1:26,73,115,1:26,73,116,1:26,73,117,1:26,73,118,1:26,73,119,1:26,73,120,1:26,72,121,1:26,72,122,1:26,72,123,1:26,72,124,1:26,72,125,1:26,72,126,1:26,72,127,1:26,72,128,1:26,72,129,1:26,71,130,1:26,71,131,1:26,71,132,1:26,71,133,1:26,71,134,1:26,71,135,1:26,71,136,1:26,71,137,1:26,71,138,1:26,70,139,1:26,70,140,1:26,70,141,1:26,70,142,1:26,70,143,1:26,70,144,1:26,70,145,1:26,70,146,1:26,70,147,1:26,69,148,1:26,69,149,1:26,69,150,1:26,69,151,1:26,69,152,1:26,69,153,1:26,69,154,1:26,69,155,1:26,68,156,1:26,68,157,1:26,68,158,1:26,68,159,1:26,68,160,1:26,68,161,1:26,68,162,1:26,67,163,1:26,67,164,1:26,67,165,1:26,67,166,1:26,67,167,1:26,67,168,1:26,67,169,1:26,66,170,1:26,66,171,1:26,66,172,1:26,66,173,1:26,66,174,1:26,66,175,1:26,66,176,1:26,65,177,1:26,65,178,1:26,65,179,1:26,65,180,1:26,65,181,1:26,65,182,1:26,65,183,1:26,64,184,1:26,64,185,1:26,64,186,1:26,64,187,1:26,64,188,1:26,64,189,1:26,64,190,1:26,64,191,1:26,63,192,1:26,63,193,1:26,63,194,1:26,63,195,1:26,63,196,1:26,63,197,1:26,63,198,1:26,63,199,1:26,63,200,1:26,63,201,1:26,62,202,1:26,62,203,1:26,62,204,1:26,62,205,1:26,62,206,1:26,62,207,1:26,62,208,1:26,62,209,1:26,62,210,1:26,62,211,1:26,62,212,1:26,62,213,1:26,62,214,1:26,62,215,1:26,62,216,1:26,62,217,1:26,62,218,1:26,62,219,1:26,62,220,1:26,62,221,1:26,62,222,1:26,62,223,1:26,62,224,1:26,62,225,1:26,62,226,1:26,62,227,1:26,62,228,1:26,62,229,1:26,62,230,1:26,62,231,1:26,62,232,1:26,62,233,1:26,62,234,1:26,62,235,1:26,62,236,1:26,62,237,1:26,62,238,1:26,63,239,1:26,63,240,1:26,63,241,1:26,63,242,1:26,63,243,1:26,63,244,1:26,63,245,1:26,63,246,1:26,64,247,1:26,64,248,1:26,64,249,1:26,64,250,1:26,64,251,1:26,64,252,1:26,65,253,1:26,65,254,1:26,65,255,1:26,65,256,1:26,65,257,1:26,66,258,1:26,66,259,1:26,66,260,1:26,66,261,1:26,67,262,1:26,67,263,1:26,67,264,1:26,67,265,1:26,68,266,1:26,68,267,1:26,68,268,1:26,69,269,1:26,69,270,1:26,69,271,1:26,69,272,1:26,70,273,1:26,70,274,1:26,70,275,1:26,71,276,1:26,71,277,1:26,71,278,1:26,72,279,1:26,72,280,1:26,73,281,1:26,73,282,1:26,73,283,1:26,74,284,1:26,74,285,1:26,74,286,1:26,75,287,1:26,75,288,1:26,76,289,1:26,76,290,1:26,77,291,1:26,77,292,1:26,77,293,1:26,78,294,1:26,78,295,1:26,79,296,1:26,79,297,1:26,80,298,1:26,80,299,1:26,80,300,1:26,81,301,1:26,81,302,1:26,82,303,1:26,82,304,1:26,83,305,1:26,83,306,1:26,84,307,1:26,84,308,1:26,85,309,1:26,85,310,1:26,86,311,1:26,86,312,1:26,87,313,1:26,87,314,1:26,88,315,1:26,88,316,1:26,89,317,1:26,89,318,1:26,90,319,1:26,90,320,1:26,91,321,1:26,91,322,1:26,92,323,1:26,92,324,1:26,93,325,1:26,93,326,1:26,94,327,1:26,95,328,1:26,95,329,1:26,96,330,1:26,96,331,1:26,97,332,1:26,97,333,1:26,98,334,1:26,98,335,1:26,99,336,1:26,99,337,1:26,100,338,1:26,100,339,1:26,100,340,1:26,101,341,1:26,101,342,1:26,102,343,1:26,102,344,1:26,103,345,1:26,103,346,1:26,104,347,1:26,104,348,1:26,104,349,1:26,105,350,1:26,105,351,1:26,106,352,1:26,106,353,1:26,106,354,1:26,107,355,1:26,107,356,1:26,107,357,1:26,108,358,1:26,108,359,1:26,108,360,1:26,109,361,1:26,109,362,1:26,109,363,1:26,109,364,1:26,110,365,1:26,110,366,1:26,110,367,1:26,110,368,1:26,111,369,1:26,111,370,1:26,111,371,1:26,111,372,1:26,111,373,1:26,111,374,1:26,111,375,1:26,112,376,1:26,112,377,1:26,112,378,1:26,112,379,1:26,112,380,1:26,112,381,1:26,112,382,1:26,112,383,1:26,112,384,1:26,112,385,1:26,112,386,1:26,112,387,1:26,112,388,1:26,112,389,1:26,112,390,1:26,112,391,1:27,112,392,1:27,112,393,1:27,112,394,1:27,113,395,1:27,113,396,1:27,113,397,1:27,112,398,1:27,112,399,1:27,112,400,1:27,112,401,1:27,112,402,1:27,112,403,1:27,112,404,1:27,112,405,1:27,112,406,1:27,112,407,1:27,112,408,1:26,112,409,1:26,112,410,1:26,112,411,1:26,112,412,1:26,112,413,1:26,112,414,1:26,112,415,1:26,112,416,1:26,112,417,1:26,112,418,1:26,112,419,1:26,112,420,1:26,113,421,1:26,113,422,1:26,113,423,1:26,113,424,1:26,113,425,1:26,113,426,1:26,113,427,1:26,113,428,1:25,113,429,1:25,113,430,1:25,113,431,1:25,113,432,1:25,113,433,1:25,113,434,1:25,113,435,1:25,113,436,1:25,113,437,1:25,113,438,1:25,113,439,1:25,113,440,1:25,114,441,1:25,114,442,1:24,114,443,1:24,114,444,1:24,114,445,1:24,114,446,1:24,114,447,1:24,114,448,1:24,114,449,1:24,114,450,1:24,114,451,1:24,114,452,1:24,114,453,1:24,114,454,1:24,114,455,1:24,114,456,1:24,114,457,1:24,114,458,1:24,114,459,1:24,114,460,1:24,114,461,1:24,114,462,1:24,114,463,1:25,114,464,1:25,114,465,1:25,114,466,1:25,114,467,1:25,114,468,1:25,114,469,1:25,114,470,1:25,114,471,1:25,114,472,1:26,114,473,1:26,114,474,1:26,114,475,1:26,114,476,1:26,114,477,1:26,114,478,1:27,114,479,1:27,114,480,1:27,114,481,1:27,114,482,1:27,113,483,1:27,113,484,1:28,113,485,1:28,113,486,1:28,113,487,1:28,113,488,1:28,113,489,1:28,113,490,1:28,113,491,1:29,113,492,1:29,113,493,1:29,113,494,1:29,113,495,1:29,113,496,1:29,113,497,1:29,113,498,1:29,113,499,1:29,113,500,1:29,113,501,1:29,113,502,1:29,113,503,1:29,113,504,1:29,113,505,1:29,113,506,1:29,113,507,1:29,113,508,1:29,113,509,1:29,113,510,1:29,113,511,1:28,113,512,1:28,114,513,1:28,114,514,1:28,114,515,1:27,114,516,1:27,114,517,1:27,114,518,1:26,115,519,1:26,115,520,1:25,115,521,1:25,115,522,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 23,97,0,1:24,98,2,1:24,95,17,1:24,90,33,1:25,87,46,1:24,83,59,1:24,81,72,1:24,113,413,1:24,111,432,1:25,109,451,1:25,109,468,1:26,115,485,1:25,119,500,1:26,116,512,1:26,122,522,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject VG
if not os.path.isdir(path_results + '/T2/VG'):
    os.makedirs(path_results + '/T2/VG')
os.chdir(path_results + '/T2/VG')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/VG/T2/01_0024_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-11/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 19,98,138,20:21,91,167,19:21,84,192,18:25,79,217,17:25,75,240,16:26,73,262,15:26,73,283,14:28,75,302,13:29,81,320,12:30,89,339,11:32,98,358,10:32,107,371,9:32,116,389,8:32,123,404,7:33,126,419,6:32,129,433,5:31,132,448,4:31,134,461,3:27,135,513,2:25,137,535,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 1,2 -start 11,71 -end 246,549')
f_crop = open('crop.txt', 'w')
f_crop.write('71,549,0,423,11,246')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 16,87,0,1:16,86,19,1:16,81,36,1:16,79,53,1:18,75,73,1:19,71,93,1:21,67,112,1:22,64,131,1:23,61,151,1:24,58,166,1:25,57,183,1:25,56,200,1:26,57,214,1:27,59,230,1:28,63,244,1:30,69,261,1:31,76,278,1:32,84,293,1:32,91,308,1:33,97,320,1:33,103,334,1:33,107,348,1:33,112,364,1:32,115,382,1:32,115,393,1:31,116,408,1:30,117,419,1:29,118,429,1:29,120,438,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 29,117,425,1:28,118,430,1:28,120,437,1:27,122,445,1:26,123,453,1:26,123,461,1:26,125,470,1:25,129,478,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject example
if not os.path.isdir(path_results + '/T2/VP'):
    os.makedirs(path_results + '/T2/VP')
os.chdir(path_results + '/T2/VP')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/VP/01_0100_space-compo/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 29,101,37,20:29,93,68,19:29,86,100,18:29,82,130,17:28,80,157,16:27,82,183,15:26,84,207,14:27,87,232,13:27,91,253,12:26,96,275,11:26,104,296,10:24,116,315,9:23,127,334,8:21,135,353,7:20,139,368,6:18,140,385,5:16,140,404,4:14,138,423,3:12,120,487,2:12,123,510,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 525')
f_crop = open('crop.txt', 'w')
f_crop.write('0,525,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 29,101,0,1:29,94,27,1:29,87,54,1:29,83,78,1:29,80,104,1:29,78,127,1:28,76,150,1:28,76,176,1:27,77,196,1:26,80,223,1:27,85,244,1:27,89,263,1:26,94,284,1:26,99,296,1:25,108,315,1:24,115,331,1:22,124,349,1:20,129,367,1:18,131,387,1:16,131,405,1:14,129,422,1:12,125,441,1:12,121,458,1:11,122,475,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 28,99,0,1:28,96,18,1:29,92,28,1:29,90,48,1:29,85,69,1:29,82,87,1:29,80,104,1:29,78,122,1:29,77,137,1:28,76,155,1:28,76,174,1:28,76,187,1:27,78,202,1:26,79,215,1:26,81,230,1:12,122,452,1:12,122,460,1:12,122,469,1:12,122,477,1:12,122,484,1:12,121,493,1:12,121,501,1:12,120,508,1:12,121,516,1:12,126,525,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject example
if not os.path.isdir(path_results + '/T2/AM'):
    os.makedirs(path_results + '/T2/AM')
os.chdir(path_results + '/T2/AM')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/AM/01_0100_compo-t2-spine/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
# sct_label_utils -i labels_vertebral.nii.gz -t display-voxel
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 25,96,32,20:22,91,66,19:21,89,98,18:19,89,128,17:19,88,155,16:19,87,182,15:21,88,209,14:23,91,234,13:25,96,258,12:28,104,281,11:30,113,303,10:32,122,324,9:33,129,344,8:34,133,360,7:35,134,377,6:35,133,393,5:35,133,411,4:35,132,430,3:39,124,498,2:36,127,523,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 539')
f_crop = open('crop.txt', 'w')
f_crop.write('0,539,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 25,87,0,1:24,83,22,1:23,82,44,1:23,81,64,1:22,81,83,1:21,80,104,1:20,80,126,1:20,80,144,1:20,80,161,1:20,81,180,1:21,82,201,1:23,83,221,1:23,86,237,1:26,90,256,1:27,95,273,1:29,101,291,1:30,106,305,1:32,113,322,1:33,119,344,1:34,123,358,1:34,127,375,1:34,127,394,1:34,125,410,1:35,124,427,1:36,121,442,1:37,119,455,1:39,117,470,1:40,115,482,1:39,118,493,1:38,119,505,1:37,120,515,1:37,124,527,1:36,129,539,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 40,116,479,1:40,119,488,1:39,121,499,1:38,123,508,1:38,124,516,1:37,125,525,1:36,128,532,1:36,131,539,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject example
if not os.path.isdir(path_results + '/T2/HB'):
    os.makedirs(path_results + '/T2/HB')
os.chdir(path_results + '/T2/HB')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/HB/01_0100_t2-compo/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 38,101,7,20:37,95,44,19:35,88,77,18:35,87,108,17:35,86,138,16:34,87,168,15:34,90,195,14:33,94,221,13:31,101,247,12:30,110,272,11:28,118,294,10:25,128,317,9:24,135,338,8:23,138,357,7:23,137,375,6:22,138,391,5:23,136,411,4:23,137,431,3:24,129,496,2:23,129,520,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 536')
f_crop = open('crop.txt', 'w')
f_crop.write('0,536,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 38,91,0,1:37,88,18,1:37,86,33,1:36,84,53,1:35,83,69,1:35,81,96,1:35,81,117,1:35,80,135,1:34,81,154,1:34,82,172,1:34,84,189,1:33,86,207,1:33,89,224,1:32,92,239,1:30,98,259,1:29,104,277,1:27,110,293,1:26,116,309,1:25,120,320,1:24,125,334,1:24,129,350,1:23,131,365,1:22,131,382,1:22,132,394,1:23,131,410,1:23,130,426,1:24,128,441,1:25,125,455,1:25,123,468,1:25,123,478,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 25,122,474,1:25,122,475,1:25,124,484,1:24,125,491,1:24,125,500,1:24,126,510,1:25,126,518,1:24,128,526,1:23,132,536,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject example
if not os.path.isdir(path_results + '/T2/PA'):
    os.makedirs(path_results + '/T2/PA')
os.chdir(path_results + '/T2/PA')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/PA/01_0038_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 24,93,80,20:23,85,110,19:23,77,139,18:23,74,165,17:22,72,191,16:21,71,215,15:21,74,237,14:21,78,261,13:22,83,283,12:22,91,303,11:22,102,325,10:23,112,342,9:23,123,357,8:23,131,373,7:23,135,388,6:24,137,403,5:25,137,419,4:26,137,436,3:27,122,500,2:26,123,523,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 1,2 -start 20,0 -end 240,538')
f_crop = open('crop.txt', 'w')
f_crop.write('0,538,0,max,20,240')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 25,83,0,1:25,77,21,1:24,71,40,1:24,66,62,1:23,62,81,1:23,57,100,1:23,55,115,1:23,52,132,1:23,49,154,1:23,47,173,1:22,46,188,1:22,46,207,1:21,47,223,1:21,49,239,1:21,51,256,1:21,55,272,1:22,60,288,1:22,65,300,1:22,72,316,1:22,78,329,1:22,85,341,1:23,92,353,1:23,99,365,1:23,104,376,1:23,108,390,1:24,110,403,1:24,111,413,1:25,110,426,1:26,108,439,1:27,105,453,1:27,102,468,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 25,82,0,1:26,79,12,1:25,77,21,1:24,75,30,1:24,71,40,1:24,69,47,1:27,101,472,1:27,101,480,1:27,101,490,1:27,100,498,1:27,99,507,1:27,99,515,1:27,99,522,1:27,101,529,1:27,103,534,1:27,105,538,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Creation of necessary files for T1 preprocessing

# Preprocessing for subject errsm_02
if not os.path.isdir(path_results + '/T1/errsm_02'):
    os.makedirs(path_results + '/T1/errsm_02')
os.chdir(path_results + '/T1/errsm_02')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_02/22-SPINE_T1/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 89,74,199,20:89,70,229,19:89,66,258,18:87,64,285,17:87,63,310,16:87,64,336,15:87,66,362,14:87,71,385,13:87,76,407,12:88,81,428,11:89,89,449,10:91,98,469,9:92,108,486,8:94,112,502,7:93,115,517,6:92,118,532,5:92,119,548,4:90,120,566,3:87,127,622,2:85,134,647,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 130 -end 650')
f_crop = open('crop.txt', 'w')
f_crop.write('130,650,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 90,78,0,1:89,72,25,1:89,66,44,1:89,64,61,1:89,62,82,1:89,60,96,1:89,59,114,1:88,58,133,1:88,58,149,1:87,57,169,1:87,58,185,1:87,59,201,1:87,61,215,1:87,62,233,1:87,65,247,1:87,67,259,1:87,69,272,1:88,73,285,1:89,76,298,1:89,80,310,1:90,85,326,1:91,91,339,1:92,96,350,1:92,101,365,1:93,105,378,1:93,107,391,1:92,109,404,1:92,110,416,1:92,111,428,1:91,111,444,1:90,111,459,1:89,112,470,1:88,116,479,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 88,115,478,1:88,119,486,1:88,122,493,1:88,125,501,1:88,127,508,1:88,128,514,1:88,130,520,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_04
if not os.path.isdir(path_results + '/T1/errsm_04'):
    os.makedirs(path_results + '/T1/errsm_04')
os.chdir(path_results + '/T1/errsm_04')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_04/16-SPINE_memprage/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 85,83,80,20:86,80,110,19:86,77,139,18:85,75,168,17:85,74,194,16:85,74,220,15:85,77,245,14:84,82,269,13:84,90,292,12:86,99,313,11:87,108,333,10:88,118,352,9:89,126,370,8:89,130,386,7:89,130,403,6:87,132,420,5:87,135,436,4:87,138,453,3:88,145,507,2:88,150,533,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 72 -end 548')
f_crop = open('crop.txt', 'w')
f_crop.write('72,548,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 86,74,0,1:85,70,33,1:85,69,58,1:85,68,105,1:85,69,148,1:84,74,186,1:85,83,217,1:86,93,242,1:87,107,275,1:89,117,304,1:88,123,331,1:87,127,371,1:89,132,415,1:87,142,440,1:89,157,476,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 88,132,418,1:88,135,425,1:88,137,433,1:88,139,441,1:88,141,449,1:88,143,456,1:88,145,464,1:88,148,470,1:88,151,476,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_05
if not os.path.isdir(path_results + '/T1/errsm_05'):
    os.makedirs(path_results + '/T1/errsm_05')
os.chdir(path_results + '/T1/errsm_05')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_05/23-SPINE_MEMPRAGE/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 84,78,106,20:84,72,133,19:85,65,160,18:85,60,184,17:85,55,206,16:86,53,228,15:86,53,251,14:85,57,272,13:85,61,293,12:86,69,312,11:88,78,330,10:88,90,345,9:89,102,360,8:90,109,375,7:90,115,388,6:90,119,402,5:89,122,416,4:88,125,431,3:86,127,488,2:86,131,514,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  35 -end 528 ')
f_crop = open('crop.txt', 'w')
f_crop.write('35,528,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 83,76,0,1:83,72,34,1:84,67,66,1:85,63,99,1:86,55,136,1:86,49,173,1:86,47,206,1:86,51,239,1:87,62,274,1:89,84,314,1:90,106,354,1:89,115,395,1:87,117,430,1:86,123,452,1:85,136,493,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 87,118,434,1:86,120,442,1:86,121,449,1:86,122,458,1:85,123,465,1:85,123,472,1:85,125,480,1:85,128,487,1:85,132,493,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_09
if not os.path.isdir(path_results + '/T1/errsm_09'):
    os.makedirs(path_results + '/T1/errsm_09')
os.chdir(path_results + '/T1/errsm_09')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_09/34-SPINE_MEMPRAGE2/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 82,67,96,20:83,63,123,19:83,63,148,18:84,62,171,17:85,61,193,16:85,62,214,15:86,65,235,14:86,68,256,13:87,73,275,12:88,79,293,11:88,86,312,10:88,94,329,9:88,102,345,8:88,106,361,7:88,108,377,6:88,110,393,5:88,112,410,4:89,115,426,3:89,121,482,2:89,132,504,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 32 -end 520')
f_crop = open('crop.txt', 'w')
f_crop.write('32,520,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 80,65,0,1:81,61,35,1:82,59,61,1:83,57,94,1:84,56,124,1:84,56,147,1:85,56,170,1:85,59,198,1:86,66,239,1:87,75,266,1:88,90,301,1:88,100,333,1:89,104,358,1:88,107,385,1:89,108,416,1:89,117,448,1:88,137,487,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 89,111,435,1:89,115,443,1:89,117,449,1:89,119,456,1:89,121,462,1:89,124,468,1:89,127,474,1:89,130,479,1:89,132,483,1:89,134,488,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_10
if not os.path.isdir(path_results + '/T1/errsm_10'):
    os.makedirs(path_results + '/T1/errsm_10')
os.chdir(path_results + '/T1/errsm_10')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_10/13-SPINE_MEMPRAGE/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 75,78,125,20:75,69,155,19:74,62,187,18:75,59,215,17:74,56,240,16:75,56,264,15:75,60,289,14:76,66,313,13:76,73,334,12:77,81,353,11:77,89,372,10:78,97,391,9:78,104,411,8:79,110,428,7:80,113,445,6:82,116,462,5:83,118,478,4:84,121,497,3:85,125,558,2:86,132,581,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 50 -end 595')
f_crop = open('crop.txt', 'w')
f_crop.write('50,595,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 78,79,0,1:76,75,34,1:74,68,72,1:74,60,112,1:74,55,145,1:74,52,175,1:74,52,202,1:75,55,235,1:76,62,267,1:76,72,293,1:77,84,324,1:78,97,355,1:80,106,386,1:83,111,414,1:84,114,446,1:85,114,477,1:85,121,505,1:86,128,527,1:86,138,545,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 85,115,490,1:85,118,499,1:85,120,507,1:85,122,514,1:85,124,521,1:85,126,528,1:85,129,535,1:85,131,540,1:85,134,545,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_11
if not os.path.isdir(path_results + '/T1/errsm_11'):
    os.makedirs(path_results + '/T1/errsm_11')
os.chdir(path_results + '/T1/errsm_11')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_11/24-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 91,73,91,20:90,67,121,19:88,61,151,18:86,57,180,17:84,55,205,16:83,56,231,15:83,57,258,14:83,62,283,13:83,71,306,12:82,82,327,11:80,94,346,10:79,112,361,9:79,127,377,8:79,139,391,7:79,142,406,6:81,144,421,5:81,145,437,4:82,143,455,3:84,133,517,2:85,137,540,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  15 -end 555 ')
f_crop = open('crop.txt', 'w')
f_crop.write('15,555,0,480')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 92,75,0,1:91,69,33,1:90,63,78,1:89,59,109,1:88,55,141,1:86,52,168,1:83,51,202,1:83,53,241,1:83,59,273,1:82,71,301,1:79,93,335,1:78,115,360,1:78,122,369,1:79,134,392,1:80,136,401,1:81,137,416,1:81,137,427,1:83,135,446,1:83,132,457,1:84,131,469,1:85,129,483,1:86,132,507,1:84,141,540,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 84,129,480,1:84,129,489,1:84,129,497,1:84,129,505,1:84,129,512,1:84,129,518,1:84,130,524,1:84,132,529,1:84,134,534,1:84,137,540,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_12
if not os.path.isdir(path_results + '/T1/errsm_12'):
    os.makedirs(path_results + '/T1/errsm_12')
os.chdir(path_results + '/T1/errsm_12')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_12/19-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 87,74,161,20:88,66,189,19:89,62,216,18:89,59,240,17:89,57,264,16:89,59,286,15:89,61,308,14:89,65,330,13:89,72,349,12:88,79,367,11:88,85,386,10:87,91,404,9:87,98,421,8:86,101,438,7:86,105,452,6:84,108,468,5:83,111,483,4:83,115,499,3:85,130,554,2:85,135,576,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  94 -end 589 ')
f_crop = open('crop.txt', 'w')
f_crop.write('94,589,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 85,74,0,1:85,74,21,1:86,65,54,1:88,57,96,1:89,54,120,1:89,52,150,1:89,52,181,1:89,54,206,1:89,59,231,1:88,67,263,1:88,74,284,1:86,85,313,1:86,96,352,1:83,106,398,1:84,114,430,1:85,118,450,1:86,129,473,1:84,140,495,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 84,116,443,1:84,119,451,1:84,122,460,1:84,125,467,1:84,128,476,1:84,130,483,1:84,133,488,1:84,137,495,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_13
if not os.path.isdir(path_results + '/T1/errsm_13'):
    os.makedirs(path_results + '/T1/errsm_13')
os.chdir(path_results + '/T1/errsm_13')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_13/33-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 95,66,117,20:92,63,151,19:89,60,183,18:86,59,213,17:84,61,239,16:83,64,265,15:83,68,292,14:83,75,317,13:83,82,343,12:83,89,367,11:84,97,389,10:84,106,409,9:84,115,428,8:84,117,449,7:85,120,467,6:85,123,486,5:86,125,504,4:86,126,525,3:87,126,594,2:89,132,619,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  38 -end 635 ')
f_crop = open('crop.txt', 'w')
f_crop.write('38,635,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 98,65,0,1:97,60,34,1:95,57,70,1:92,55,107,1:89,54,136,1:86,54,170,1:84,57,213,1:83,60,242,1:83,70,285,1:84,81,322,1:84,93,358,1:84,104,394,1:84,113,428,1:86,115,464,1:86,115,499,1:86,118,536,1:87,127,565,1:89,139,597,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 86,117,536,1:86,121,545,1:86,123,554,1:86,126,564,1:86,127,573,1:87,128,582,1:88,135,590,1:89,140,597,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_14
if not os.path.isdir(path_results + '/T1/errsm_14'):
    os.makedirs(path_results + '/T1/errsm_14')
os.chdir(path_results + '/T1/errsm_14')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_14/5002-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 89,96,103,20:88,89,126,19:87,82,152,18:87,79,175,17:87,77,198,16:86,77,218,15:86,77,239,14:86,78,258,13:86,80,278,12:86,83,297,11:86,87,315,10:86,91,332,9:85,96,349,8:86,100,364,7:85,103,379,6:85,107,393,5:86,110,407,4:86,113,421,3:88,126,477,2:89,136,500,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  41 -end 514 ')
f_crop = open('crop.txt', 'w')
f_crop.write('41,514,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 91,99,0,1:89,92,40,1:88,82,77,1:87,74,113,1:86,72,149,1:86,71,181,1:86,73,218,1:85,77,247,1:85,82,276,1:85,89,306,1:86,99,349,1:86,105,378,1:87,111,411,1:88,123,436,1:89,134,457,1:90,142,473,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 88,115,422,1:88,119,430,1:88,123,438,1:89,125,446,1:89,129,453,1:89,133,462,1:89,136,468,1:89,139,473,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_16
if not os.path.isdir(path_results + '/T1/errsm_16'):
    os.makedirs(path_results + '/T1/errsm_16')
os.chdir(path_results + '/T1/errsm_16')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_16/23-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 91,73,179,20:88,66,206,19:85,61,232,18:82,58,255,17:80,56,278,16:79,57,301,15:80,58,323,14:80,61,344,13:81,65,364,12:82,70,385,11:84,76,403,10:85,84,423,9:85,90,440,8:85,93,456,7:86,96,473,6:86,99,489,5:86,103,506,4:87,108,524,3:88,116,582,2:87,124,605,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  114 -end 618 ')
f_crop = open('crop.txt', 'w')
f_crop.write('114,618,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 95,77,0,1:95,73,21,1:93,68,48,1:90,62,76,1:87,57,103,1:84,54,129,1:80,52,162,1:80,52,190,1:79,55,225,1:81,61,255,1:84,70,289,1:85,79,317,1:85,86,343,1:86,92,372,1:86,98,401,1:87,101,427,1:87,105,452,1:88,118,480,1:87,130,504,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 87,105,453,1:87,108,461,1:87,111,470,1:87,114,479,1:87,116,486,1:87,119,493,1:87,123,499,1:87,127,504,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_17
if not os.path.isdir(path_results + '/T1/errsm_17'):
    os.makedirs(path_results + '/T1/errsm_17')
os.chdir(path_results + '/T1/errsm_17')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_17/41-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 84,80,156,20:87,73,184,19:88,68,210,18:88,61,236,17:90,59,259,16:90,59,282,15:91,60,304,14:92,63,326,13:93,67,347,12:93,74,365,11:94,81,384,10:94,90,401,9:94,98,418,8:93,104,433,7:92,107,449,6:91,109,466,5:90,112,482,4:90,113,498,3:85,112,556,2:85,121,579,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 84 -end 593')
f_crop = open('crop.txt', 'w')
f_crop.write('84,593,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 85,86,0,1:84,78,29,1:84,71,62,1:86,66,94,1:88,60,129,1:89,56,160,1:90,54,178,1:91,55,217,1:93,61,254,1:94,71,287,1:94,82,314,1:94,92,339,1:93,97,357,1:92,102,383,1:90,104,406,1:88,103,432,1:87,104,460,1:85,114,486,1:84,127,509,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 86,104,459,1:86,107,467,1:84,110,475,1:84,113,484,1:84,116,494,1:84,120,502,1:84,125,509,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_18
if not os.path.isdir(path_results + '/T1/errsm_18'):
    os.makedirs(path_results + '/T1/errsm_18')
os.chdir(path_results + '/T1/errsm_18')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_18/36-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 86,82,88,20:86,76,120,19:86,72,151,18:86,71,178,17:85,69,204,16:85,67,230,15:85,67,255,14:85,68,279,13:85,74,302,12:85,79,325,11:86,87,345,10:85,95,365,9:84,102,385,8:83,106,402,7:83,109,419,6:82,114,436,5:83,118,454,4:85,123,473,3:85,132,531,2:85,135,555,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 16 -end 569')
f_crop = open('crop.txt', 'w')
f_crop.write('16,569,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 85,87,0,1:85,79,29,1:86,74,59,1:86,69,87,1:86,67,116,1:86,65,142,1:85,63,176,1:85,61,207,1:85,61,228,1:85,63,262,1:85,71,299,1:85,81,335,1:85,90,361,1:83,99,390,1:83,105,419,1:84,111,450,1:86,119,496,1:85,130,523,1:84,138,553,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 86,120,498,1:85,124,507,1:85,126,516,1:84,127,524,1:84,128,532,1:85,130,540,1:85,133,547,1:85,137,553,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_21
if not os.path.isdir(path_results + '/T1/errsm_21'):
    os.makedirs(path_results + '/T1/errsm_21')
os.chdir(path_results + '/T1/errsm_21')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_21/27-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 83,64,120,20:82,60,152,19:82,57,180,18:81,57,207,17:81,57,233,16:80,59,258,15:81,61,284,14:81,65,307,13:81,73,328,12:81,82,349,11:81,91,367,10:84,103,385,9:86,115,403,8:88,121,419,7:88,122,435,6:89,120,452,5:89,118,470,4:89,117,490,3:85,114,549,2:85,116,575,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 46 -end 589')
f_crop = open('crop.txt', 'w')
f_crop.write('46,589,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 84,66,0,1:83,60,33,1:82,57,60,1:82,53,100,1:82,51,130,1:81,50,163,1:80,51,203,1:80,55,233,1:80,65,277,1:82,79,310,1:83,94,340,1:87,111,377,1:88,111,411,1:88,108,444,1:87,106,469,1:86,111,501,1:85,115,527,1:84,119,543,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 86,107,487,1:86,109,494,1:86,110,501,1:85,110,509,1:85,110,517,1:85,110,525,1:84,111,532,1:84,114,538,1:84,118,543,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_22
if not os.path.isdir(path_results + '/T1/errsm_22'):
    os.makedirs(path_results + '/T1/errsm_22')
os.chdir(path_results + '/T1/errsm_22')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_22/29-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 84,61,140,20:85,59,172,19:85,59,198,18:85,56,224,17:84,55,247,16:84,54,269,15:84,55,292,14:84,57,313,13:84,61,334,12:84,67,354,11:85,75,373,10:86,84,390,9:87,90,406,8:86,92,423,7:87,91,440,6:86,91,457,5:86,94,474,4:84,97,493,3:86,101,546,2:86,111,569,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 65 -end 582')
f_crop = open('crop.txt', 'w')
f_crop.write('65,582,0,460')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 83,62,0,1:83,58,22,1:83,53,55,1:84,52,88,1:85,50,126,1:84,50,158,1:84,49,188,1:84,50,225,1:84,53,249,1:84,58,274,1:85,69,307,1:86,83,348,1:86,86,385,1:84,89,425,1:82,91,460,1:86,102,487,1:87,107,502,1:88,116,517,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 82,91,461,1:83,94,471,1:83,95,478,1:85,99,486,1:85,101,493,1:86,104,500,1:86,107,507,1:87,110,512,1:87,114,517,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')

# Preprocessing for subject errsm_23
if not os.path.isdir(path_results + '/T1/errsm_23'):
    os.makedirs(path_results + '/T1/errsm_23')
os.chdir(path_results + '/T1/errsm_23')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_23/29-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 79,82,135,20:80,80,165,19:80,77,191,18:81,76,216,17:82,75,239,16:83,74,262,15:84,75,285,14:85,77,307,13:85,80,327,12:85,84,346,11:85,89,365,10:83,94,384,9:83,98,401,8:82,99,418,7:83,98,435,6:83,99,451,5:84,102,469,4:84,107,485,3:84,112,541,2:86,122,562,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 60 -end 580')
f_crop = open('crop.txt', 'w')
f_crop.write('60,580,10,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 78,81,0,1:78,75,32,1:78,73,74,1:81,71,118,1:81,70,145,1:82,69,178,1:83,69,200,1:84,70,236,1:85,75,272,1:84,81,301,1:83,87,333,1:83,92,374,1:84,94,401,1:84,98,426,1:84,103,464,1:84,113,485,1:85,121,504,1:86,131,520,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 84,104,468,1:84,107,475,1:84,110,482,1:84,112,489,1:84,114,495,1:85,116,502,1:85,120,508,1:86,125,513,1:86,130,520,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_24
if not os.path.isdir(path_results + '/T1/errsm_24'):
    os.makedirs(path_results + '/T1/errsm_24')
os.chdir(path_results + '/T1/errsm_24')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_24/20-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 75,72,106,20:75,72,138,19:75,68,167,18:76,66,195,17:77,63,222,16:77,63,249,15:78,65,275,14:79,69,300,13:81,76,324,12:82,86,346,11:83,97,367,10:84,111,384,9:85,123,399,8:85,128,416,7:85,128,432,6:85,128,449,5:84,127,466,4:86,130,483,3:87,130,541,2:89,135,564,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 26 -end 579')
f_crop = open('crop.txt', 'w')
f_crop.write('26,579,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 77,72,0,1:76,67,33,1:74,64,57,1:75,63,86,1:75,60,127,1:76,60,156,1:77,58,181,1:78,57,219,1:79,60,248,1:80,65,274,1:81,72,301,1:83,88,335,1:84,105,362,1:85,116,384,1:85,120,408,1:85,121,441,1:86,120,473,1:87,120,494,1:88,128,517,1:88,132,535,1:89,139,553,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 87,122,501,1:88,125,508,1:88,126,515,1:88,128,523,1:88,128,531,1:89,130,539,1:89,132,546,1:89,135,553,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_25
if not os.path.isdir(path_results + '/T1/errsm_25'):
    os.makedirs(path_results + '/T1/errsm_25')
os.chdir(path_results + '/T1/errsm_25')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_25/25-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 89,70,174,20:88,67,204,19:85,63,232,18:85,61,256,17:84,61,280,16:85,61,302,15:85,63,324,14:85,66,346,13:86,72,366,12:86,77,384,11:86,83,402,10:86,92,420,9:86,101,438,8:86,107,454,7:86,112,469,6:85,117,484,5:85,121,500,4:85,126,518,3:87,137,577,2:89,150,601,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 105 -end 614')
f_crop = open('crop.txt', 'w')
f_crop.write('105,614,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 91,76,0,1:90,68,33,1:89,61,63,1:87,58,96,1:86,56,118,1:84,54,148,1:84,54,182,1:85,56,212,1:85,61,240,1:86,67,266,1:86,76,293,1:86,91,326,1:85,100,352,1:85,109,379,1:86,115,402,1:86,120,436,1:86,129,465,1:87,139,481,1:89,158,509,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 87,123,454,1:87,126,462,1:87,129,468,1:87,132,474,1:87,136,481,1:87,140,488,1:87,143,495,1:87,148,501,1:87,153,506,1:87,155,509,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_30
if not os.path.isdir(path_results + '/T1/errsm_30'):
    os.makedirs(path_results + '/T1/errsm_30')
os.chdir(path_results + '/T1/errsm_30')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_30/51-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 74,73,114,20:76,66,144,19:76,61,173,18:76,58,199,17:76,56,224,16:76,55,247,15:77,57,271,14:78,60,294,13:81,67,317,12:82,75,337,11:84,83,357,10:85,97,374,9:86,108,391,8:87,114,407,7:87,118,422,6:88,121,437,5:88,121,455,4:87,123,475,3:84,121,535,2:84,124,560,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 36 -end 575')
f_crop = open('crop.txt', 'w')
f_crop.write('36,575,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 72,75,0,1:73,71,26,1:74,67,49,1:75,65,73,1:75,60,99,1:75,57,120,1:75,52,155,1:76,50,195,1:76,51,224,1:79,54,256,1:80,61,278,1:82,71,304,1:85,82,327,1:86,95,351,1:87,107,373,1:87,112,397,1:88,113,421,1:86,113,454,1:84,113,479,1:83,116,497,1:83,121,518,1:84,128,539,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 83,113,484,1:83,115,495,1:83,116,503,1:83,117,511,1:83,118,519,1:83,120,527,1:83,123,534,1:83,126,539,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_31
if not os.path.isdir(path_results + '/T1/errsm_31'):
    os.makedirs(path_results + '/T1/errsm_31')
os.chdir(path_results + '/T1/errsm_31')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_31/31-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 97,82,90,20:95,78,123,19:94,74,154,18:91,70,186,17:90,66,214,16:88,64,241,15:88,65,268,14:87,69,295,13:87,73,320,12:87,79,343,11:87,88,365,10:86,95,388,9:86,102,409,8:86,106,429,7:86,111,448,6:87,113,467,5:90,115,486,4:89,116,505,3:85,107,568,2:85,114,591,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  14 -end 607 ')
f_crop = open('crop.txt', 'w')
f_crop.write('14,607,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 100,76,0,1:99,74,32,1:98,70,59,1:96,67,90,1:94,65,119,1:93,64,141,1:92,62,162,1:90,59,196,1:89,58,224,1:88,60,258,1:87,63,283,1:87,69,314,1:86,75,335,1:86,82,361,1:86,91,390,1:86,97,416,1:86,102,437,1:87,105,480,1:86,102,511,1:85,102,537,1:83,106,560,1:84,109,575,1:85,118,593,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 84,101,539,1:84,103,548,1:84,104,556,1:84,105,564,1:84,106,572,1:84,108,579,1:84,111,585,1:84,114,589,1:84,117,593,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_32
if not os.path.isdir(path_results + '/T1/errsm_32'):
    os.makedirs(path_results + '/T1/errsm_32')
os.chdir(path_results + '/T1/errsm_32')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_32/16-SPINE_T1/echo_2.09 /*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 78,64,80,20:80,61,113,19:81,59,143,18:82,58,170,17:84,57,196,16:84,56,221,15:86,58,245,14:87,62,269,13:87,67,292,12:89,76,312,11:89,85,332,10:89,97,351,9:89,106,369,8:89,110,386,7:89,109,402,6:89,108,418,5:89,108,436,4:88,107,456,3:87,102,517,2:86,107,544,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 7 -end 559')
f_crop = open('crop.txt', 'w')
f_crop.write('7,559,0,484')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 75,64,0,1:76,59,32,1:78,56,66,1:80,53,109,1:82,51,139,1:84,51,170,1:85,51,214,1:86,53,241,1:87,60,276,1:89,71,310,1:89,84,338,1:88,94,363,1:89,100,384,1:88,99,409,1:88,99,435,1:88,96,464,1:87,96,495,1:87,102,521,1:85,111,552,1')
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x 88,94,485,1:87,96,496,1:87,97,505,1:86,99,515,1:86,99,523,1:86,100,531,1:85,103,540,1:85,107,546,1:85,111,552,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_33
if not os.path.isdir(path_results + '/T1/errsm_33'):
    os.makedirs(path_results + '/T1/errsm_33')
os.chdir(path_results + '/T1/errsm_33')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_33/30-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 85,75,66,20:84,70,98,19:85,68,131,18:85,69,160,17:86,69,187,16:86,69,212,15:87,72,238,14:87,74,264,13:88,81,287,12:89,90,309,11:90,101,329,10:90,114,348,9:90,124,364,8:91,129,382,7:90,133,397,6:90,133,414,5:90,133,431,4:90,133,448,3:90,127,509,2:89,135,532,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 26 -end 549')
f_crop = open('crop.txt', 'w')
f_crop.write('26,549,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 82,70,0,1:83,66,22,1:84,62,58,1:84,60,87,1:85,60,121,1:86,62,141,1:86,62,170,1:87,65,207,1:87,70,241,1:89,80,272,1:90,93,301,1:91,109,332,1:91,121,367,1:91,123,404,1:89,119,441,1:88,127,484,1:89,131,500,1:88,138,523,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject 1
if not os.path.isdir(path_results + '/T1/1'):
    os.makedirs(path_results + '/T1/1')
os.chdir(path_results + '/T1/1')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/ED/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-101/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 83,100,128,20:85,93,160,19:85,88,187,18:85,85,214,17:84,84,238,16:83,83,263,15:84,87,288,14:85,93,311,13:86,101,333,12:86,110,353,11:87,119,371,10:86,129,388,9:85,137,405,8:84,143,421,7:84,145,439,6:83,146,455,5:82,149,472,4:82,150,490,3:80,148,548,2:80,156,570,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 42 -end 574')
f_crop = open('crop.txt', 'w')
f_crop.write('42,574,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 80,108,0,1:79,101,25,1:80,96,49,1:81,89,84,1:83,85,111,1:84,82,139,1:85,79,168,1:84,78,197,1:85,82,249,1:86,93,285,1:86,105,313,1:85,122,349,1:85,133,375,1:83,138,402,1:82,140,425,1:81,139,452,1:80,140,487,1:80,148,508,1:79,157,532,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject ALT
if not os.path.isdir(path_results + '/T1/ALT'):
    os.makedirs(path_results + '/T1/ALT')
os.chdir(path_results + '/T1/ALT')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/ALT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 76,95,73,20:76,88,105,19:76,82,133,18:76,78,160,17:76,75,186,16:77,74,210,15:78,76,235,14:79,81,259,13:81,89,282,12:82,98,302,11:83,108,321,10:85,119,341,9:87,131,360,8:87,139,377,7:89,143,394,6:91,146,412,5:93,147,430,4:94,145,451,3:93,133,513,2:92,131,539,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 63 -end 556')
f_crop = open('crop.txt', 'w')
f_crop.write('63,556,0,410')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 76,85,0,1:76,81,20,1:76,80,33,1:76,79,46,1:76,76,66,1:76,73,85,1:76,71,99,1:77,71,112,1:77,69,130,1:77,68,143,1:78,69,161,1:78,71,174,1:80,77,198,1:81,82,215,1:82,91,236,1:82,94,244,1:85,108,271,1:85,113,280,1:86,121,296,1:87,127,307,1:88,136,328,1:89,138,339,1:90,140,350,1:91,140,358,1:93,139,380,1:94,137,392,1:95,132,418,1:95,131,427,1:92,132,451,1:91,130,473,1:89,131,493,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject JD
if not os.path.isdir(path_results + '/T1/JD'):
    os.makedirs(path_results + '/T1/JD')
os.chdir(path_results + '/T1/JD')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/JD/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 77,76,13,20:75,70,51,19:74,69,84,18:74,68,115,17:74,67,144,16:74,67,172,15:73,69,199,14:74,72,226,13:75,80,253,12:76,87,278,11:77,96,302,10:79,109,324,9:79,118,342,8:78,123,363,7:78,124,382,6:76,124,403,5:76,122,423,4:75,122,445,3:77,105,512,2:80,113,542,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 13 -end 555')
f_crop = open('crop.txt', 'w')
f_crop.write('13,555,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 76,65,0,1:74,62,43,1:74,60,79,1:74,60,102,1:73,60,130,1:73,60,164,1:74,63,198,1:74,69,225,1:76,77,256,1:77,90,289,1:79,105,320,1:78,114,345,1:77,116,372,1:76,116,399,1:74,111,432,1:75,103,460,1:76,102,493,1:79,107,516,1:81,114,541,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject JW
if not os.path.isdir(path_results + '/T1/JW'):
    os.makedirs(path_results + '/T1/JW')
os.chdir(path_results + '/T1/JW')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/JW/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 99,107,46,20:98,103,79,19:97,96,108,18:95,93,138,17:93,92,164,16:91,92,190,15:90,95,217,14:90,99,241,13:90,105,263,12:90,114,285,11:90,123,305,10:90,132,325,9:89,136,346,8:89,140,364,7:89,141,380,6:89,141,398,5:88,140,415,4:86,139,434,3:88,129,495,2:88,126,521,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 42 -end 538')
f_crop = open('crop.txt', 'w')
f_crop.write('42,538,0,426')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 98,98,0,1:98,96,21,1:98,92,47,1:96,89,72,1:94,87,108,1:92,87,140,1:91,89,171,1:90,98,218,1:90,107,244,1:90,118,273,1:89,126,301,1:89,132,340,1:88,131,374,1:87,129,401,1:87,126,429,1:88,126,454,1:86,125,477,1:88,126,496,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject MLL
if not os.path.isdir(path_results + '/T1/MLL'):
    os.makedirs(path_results + '/T1/MLL')
os.chdir(path_results + '/T1/MLL')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/MLL_1016/01_0008_sc-mprage-1mm-2palliers-fov384-comp-sp-7/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 100,95,121,20:99,91,153,19:98,86,181,18:97,83,206,17:97,82,230,16:95,82,254,15:94,84,278,14:93,87,301,13:92,93,323,12:91,99,343,11:91,108,363,10:90,118,381,9:89,126,399,8:89,129,415,7:89,132,432,6:89,133,449,5:87,133,467,4:88,130,485,3:83,118,543,2:82,126,565,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 1,2 -start 25,50 -end 252,578')
f_crop = open('crop.txt', 'w')
f_crop.write('50,578,0,max,25,252')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 100,76,0,1:100,69,28,1:100,64,50,1:99,60,81,1:99,57,103,1:98,54,132,1:96,51,166,1:95,52,199,1:94,53,221,1:93,58,256,1:91,68,290,1:90,75,311,1:90,86,340,1:89,97,372,1:89,99,402,1:88,94,431,1:84,87,468,1:83,90,484,1:82,96,506,1:81,103,528,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject MT
if not os.path.isdir(path_results + '/T1/MT'):
    os.makedirs(path_results + '/T1/MT')
os.chdir(path_results + '/T1/MT')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/MT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 93,82,50,20:93,77,81,19:93,76,108,18:90,72,137,17:91,73,163,16:89,74,189,15:89,78,215,14:89,82,239,13:89,91,263,12:90,101,284,11:89,112,304,10:89,123,322,9:88,133,339,8:87,137,358,7:86,138,374,6:86,138,391,5:86,135,408,4:86,134,427,3:86,122,484,2:87,119,509,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 19 -end 526')
f_crop = open('crop.txt', 'w')
f_crop.write('19,526,0,440')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 94,78,0,1:94,71,32,1:92,71,53,1:91,68,90,1:91,66,120,1:90,67,152,1:90,70,184,1:89,74,208,1:89,83,238,1:90,97,270,1:90,113,302,1:87,124,330,1:86,130,366,1:86,128,394,1:86,126,412,1:86,118,445,1:88,116,474,1:89,115,507,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject T045
if not os.path.isdir(path_results + '/T1/T045'):
    os.makedirs(path_results + '/T1/T045')
os.chdir(path_results + '/T1/T045')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/T045/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 82,90,120,20:82,89,151,19:84,85,179,18:85,81,203,17:86,78,227,16:86,77,252,15:86,78,276,14:86,81,300,13:86,87,323,12:87,92,343,11:86,99,362,10:87,107,383,9:88,113,404,8:88,118,421,7:89,120,438,6:88,123,455,5:87,123,474,4:86,122,492,3:85,123,559,2:86,126,586,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 1,2 -start 12,80 -end 254,602')
f_crop = open('crop.txt', 'w')
f_crop.write('80,602,0,max,12,254')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 82,70,0,1:82,70,39,1:83,68,80,1:84,64,110,1:86,60,153,1:86,60,186,1:86,64,223,1:86,73,257,1:87,82,288,1:88,92,316,1:89,100,344,1:88,103,367,1:88,104,381,1:87,105,391,1:86,104,402,1:85,103,416,1:85,103,429,1:84,103,451,1:85,108,479,1:86,113,505,1:87,116,522,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject T047
if not os.path.isdir(path_results + '/T1/T047'):
    os.makedirs(path_results + '/T1/T047')
os.chdir(path_results + '/T1/T047')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/T047/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 87,105,96,20:85,99,129,19:85,94,157,18:85,91,184,17:85,89,208,16:85,89,232,15:85,90,257,14:85,94,280,13:84,99,303,12:84,106,325,11:84,113,344,10:84,120,363,9:84,127,381,8:84,131,398,7:84,131,415,6:85,132,432,5:85,131,449,4:86,130,467,3:85,131,535,2:85,131,561,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 30 -end 577')
f_crop = open('crop.txt', 'w')
f_crop.write('30,577,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 87,101,0,1:86,96,29,1:86,91,61,1:85,88,103,1:85,85,136,1:85,84,160,1:85,83,184,1:85,86,234,1:85,89,257,1:85,95,280,1:84,105,310,1:84,111,333,1:85,118,358,1:86,123,393,1:86,122,416,1:86,121,450,1:85,122,478,1:84,126,498,1:85,128,518,1:86,132,547,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject VC
if not os.path.isdir(path_results + '/T1/VC'):
    os.makedirs(path_results + '/T1/VC')
os.chdir(path_results + '/T1/VC')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/VC/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 83,81,107,20:83,73,137,19:83,67,166,18:83,65,193,17:84,62,217,16:84,60,240,15:85,59,262,14:85,62,285,13:85,71,306,12:85,80,326,11:86,89,344,10:85,101,362,9:86,112,379,8:86,116,396,7:86,119,412,6:86,119,427,5:86,119,442,4:85,118,457,3:85,107,519,2:85,110,544,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 55 -end 562')
f_crop = open('crop.txt', 'w')
f_crop.write('55,562,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 82,86,0,1:82,74,36,1:83,65,69,1:83,62,96,1:84,58,130,1:84,56,158,1:85,54,193,1:85,55,220,1:85,63,247,1:85,76,277,1:84,88,302,1:86,101,329,1:86,108,349,1:86,111,370,1:86,108,399,1:86,103,427,1:86,101,449,1:85,103,472,1:84,111,507,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject VG
if not os.path.isdir(path_results + '/T1/VG'):
    os.makedirs(path_results + '/T1/VG')
os.chdir(path_results + '/T1/VG')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/VG/T1/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 64,88,120,20:64,79,148,19:66,73,173,18:68,67,197,17:69,62,218,16:71,59,240,15:72,59,260,14:74,62,280,13:77,67,299,12:80,75,317,11:81,81,333,10:82,88,347,9:83,98,364,8:84,105,378,7:85,108,392,6:86,111,405,5:86,113,419,4:86,114,433,3:85,117,484,2:84,120,508,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 47 -end 525')
f_crop = open('crop.txt', 'w')
f_crop.write('47,525,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 67,92,0,1:63,84,39,1:63,78,76,1:65,72,97,1:66,66,124,1:67,61,146,1:69,56,169,1:70,53,191,1:73,54,215,1:75,58,238,1:78,65,260,1:80,75,287,1:83,88,314,1:84,98,336,1:86,107,371,1:86,107,396,1:86,108,415,1:84,113,439,1:83,116,458,1:83,119,478,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject VP
if not os.path.isdir(path_results + '/T1/VP'):
    os.makedirs(path_results + '/T1/VP')
os.chdir(path_results + '/T1/VP')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/VP/01_0011_sc-mprage-1mm-2palliers-fov384-comp-sp-25/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 105,91,68,20:105,83,100,19:105,75,132,18:105,71,161,17:105,69,187,16:105,69,214,15:103,70,240,14:103,74,262,13:103,79,285,12:102,84,306,11:100,92,326,10:99,102,346,9:98,114,365,8:97,121,383,7:95,125,400,6:93,125,417,5:91,126,436,4:89,124,455,3:87,111,516,2:87,110,542,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 63 -end 561')
f_crop = open('crop.txt', 'w')
f_crop.write('63,561,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 104,79,0,1:104,74,26,1:105,70,48,1:106,66,75,1:105,64,92,1:105,63,115,1:104,63,150,1:103,64,173,1:103,66,191,1:103,70,209,1:102,76,234,1:100,86,264,1:98,100,297,1:97,108,317,1:95,114,335,1:93,116,356,1:90,114,385,1:87,109,412,1:87,107,438,1:87,107,454,1:88,110,476,1:86,113,498,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject pain_pilot_1
if not os.path.isdir(path_results + '/T1/pain_pilot_1'):
    os.makedirs(path_results + '/T1/pain_pilot_1')
os.chdir(path_results + '/T1/pain_pilot_1')
sct.run('dcm2nii -o . -r N '+folder_data_pain+'/d_sp_pain_pilot1/24-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 91,89,108,20:91,85,141,19:91,82,171,18:91,79,199,17:91,79,225,16:91,77,249,15:91,77,274,14:91,81,297,13:91,86,319,12:91,91,341,11:91,97,362,10:93,104,383,9:93,107,403,8:93,108,420,7:91,108,436,6:90,108,451,5:90,109,468,4:89,108,486,3:90,102,539,2:89,108,564,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 30 -end 578')
f_crop = open('crop.txt', 'w')
f_crop.write('30,578,0,483')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 87,84,0,1:88,82,25,1:90,80,42,1:91,77,80,1:89,75,111,1:91,72,145,1:91,71,185,1:90,70,218,1:91,72,245,1:90,77,276,1:91,84,309,1:93,90,335,1:93,97,364,1:92,100,390,1:91,100,412,1:89,99,440,1:90,97,461,1:91,94,481,1:91,98,502,1:89,104,527,1:88,107,548,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject pain_pilot_2
if not os.path.isdir(path_results + '/T1/pain_pilot_2'):
    os.makedirs(path_results + '/T1/pain_pilot_2')
os.chdir(path_results + '/T1/pain_pilot_2')
sct.run('dcm2nii -o . -r N '+folder_data_pain+'/d_sp_pain_pilot2/13-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 89,82,55,20:88,79,88,19:87,78,116,18:87,77,143,17:86,77,167,16:86,77,192,15:86,77,217,14:86,79,241,13:86,82,264,12:86,86,287,11:86,93,307,10:87,101,327,9:88,108,348,8:90,111,367,7:90,115,386,6:90,113,404,5:88,113,423,4:86,114,443,3:86,110,502,2:89,118,526,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 2 -end 543')
f_crop = open('crop.txt', 'w')
f_crop.write('2,543,0,476')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 92,76,0,1:90,74,20,1:89,72,50,1:88,73,74,1:87,72,100,1:87,71,129,1:86,71,162,1:86,71,201,1:86,72,225,1:87,76,258,1:87,83,289,1:87,89,311,1:88,96,337,1:89,103,358,1:90,107,381,1:87,105,412,1:86,103,440,1:84,100,467,1:83,104,491,1:87,112,512,1:90,121,539,1:90,121,541,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject pain_pilot_4
if not os.path.isdir(path_results + '/T1/pain_pilot_4'):
    os.makedirs(path_results + '/T1/pain_pilot_4')
os.chdir(path_results + '/T1/pain_pilot_4')
sct.run('dcm2nii -o . -r N '+folder_data_pain+'/d_sp_pain_pilot4/33-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 77,76,87,20:78,71,119,19:78,70,150,18:79,69,176,17:79,68,202,16:80,67,228,15:81,68,253,14:82,72,277,13:83,77,300,12:85,82,322,11:86,90,343,10:86,99,362,9:85,107,381,8:83,111,398,7:82,113,415,6:83,114,433,5:82,114,452,4:82,116,472,3:82,124,538,2:82,130,561,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 10 -end 579')
f_crop = open('crop.txt', 'w')
f_crop.write('10,579,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 73,76,0,1:75,69,32,1:77,65,76,1:78,64,108,1:78,64,130,1:79,63,157,1:80,62,188,1:81,62,220,1:81,64,250,1:83,69,279,1:85,76,304,1:86,84,331,1:86,94,361,1:84,102,388,1:83,106,415,1:83,107,446,1:82,108,469,1:82,111,503,1:81,118,522,1:81,124,540,1:81,134,569,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject TM
if not os.path.isdir(path_results + '/T1/TM'):
    os.makedirs(path_results + '/T1/TM')
os.chdir(path_results + '/T1/TM')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/TM_T057c/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 100,100,48,20:99,92,79,19:99,83,111,18:98,79,141,17:97,76,168,16:97,75,195,15:97,75,221,14:97,80,246,13:96,87,271,12:97,94,294,11:95,105,314,10:95,119,332,9:94,130,348,8:92,134,372,7:92,135,390,6:92,135,408,5:92,133,426,4:92,131,447,3:89,111,510,2:89,112,537,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 1,2 -start 14,12 -end 252,556')
f_crop = open('crop.txt', 'w')
f_crop.write('12,556,0,max,14,252')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 98,76,0,1:98,73,22,1:98,67,42,1:98,62,81,1:98,60,107,1:97,57,142,1:97,54,183,1:97,57,216,1:96,62,244,1:97,71,274,1:95,86,305,1:94,105,336,1:92,112,379,1:92,107,427,1:91,97,459,1:89,96,501,1:88,97,525,1:87,97,544,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_20
if not os.path.isdir(path_results + '/T1/errsm_20'):
    os.makedirs(path_results + '/T1/errsm_20')
os.chdir(path_results + '/T1/errsm_20')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_20/12-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 86,74,139,20:86,65,173,19:86,58,204,18:86,55,234,17:85,54,263,16:84,55,291,15:84,60,318,14:83,67,344,13:83,79,367,12:83,91,388,11:83,107,404,10:83,124,421,9:84,138,437,8:85,147,453,7:85,148,469,6:86,148,486,5:86,147,503,4:86,141,521,3:87,130,583,2:87,137,612,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 100 -end 630')
f_crop = open('crop.txt', 'w')
f_crop.write('100,630,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 86,74,0,1:86,68,18,1:86,64,39,1:86,59,65,1:86,55,83,1:85,52,109,1:86,49,135,1:85,48,163,1:84,49,193,1:83,61,242,1:83,76,273,1:83,92,298,1:83,109,317,1:84,134,350,1:85,139,379,1:87,133,412,1:88,124,450,1:87,126,479,1:86,133,502,1:86,142,530,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject pain_pilot_3
if not os.path.isdir(path_results + '/T1/pain_pilot_3'):
    os.makedirs(path_results + '/T1/pain_pilot_3')
os.chdir(path_results + '/T1/pain_pilot_3')
sct.run('dcm2nii -o . -r N '+folder_data_pain+'/d_sp_pain_pilot3/16-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 78,60,116,20:78,59,148,19:80,61,178,18:82,62,206,17:82,63,231,16:84,64,257,15:85,66,282,14:87,70,306,13:87,76,328,12:89,82,350,11:89,91,370,10:89,103,386,9:89,111,405,8:88,118,421,7:88,121,438,6:88,127,454,5:87,131,470,4:88,132,487,3:90,128,546,2:94,133,573,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  74 -end 592 ')
f_crop = open('crop.txt', 'w')
f_crop.write('74,592,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 76,51,0,1:77,49,34,1:78,49,59,1:80,51,103,1:82,53,133,1:83,55,167,1:85,58,197,1:86,61,222,1:87,67,246,1:88,76,276,1:89,88,305,1:89,106,338,1:88,117,372,1:88,121,406,1:88,119,442,1:90,122,467,1:92,128,492,1:95,133,516,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_34
if not os.path.isdir(path_results + '/T1/errsm_34'):
    os.makedirs(path_results + '/T1/errsm_34')
os.chdir(path_results + '/T1/errsm_34')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_34/41-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 81,79,120,20:82,76,153,19:85,73,183,18:86,70,211,17:87,67,236,16:89,65,261,15:89,66,286,14:90,68,308,13:90,73,332,12:90,79,354,11:91,86,375,10:91,98,394,9:91,108,413,8:90,114,430,7:89,117,448,6:87,120,466,5:86,126,484,4:85,131,504,3:83,134,569,2:83,138,595,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 77 -end 615')
f_crop = open('crop.txt', 'w')
f_crop.write('77,615,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 80,71,0,1:81,69,27,1:83,68,74,1:85,65,116,1:86,63,140,1:88,60,174,1:89,59,206,1:90,62,228,1:90,66,251,1:91,77,289,1:91,92,322,1:90,106,357,1:88,113,388,1:85,120,421,1:84,123,455,1:85,127,475,1:84,133,501,1:82,142,538,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_35
if not os.path.isdir(path_results + '/T1/errsm_35'):
    os.makedirs(path_results + '/T1/errsm_35')
os.chdir(path_results + '/T1/errsm_35')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_35/37-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 89,73,95,20:88,68,126,19:87,63,156,18:87,61,184,17:86,59,210,16:86,59,235,15:85,60,260,14:85,64,283,13:84,69,305,12:84,73,326,11:84,78,345,10:84,86,363,9:83,96,380,8:83,99,396,7:82,102,411,6:83,104,426,5:83,107,442,4:84,109,460,3:87,117,519,2:88,129,543,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 56 -end 565')
f_crop = open('crop.txt', 'w')
f_crop.write('56,565,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 92,64,0,1:89,59,32,1:87,57,68,1:87,55,93,1:87,53,125,1:85,53,148,1:85,53,170,1:84,55,207,1:84,58,233,1:84,64,261,1:83,71,288,1:83,84,324,1:82,93,353,1:83,97,379,1:84,99,408,1:86,103,442,1:88,121,468,1:89,129,490,1:90,137,508,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject pain_pilot_7
if not os.path.isdir(path_results + '/T1/pain_pilot_7'):
    os.makedirs(path_results + '/T1/pain_pilot_7')
os.chdir(path_results + '/T1/pain_pilot_7')
sct.run('dcm2nii -o . -r N '+folder_data_pain+'/d_sp_pain_pilot7/32-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 94,84,96,20:92,77,128,19:91,71,156,18:88,64,183,17:86,60,209,16:84,56,234,15:83,56,259,14:82,59,284,13:81,64,308,12:80,71,331,11:80,80,351,10:79,90,372,9:78,102,392,8:78,112,411,7:81,117,429,6:82,122,448,5:83,128,466,4:85,132,486,3:85,124,549,2:85,131,571,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 57 -end 590')
f_crop = open('crop.txt', 'w')
f_crop.write('57,590,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 96,80,0,1:94,74,32,1:92,67,72,1:90,61,105,1:87,56,140,1:85,52,173,1:83,51,195,1:82,53,227,1:80,61,259,1:79,73,293,1:78,89,328,1:79,106,364,1:83,117,403,1:84,120,436,1:85,120,472,1:84,125,502,1:85,136,533,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_03
if not os.path.isdir(path_results + '/T1/errsm_03'):
    os.makedirs(path_results + '/T1/errsm_03')
os.chdir(path_results + '/T1/errsm_03')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_03/32-SPINE_all/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 87,83,102,20:88,81,133,19:88,79,162,18:89,76,189,17:90,76,214,16:91,77,239,15:91,78,263,14:92,81,287,13:92,85,309,12:92,91,331,11:92,100,351,10:93,113,371,9:93,125,388,8:92,131,404,7:91,133,419,6:91,132,434,5:91,130,451,4:91,131,468,3:89,133,530,2:89,140,554,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 58 -end 574')
f_crop = open('crop.txt', 'w')
f_crop.write('58,574,0,454')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 87,74,0,1:87,73,24,1:88,73,50,1:89,73,78,1:89,70,115,1:90,69,154,1:91,71,197,1:92,77,240,1:92,86,274,1:93,101,307,1:92,120,344,1:91,124,368,1:91,123,392,1:91,122,426,1:90,124,455,1:89,133,477,1:90,138,496,1:89,149,516,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject FR
if not os.path.isdir(path_results + '/T1/FR'):
    os.makedirs(path_results + '/T1/FR')
os.chdir(path_results + '/T1/FR')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/FR_T080/01_0039_sc-mprage-1mm-3palliers-fov384-comp-sp-13/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 91,102,306,20:91,100,343,19:91,94,376,18:91,89,407,17:90,85,434,16:89,81,464,15:88,80,492,14:87,80,518,13:87,82,543,12:87,87,567,11:87,94,590,10:87,104,611,9:85,114,630,8:84,120,650,7:82,124,669,6:82,125,688,5:82,125,708,4:83,125,729,3:85,117,797,2:86,120,823,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 263 -end 840')
f_crop = open('crop.txt', 'w')
f_crop.write('263,840,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 94,100,0,1:92,93,31,1:91,89,58,1:91,86,100,1:91,81,128,1:91,77,156,1:89,74,191,1:88,73,222,1:87,75,257,1:87,78,288,1:87,87,322,1:86,99,352,1:83,111,385,1:82,117,421,1:82,115,457,1:84,111,493,1:85,111,518,1:86,116,547,1:87,125,575,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject GB
if not os.path.isdir(path_results + '/T1/GB'):
    os.makedirs(path_results + '/T1/GB')
os.chdir(path_results + '/T1/GB')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/GB_T083/01_0029_sc-mprage-1mm-2palliers-fov384-comp-sp-5/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 94,83,93,20:95,82,125,19:95,79,154,18:96,76,181,17:96,73,205,16:96,72,230,15:96,72,253,14:96,76,275,13:96,84,298,12:96,91,319,11:94,100,339,10:92,110,359,9:92,117,378,8:91,121,395,7:91,123,411,6:91,122,428,5:91,120,445,4:90,118,463,3:88,113,527,2:86,115,554,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 55 -end 572')
f_crop = open('crop.txt', 'w')
f_crop.write('55,572,0,449')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 94,73,0,1:94,70,34,1:94,70,62,1:95,69,96,1:96,68,124,1:96,66,152,1:96,66,179,1:96,72,226,1:95,79,251,1:94,87,271,1:92,96,297,1:91,108,327,1:91,114,361,1:91,112,389,1:90,108,421,1:88,108,458,1:86,113,489,1:84,114,517,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm 36
if not os.path.isdir(path_results + '/T1/errsm_36'):
    os.makedirs(path_results + '/T1/errsm_36')
os.chdir(path_results + '/T1/errsm_36')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_36/30-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 87,75,85,20:87,70,118,19:87,66,151,18:88,63,180,17:88,62,208,16:88,60,236,15:89,62,264,14:89,65,289,13:89,72,315,12:89,81,339,11:88,92,361,10:87,103,383,9:86,114,403,8:85,122,419,7:85,127,437,6:85,130,454,5:85,129,473,4:86,128,494,3:87,123,565,2:89,126,589,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 606')
f_crop = open('crop.txt', 'w')
f_crop.write('0,606,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 82,83,0,1:83,81,11,1:83,81,24,1:85,80,38,1:85,77,49,1:86,75,61,1:87,73,74,1:87,71,84,1:87,69,99,1:87,67,111,1:87,64,123,1:87,64,134,1:87,62,143,1:87,60,161,1:87,59,180,1:88,58,197,1:88,57,214,1:88,56,229,1:88,57,245,1:88,58,262,1:89,60,277,1:89,62,293,1:89,68,314,1:89,74,332,1:89,82,351,1:88,90,366,1:88,98,380,1:87,106,395,1:87,113,408,1:86,119,422,1:85,122,437,1:85,124,450,1:85,124,462,1:85,124,476,1:86,123,492,1:86,121,507,1:85,119,521,1:86,118,534,1:87,119,545,1:87,119,555,1:89,120,566,1:89,120,575,1:89,121,583,1:89,123,591,1:89,127,598,1:90,131,606,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_37
if not os.path.isdir(path_results + '/T1/errsm_37'):
    os.makedirs(path_results + '/T1/errsm_37')
os.chdir(path_results + '/T1/errsm_37')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_37/19-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 78,95,190,20:79,89,218,19:79,85,245,18:79,81,270,17:79,79,294,16:79,78,318,15:78,79,341,14:78,82,364,13:79,86,385,12:79,92,405,11:79,98,424,10:80,104,442,9:81,108,457,8:81,111,472,7:81,112,486,6:82,113,500,5:82,113,516,4:83,113,530,3:85,118,583,2:85,124,603,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 100 -end 618')
f_crop = open('crop.txt', 'w')
f_crop.write('100,618,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 75,106,0,1:76,104,13,1:75,102,24,1:74,100,37,1:74,97,47,1:75,94,58,1:76,93,70,1:77,93,77,1:78,91,84,1:78,87,102,1:79,85,119,1:79,83,133,1:79,81,145,1:79,79,161,1:79,77,176,1:79,75,192,1:79,75,206,1:79,75,220,1:79,75,235,1:79,77,251,1:79,79,264,1:79,81,276,1:79,84,289,1:79,87,302,1:79,91,315,1:79,95,329,1:80,99,340,1:81,103,352,1:81,105,365,1:81,107,378,1:82,108,390,1:82,108,402,1:82,108,411,1:83,108,424,1:83,108,436,1:83,108,447,1:84,108,459,1:84,110,471,1:85,112,479,1:85,116,487,1:85,118,495,1:85,121,503,1:86,125,510,1:86,131,518,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_44
if not os.path.isdir(path_results + '/T1/errsm_44'):
    os.makedirs(path_results + '/T1/errsm_44')
os.chdir(path_results + '/T1/errsm_44')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_44/18-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 76,59,83,20:77,56,115,19:76,55,144,18:75,52,172,17:75,51,197,16:76,49,222,15:77,49,247,14:79,53,270,13:82,59,291,12:84,69,312,11:85,81,330,10:86,94,349,9:87,109,368,8:86,118,382,7:86,127,397,6:86,128,411,5:85,128,429,4:84,127,447,3:82,127,518,2:83,128,540,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 559')
f_crop = open('crop.txt', 'w')
f_crop.write('0,559,0,485')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 78,64,0,1:77,60,21,1:76,57,40,1:76,55,56,1:76,53,75,1:77,51,92,1:77,51,109,1:76,50,129,1:76,49,146,1:76,48,162,1:76,47,177,1:75,47,191,1:75,46,208,1:76,45,224,1:77,45,240,1:78,47,258,1:81,50,277,1:82,55,292,1:83,62,307,1:84,70,321,1:85,79,334,1:86,86,345,1:86,94,355,1:86,102,366,1:87,109,376,1:87,115,386,1:86,121,397,1:86,123,411,1:85,123,422,1:84,123,436,1:84,121,449,1:83,119,463,1:83,117,477,1:83,117,489,1:83,117,498,1:83,120,504,1:83,121,512,1:83,122,521,1:83,123,529,1:83,125,538,1:83,128,545,1:83,132,552,1:83,137,559,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject AM
if not os.path.isdir(path_results + '/T1/AM'):
    os.makedirs(path_results + '/T1/AM')
os.chdir(path_results + '/T1/AM')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/AM/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 78,87,42,20:75,84,76,19:74,83,107,18:73,82,137,17:73,81,164,16:73,81,191,15:74,82,217,14:76,85,243,13:79,91,268,12:81,97,290,11:83,106,313,10:84,113,334,9:85,118,355,8:85,121,372,7:86,123,388,6:85,123,403,5:85,122,420,4:86,120,440,3:90,118,508,2:88,119,532,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 550')
f_crop = open('crop.txt', 'w')
f_crop.write('0,550,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 80,88,0,1:79,86,14,1:78,84,28,1:78,82,43,1:76,81,58,1:76,79,73,1:75,79,87,1:74,78,100,1:74,78,115,1:73,77,131,1:73,76,145,1:73,76,159,1:73,76,173,1:73,76,189,1:73,77,204,1:74,78,222,1:76,80,235,1:77,82,248,1:78,85,263,1:80,89,279,1:81,93,291,1:82,99,306,1:83,104,320,1:84,108,333,1:84,111,347,1:85,114,360,1:85,117,373,1:86,119,386,1:86,119,398,1:86,118,412,1:85,117,425,1:86,116,438,1:87,114,451,1:88,113,463,1:89,112,475,1:90,110,486,1:90,111,495,1:90,113,503,1:89,114,511,1:89,114,519,1:89,115,527,1:88,116,535,1:87,118,541,1:87,125,550,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject HB
if not os.path.isdir(path_results + '/T1/HB'):
    os.makedirs(path_results + '/T1/HB')
os.chdir(path_results + '/T1/HB')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/HB/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-29/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 90,88,25,20:89,84,62,19:88,79,96,18:88,78,127,17:88,78,156,16:87,79,185,15:87,81,214,14:85,86,241,13:84,93,268,12:82,101,292,11:80,110,314,10:79,118,336,9:77,124,358,8:76,128,376,7:76,127,393,6:76,128,411,5:76,127,430,4:77,126,450,3:77,120,517,2:77,120,538,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 556')
f_crop = open('crop.txt', 'w')
f_crop.write('0,556,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 91,91,0,1:90,85,24,1:89,82,43,1:89,80,58,1:89,79,75,1:88,76,93,1:88,75,111,1:88,74,129,1:88,74,147,1:88,74,166,1:87,75,187,1:87,76,206,1:86,79,227,1:85,82,246,1:84,87,264,1:82,93,283,1:81,101,304,1:80,109,321,1:79,114,337,1:77,119,355,1:76,123,376,1:76,123,391,1:76,124,405,1:75,124,422,1:76,123,438,1:77,121,453,1:78,119,467,1:78,116,481,1:78,116,494,1:78,117,504,1:78,117,513,1:78,118,527,1:78,117,537,1:77,120,547,1:76,124,556,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject PA
if not os.path.isdir(path_results + '/T1/PA'):
    os.makedirs(path_results + '/T1/PA')
os.chdir(path_results + '/T1/PA')
sct.run('dcm2nii -o . -r N '+folder_data_marseille+'/PA/01_0034_sc-mprage-1mm-2palliers-fov384-comp-sp-5/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 81,76,89,20:80,68,121,19:79,63,147,18:80,59,175,17:79,58,200,16:78,57,224,15:79,60,248,14:79,64,272,13:80,71,294,12:80,79,315,11:80,87,334,10:79,98,355,9:80,107,370,8:80,115,385,7:80,119,398,6:81,122,412,5:82,122,428,4:83,120,446,3:84,111,512,2:83,110,533,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 550')
f_crop = open('crop.txt', 'w')
f_crop.write('0,550,0,max')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 85,93,0,1:84,89,18,1:83,85,34,1:82,82,47,1:81,76,63,1:81,73,82,1:80,68,103,1:80,64,118,1:80,62,131,1:79,59,150,1:80,56,165,1:79,54,184,1:79,54,201,1:79,54,216,1:79,55,237,1:79,57,256,1:79,60,272,1:79,65,289,1:80,71,305,1:80,78,322,1:80,84,335,1:79,91,349,1:79,98,362,1:79,106,375,1:80,112,389,1:80,116,403,1:81,118,417,1:82,118,432,1:83,115,450,1:83,112,467,1:84,109,481,1:83,107,496,1:83,107,507,1:83,106,519,1:82,107,532,1:82,108,540,1:83,113,550,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')


# Preprocessing for subject errsm_43
if not os.path.isdir(path_results + '/T1/errsm_43'):
    os.makedirs(path_results + '/T1/errsm_43')
os.chdir(path_results + '/T1/errsm_43')
sct.run('dcm2nii -o . -r N '+folder_data_errsm+'/errsm_43/22-SPINE_T1/echo_2.09/*.dcm')
sct.run('mv *.nii.gz data.nii.gz')
sct.run('sct_image -i data.nii.gz -setorient RPI')
sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x 76,75,75,20:76,73,109,19:76,69,140,18:76,65,167,17:76,63,194,16:76,62,219,15:76,61,245,14:76,63,269,13:78,68,292,12:79,74,314,11:80,82,336,10:82,90,356,9:83,100,375,8:84,107,392,7:87,111,407,6:88,114,423,5:89,116,441,4:90,116,458,3:88,124,520,2:87,129,542,1')
sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start 0 -end 558')
f_crop = open('crop.txt', 'w')
f_crop.write('0,558,0,494')
f_crop.close()
sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x 78,77,0,1:78,76,18,1:78,75,33,1:78,73,56,1:77,71,77,1:76,70,95,1:77,68,110,1:76,66,127,1:76,64,145,1:76,62,163,1:76,60,180,1:76,59,196,1:76,58,212,1:76,57,229,1:76,58,245,1:76,59,260,1:77,61,276,1:77,64,292,1:78,68,306,1:80,73,321,1:81,79,339,1:82,87,358,1:83,93,371,1:84,100,385,1:86,104,399,1:87,108,415,1:88,110,431,1:89,111,445,1:90,111,462,1:91,110,476,1:90,111,489,1:90,114,501,1:88,117,512,1:87,121,522,1:87,123,532,1:87,125,541,1:86,129,549,1:85,135,558,1')
os.remove('data.nii.gz')
os.remove('data_RPI.nii.gz')
os.remove('data_RPI_crop.nii.gz')
os.chdir('../..')

"""
