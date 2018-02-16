#!/usr/bin/env python


import sys, io, os

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, 'scripts'))

import sct_utils as sct



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
                       ['MT','/Volumes/data_shared/marseille/MT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/MT/01_0100_t2composing'],['T045', '/Volumes/data_shared/marseille/T045/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/T045/01_0101_t2-3d-composing'],\
                       ['T047','/Volumes/data_shared/marseille/T047/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/T047/01_0100_t2-3d-composing'],['VC','/Volumes/data_shared/marseille/VC/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/VC/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-113'],\
                       ['VG','/Volumes/data_shared/marseille/VG/T1/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15','/Volumes/data_shared/marseille/VG/T2/01_0024_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-11'],['VP','/Volumes/data_shared/marseille/VP/01_0011_sc-mprage-1mm-2palliers-fov384-comp-sp-25','/Volumes/data_shared/marseille/VP/01_0100_space-compo'],\
                       ['pain_pilot_1','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/24-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/25-SPINE'],['pain_pilot_2','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/13-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/30-SPINE_T2'],\
                       ['pain_pilot_4','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/33-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/32-SPINE_T2'],['TM', '/Volumes/data_shared/marseille/TM_T057c/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/TM_T057c/01_0105_t2-composing'],\
                       ['errsm_20', '/Volumes/data_shared/montreal_criugm/errsm_20/12-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_20/34-SPINE_T2'],['pain_pilot_3','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/16-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/31-SPINE_T2'],\
                       ['errsm_34','/Volumes/data_shared/montreal_criugm/errsm_34/41-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/errsm_34/40-SPINE_T2'],['errsm_35','/Volumes/data_shared/montreal_criugm/errsm_35/37-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/errsm_35/38-SPINE_T2'],\
                       ['pain_pilot_7', '/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot7/32-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot7/33-SPINE_T2'], ['errsm_03', '/Volumes/data_shared/montreal_criugm/errsm_03/32-SPINE_all/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_03/38-SPINE_all_space'],\
                       ['FR', '/Volumes/data_shared/marseille/FR_T080/01_0039_sc-mprage-1mm-3palliers-fov384-comp-sp-13', '/Volumes/data_shared/marseille/FR_T080/01_0104_spine2'],\
                       ['GB', '/Volumes/data_shared/marseille/GB_T083/01_0029_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/GB_T083/01_0033_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7']]
print len(SUBJECTS_LIST_total)

SUBJECTS_LIST = SUBJECTS_LIST_total
path_i = '/Users/tamag/data/data_template/info/template_subjects'
contrast ='T1'
path_result = ''
os.chdir(path_i)

f_name='all_crop_'+contrast+'.txt'
f = io.open(path_i+'/'+f_name, 'w')
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
        #f.write("""os.makedirs(path_results + '/"""+contrast+"""/""" + subject+"""')\nos.chdir(path_results + '/"""+contrast+"""/""" + subject+"""')\n""")
        f.write("""if not os.path.isdir(path_results + '/"""+contrast+"""/""" + subject+"""'):\n\tos.makedirs(path_results + '/"""+contrast+"""/""" + subject+"""')\nos.chdir(path_results + '/"""+contrast+"""/""" + subject+"""')\n""")
        f.write("""sct.run('dcm2nii -o . -r N """+ SUBJECTS_LIST[i][1] + """/*.dcm')\nsct.run('mv *.nii.gz data.nii.gz')\nsct.run('sct_orientation -i data.nii.gz -s RPI')\n""")
        #f.write("""sct.run('cp """+ path_i +"""/"""+contrast+"""/"""+subject+"""/image_RPI.nii.gz data_RPI.nii.gz')\n""")
        if 'labels_vertebral.nii.gz' in list_d_2:
            #f.write("""sct.run('sct_crop_image -i data_RPI.nii.gz -dim 2 -start 2 -end 1 -b 0 -o labels_vertebral.nii.gz')\n""")
            status, output = sct.run('sct_label_utils -i ' + path_i+'/' + contrast+ '/' + subject + '/labels_vertebral.nii.gz -t display-voxel')
            nb = output.find('notation')
            int_nb = nb + 10
            labels = output[int_nb:]
            f.write("""sct.run('sct_label_utils -i data_RPI.nii.gz -o labels_vertebral.nii.gz -t create -x """+labels+"""')\n""")
        if ymin_anatomic == None and ymax_anatomic == None:
            f.write("""sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 2 -start  """ +zmin_anatomic+""" -end """+zmax_anatomic+""" ')\nf_crop = open('crop.txt', 'w')\nf_crop.write('"""+zmin_anatomic+','+zmax_anatomic+','+zmin_seg+','+zmax_seg+"""')\nf_crop.close()\n""")
        else: f.write("""sct.run('sct_crop_image -i data_RPI.nii.gz -o data_RPI_crop.nii.gz -dim 1,2 -start """ +ymin_anatomic +','+zmin_anatomic+ ' -end ' + ymax_anatomic+','+zmax_anatomic+ """')\nf_crop = open('crop.txt', 'w')\nf_crop.write('"""+zmin_anatomic+','+zmax_anatomic+','+zmin_seg+','+zmax_seg+','+ymin_anatomic+','+ymax_anatomic+"""')\nf_crop.close()\n""")

        if 'centerline_propseg_RPI.nii.gz' in list_d_2:
            status_1, output_1 = sct.run('sct_label_utils -i ' + path_i+'/' + contrast+ '/' + subject + '/centerline_propseg_RPI.nii.gz -t display-voxel')
            nb_1 = output_1.find('notation')
            int_nb_1 = nb_1 + 10
            labels_1 = output_1[int_nb_1:]
            #f.write("""sct.run('sct_crop_image -i data_RPI_crop.nii.gz -dim 2 -start 2 -end 1 -b 0 -o centerline_propseg_RPI.nii.gz')\nsct.run('sct_label_utils -i centerline_propseg_RPI.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x """+labels_1+"""')\n""")
            f.write("""sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o centerline_propseg_RPI.nii.gz -t create -x """+labels_1+"""')\n""")
        if 'labels_updown.nii.gz' in list_d_2:
            status_2, output_2 = sct.run('sct_label_utils -i ' + path_i+'/' + contrast+ '/' + subject + '/labels_updown.nii.gz -t display-voxel')
            nb_2 = output_2.find('notation')
            int_nb_2 = nb_2 + 10
            labels_2 = output_2[int_nb_2:]
            #f.write("""sct.run('sct_crop_image -i data_RPI_crop.nii.gz -dim 2 -start 2 -end 1 -b 0 -o labels_updown.nii.gz')\nsct.run('sct_label_utils -i labels_updown.nii.gz -o labels_updown.nii.gz -t create -x """+labels_2+"""')\n""")
            f.write("""sct.run('sct_label_utils -i data_RPI_crop.nii.gz -o labels_updown.nii.gz -t create -x """+labels_2+"""')\n""")
        f.write("""os.remove('data.nii.gz')\nos.remove('data_RPI.nii.gz')\nos.remove('data_RPI_crop.nii.gz')\nos.chdir('../..')\n""")

f.close()
