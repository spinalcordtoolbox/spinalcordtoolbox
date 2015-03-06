#!/usr/bin/env python
#########################################################################################
#
# A pipeline to register a lot of data
#
# Data should be organized as follow :
#
# data/
# ......subject_01/
# ......subject_02/
# .................t1/
# .........................subject_02_anything_t1.nii.gz
# .........................some_landmarks_of_vertebral_levels.nii.gz
# .........................subject_01_manual_segmentation_t1.nii.gz
# .................t2/
# .........................subject_01_anything_t2.nii.gz
# .........................some_landmarks_of_vertebral_levels.nii.gz
# .........................subject_01_manual_segmentation_t2.nii.gz
# .................t2star/
# .........................subject_01_anything_t2star.nii.gz
# .........................subject_01_manual_segmentation_t2star.nii.gz
# ......subject_03/
#          .
#          .
#          .
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont
# Modified: 2015-03-02
#
# About the license: see the file LICENSE.TXT
#########################################################################################
from msct_parser import Parser
import sys
import sct_utils as sct
import os
import time


# DEFAULT PARAMETERS
class Param:
    ## The constructor
    def __init__(self):
        self.debug = 0
        self.verbose = 1
        self.path_data = ''
        self.t = 't2'


class Subject:
    def __init__(self, dir, dir_t1, name_t1, name_t1_seg, name_t1_ref, name_landmarks_t1, dir_t2, name_t2, name_t2_seg, name_t2_ref, name_landmarks_t2, dir_t2star, name_t2star, name_t2star_seg, name_t2star_ref):
        self.dir = dir
        #data t1
        self.dir_t1 = dir_t1
        self.name_t1 = name_t1
        self.name_t1_seg = name_t1_seg
        self.name_t1_ref = name_t1_ref
        self.name_landmarks_t1 = name_landmarks_t1
        #data t2
        self.dir_t2 = dir_t2
        self.name_t2 = name_t2
        self.name_t2_ref = name_t2_ref
        self.name_t2_seg = name_t2_seg
        self.name_landmarks_t2 = name_landmarks_t2
        #data t2star
        self.dir_t2star = dir_t2star
        self.name_t2star = name_t2star
        self.name_t2star_seg = name_t2star_seg
        self.name_t2star_ref = name_t2star_ref


class Pipeline:
    ## The constructor
    def __init__(self, path_data, t, seg=1, seg_params={}, reg_template=False, reg_template_params=None ,seg_t2star=False,  seg_t2star_params={},reg_multimodal=False, reg_multimodal_params=None, dice=False, dice_on=[]):
        self.path_data = path_data
        os.chdir(self.path_data)
        print sct.run('ls -a', verbose=1)
        self.t = t
        #options
        self.seg = seg
        self.seg_params = seg_params
        self.reg_template = reg_template
        self.reg_template_params = reg_template_params
        self.seg_t2star = seg_t2star
        self.seg_t2star_params = seg_t2star_params
        self.reg_multimodal = reg_multimodal
        self.reg_multimodal_params = reg_multimodal_params
        self.dice = dice
        self.dice_on = dice_on
        #generating data
        self.data = self.generate_data_list()

    #construction of the dataset list
    def generate_data_list(self):
        data = []
        #each directory in "path_data" is a subject
        for subject_dir in os.listdir('./'):
            if not os.path.isdir(subject_dir):
                sct.printv("WARNING : Found a file, data should be organized in folders ... ", verbose=1, type="warning")
            else:
                os.chdir(subject_dir)

                dir_t1 = ''
                dir_t2 = ''
                dir_t2star = ''
                print "\n\n---------------------------------------------------------------------------"
                print "SUBJECT : ", subject_dir
                print "---> containing : ", os.listdir('./')
                for dir in os.listdir('./'):
                    if 't1' in dir.lower():
                        dir_t1 = dir
                    if 't2star' in dir.lower():
                        dir_t2star = dir
                    elif 't2' in dir.lower():
                        dir_t2 = dir

                name_t1 = ''
                name_t1_seg = ''
                name_t1_ref = ''
                name_landmarks_t1 = ''
                name_t2 = ''
                name_t2_seg = ''
                name_t2_ref = ''
                name_landmarks_t2 = ''
                name_t2star = ''
                name_t2star_seg = ''
                name_t2star_ref = ''

                #T1 data
                if dir_t1 is not '':
                    for file in os.listdir(dir_t1):
                        if "t1.nii" in file.lower() and self.check_nii_gz(file) and 'warp' not in file:
                            name_t1 = file
                        if "landmark" in file.lower() and self.check_nii_gz(file):
                            name_landmarks_t1 = file
                        if 'manual_seg' in file.lower() or 'manualseg' in file.lower() or 'ref' in file.lower():
                            if self.check_nii_gz(file):
                                name_t1_ref = file
                        elif not self.seg:
                            if "seg" in file.lower() and self.check_nii_gz(file):
                                name_t1_seg = file
                    if name_t1 == '':
                        sct.printv("WARNING: could not find t1 file in folder " + dir_t1, verbose=1, type='warning')
                    if name_landmarks_t1 == '':
                        sct.printv("WARNING: could not find landmarks file in folder " + dir_t1, verbose=1, type='warning')
                    if name_t1_ref == '':
                        sct.printv("WARNING: could not find t1 reference segmentation file in folder " + dir_t1, verbose=1, type='warning')
                    if not self.seg and name_t1_seg == '':
                        sct.printv("WARNING: could not find t1 segmentation file in folder " + dir_t1, verbose=1, type='warning')
                else:
                    sct.printv('WARNING: no t1 folder', 1, 'warning')

                #T2 data
                if dir_t2 is not '':
                    for file in os.listdir(dir_t2):
                        if "t2.nii" in file.lower() and self.check_nii_gz(file) and 'warp' not in file:
                            name_t2 = file
                        if "landmark" in file.lower() and self.check_nii_gz(file):
                            name_landmarks_t2 = file
                        if 'manual_seg' in file.lower() or 'manualseg' in file.lower() or 'ref' in file.lower():
                            if self.check_nii_gz(file):
                                name_t2_ref = file
                        elif not self.seg:
                            if "seg" in file.lower() and 'mask' not in file and 'warp' not in file and self.check_nii_gz(file):
                                name_t2_seg = file
                    if name_t2 == '':
                        sct.printv("WARNING: could not find t2 file in folder " + dir_t2, verbose=1, type='warning')
                    if name_landmarks_t2 == '':
                        sct.printv("WARNING: could not find landmarks file in folder " + dir_t2, verbose=1, type='warning')
                    if name_t2_ref == '':
                        sct.printv("WARNING: could not find t2 reference segmentation file in folder " + dir_t2, verbose=1, type='warning')
                    if not self.seg and name_t2_seg == '':
                        sct.printv("WARNING: could not find t2 segmentation file in folder " + dir_t2, verbose=1, type='warning')
                else:
                    sct.printv('WARNING: no t2 folder', 1, 'warning')

                #T2star data
                if dir_t2star is not '':
                    for file in os.listdir(dir_t2star):
                        if 't2star.nii' in file and self.check_nii_gz(file) and 'warp' not in file:
                            name_t2star = file
                        if 'manual_seg' in file.lower() or 'manualseg' in file.lower() or 'ref' in file.lower():
                            if self.check_nii_gz(file):
                                name_t2star_ref = file
                        elif not self.seg_t2star:
                            if "seg" in file.lower() and 'mask' not in file and 'warp' not in file and self.check_nii_gz(file):
                                name_t2star_seg = file
                    if name_t2star == '':
                        sct.printv("WARNING: could not find t2star file in folder " + dir_t2star, verbose=1, type='warning')
                    if name_t2star_ref == '':
                        sct.printv("WARNING: could not find t2star reference segmentation file in folder " + dir_t2star, verbose=1, type='warning')
                    if not self.seg_t2star and name_t2star_seg == '':
                        sct.printv("WARNING: could not find t2star segmentation file in folder " + dir_t2star, verbose=1, type='warning')
                else:
                    sct.printv('WARNING: no t2star folder', 1, 'warning')

                #creation of a subject for each directory
                data.append(Subject(subject_dir, dir_t1, name_t1, name_t1_seg, name_t1_ref, name_landmarks_t1, dir_t2, name_t2, name_t2_seg, name_t2_ref, name_landmarks_t2, dir_t2star, name_t2star, name_t2star_seg, name_t2star_ref))
                sct.printv('\nFiles for subject ' + subject_dir +
                           '\nT1 : ' + name_t1 +
                           '\nT1 seg : ' + name_t1_seg +
                           '\nT1 ref : ' + name_t1_ref +
                           '\nT1 landmarks : ' + name_landmarks_t1 +
                           '\nT2 : ' + name_t2 +
                           '\nT2 seg : ' + name_t2_seg +
                           '\nT2 ref : ' + name_t2_ref +
                           '\nT2 landmarks : ' + name_landmarks_t2 +
                           '\nT2star :' + name_t2star +
                           '\nT2star_seg : ' + name_t2star_seg +
                           '\nT2star ref : ' + name_t2star_ref, 1, "normal")
                os.chdir('..')
        if data == []:
            sct.printv("ERROR : Data should be organized in folders ... ", verbose=1, type="error")

        return data

    #Check if file extension is .nii.gz, if yes, return True, else, print a warning message and return False
    def check_nii_gz(self, file):
        file_path, file_name, file_ext = sct.extract_fname(file)
        if file_ext != '.nii.gz':
            sct.printv('WARNING: File ' + file_name + ' should be .nii.gz instead of ' + file_ext + ' ...', 1, 'warning')
            return False
        else:
            return True

    #segmentation using sct_propseg for all the subjects of the dataset
    def segmentation(self, t, init=0.5, up=None, down=None, centerline=None, init_mask=None, radius=None):
        for subject in self.data:
            os.chdir(subject.dir)
            if t == 't1':
                path = subject.dir_t1
                name = subject.name_t1
            elif t == 't2':
                path = subject.dir_t2
                name = subject.name_t2
            elif t == 't2star':
                path = subject.dir_t2star
                name = subject.name_t2star
                t = 't2'
                for file in os.listdir(path):
                    if 'mask' in file:
                        self.seg_t2star_params['init-mask'] = file
                        init_mask = file

            if name is not '':
                os.chdir(path)
                cmd = 'sct_propseg -i ' + name + ' -o ./ -t ' + t + ' -init ' + str(init)
                if up is not None:
                    cmd = cmd + " -up " + str(up)
                if down is not None:
                    cmd = cmd + " -down " + str(down)
                if centerline is not None:
                    cmd = cmd + " -init-centerline " + centerline
                if init_mask is not None:
                    cmd = cmd + " -init-mask " + init_mask
                if radius is not None:
                    cmd = cmd + " -radius " + str(radius)

                sct.printv("\nDoing segmentation on " + subject.dir + '/' + path + "/" + name + " using sct_propseg ...", verbose=1, type="normal")
                sct.runProcess(cmd)
                for file in os.listdir('./'):
                            if "_seg.nii" in file:
                                name_seg = file
                                break
                if t == 't1':
                    subject.name_t1_seg = name_seg
                if t == 't2':
                    subject.name_t2_seg = name_seg
                if t == 't2star':
                    subject.name_t2star_seg = name_seg
                os.chdir('..')
            else:
                sct.printv("WARNING: no file to do segmentation on in folder " + path, verbose=1, type='warning')
            os.chdir('..')

    #register T2 to template for all the subjects of the dataset
    def register_warp_to_template(self, t):
        for subject in self.data:
            os.chdir(subject.dir)
            if t == 't1':
                path = subject.dir_t1
                name = subject.name_t1
                name_seg = subject.name_t1_seg
            elif t == 't2':
                path = subject.dir_t2
                name = subject.name_t2
                name_seg = subject.name_t2_seg

            os.chdir(path)

            try:
                cmd_register = 'sct_register_to_template -i ' + name + ' -m ' + name_seg + ' -l ' + subject.name_landmarks_t2

                if self.reg_template_params is not None :
                    cmd_register = cmd_register + " -p " + str(self.reg_template_params)

                sct.printv("\nDoing registration to template on " + subject.dir + '/' + path + '/' + subject.name_t2 +" using sct_register_to_template ...", verbose=1, type="normal")
                sct.runProcess(cmd_register)
            except Exception, e:
                    sct.printv('WARNING: AN ERROR OCCURRED WHEN TRYING TO REGISTER TEMPLATE TO' + t.upper() + ' : ', 1, 'warning')
                    print e
                    sct.printv('Continuing program ...', 1, 'warning')
            else:
                try:
                    #TODO change name of the warping field by a Subject attribute ??
                    cmd_warp = 'sct_warp_template -d ' + name + ' -w warp_template2anat.nii.gz'

                    sct.printv("\nWarping Template to T2 on " + subject.dir + '/' + path + "/" + name+" using sct_warp_template ...", verbose=1, type="normal")
                    sct.runProcess(cmd_warp)
                except Exception, e:
                    sct.printv('WARNING: AN ERROR OCCURRED WHEN TRYING TO WARP TEMPLATE TO ' + t.upper() + ' : ', 1, 'warning')
                    print e
                    sct.printv('Continuing program ...', 1, 'warning')
            os.chdir('../..')


    def register_warp_multimodal(self, t, src='template2anat.nii.gz'):
        for subject in self.data:
            os.chdir(subject.dir)
            if subject.path_t2star is not '':
                if t == 't1':
                    path_anat = subject.dir_t1
                elif t == 't2':
                    path_anat = subject.dir_t2


                try:
                    os.chdir(subject.dir_t2star)

                    #Register
                    cmd_register = 'sct_register_multimodal -i ' + path_anat + '/' + src + ' -d ' + subject.name_t2star + \
                        ' -s ' + path_anat + '/label/template/MNI-Poly-AMU_cord.nii.gz -t ' + subject.name_t2star_seg
                    if self.reg_multimodal_params is not None :
                        cmd_register = cmd_register + " -p " + str(self.reg_multimodal_params)

                    sct.printv("\nDoing registration template2anat to " + + subject.dir + '/' + subject.dir_t2star + "/" + subject.name_t2star +" using sct_register_multimodal ...", verbose=1, type="normal")
                    #sct.runProcess(cmd_register)
                    sct.run(cmd_register)
                except Exception, e:
                    sct.printv('WARNING: AN ERROR OCCURRED WHEN TRYING TO REGISTER TEMPLATE TO T2STAR : ', 1, 'warning')
                    print e
                    sct.printv('Continuing program ...', 1, 'warning')
                else:
                    try:
                        #concatenate transformations
                        src_name = sct.extract_fname(src)[1]
                        cmd_concat = 'sct_concat_transfo -w ' + path_anat + '/warp_template2anat.nii.gz,warp_src2dest.nii.gz -d ' + subject.name_t2star + ' -o warp_' + src_name + '2t2star.nii.gz'
                        sct.printv("\nConcatenate transformations using sct_concat_transfo ...", verbose=1, type="normal")
                        sct.runProcess(cmd_concat)
                    except Exception, e:
                        sct.printv('WARNING: AN ERROR OCCURRED WHEN TRYING TO CONCATENATE TRANSFORMATIONS : ', 1, 'warning')
                        print e
                        sct.printv('Continuing program ...', 1, 'warning')

                    else:
                        try:
                            #warp template to t2star
                            cmd_warp = 'sct_warp_template -d ' + subject.name_t2star + ' -w warp_' + src_name + '2t2star.nii.gz'

                            sct.printv("\nWarping Template to T2star on " + + subject.dir + '/' + subject.dir_t2star +"/" + subject.name_t2star +" using sct_warp_template ...", verbose=1, type="normal")
                            sct.runProcess(cmd_warp)
                        except Exception, e:
                            sct.printv('WARNING: AN ERROR OCCURRED WHEN TRYING TO WARP TEMPLATE TO T2STAR : ', 1, 'warning')
                            print e
                            sct.printv('Continuing program ...', 1, 'warning')
                os.chdir('..')

            else:
                sct.printv('WARNING: no t2star folder, did not apply sct_register_multimodal to subject ' + subject.dir, 1, 'warning')
            os.chdir('..')

    #compute the dice coefficient of all subjects in the data folder for the type of image in input (t1, t2 or t2star)
    def compute_dice(self, t, type_res):
        dice_dictionary = {}
        sum_dice = 0
        n_subjects = 0
        name_file = 'dice_coeff_' + type_res + '_' + t + '.txt'
        res_file = open(name_file, 'w')
        if type_res == 'seg':
            name_res = ' segmentation'
        elif type_res == 'reg':
            name_res = ' registration'
        else:
            name_res = ''
            sct.printv('WARNING: inputed type of results to compute dice coefficient on is unrecognized ...', 1, 'warning')
        res_file.write('Dice coefficient for ' + t.upper() + name_res + ' data in file ' + self.path_data + '\n')
        for subject in self.data:
            os.chdir(subject.dir)
            if t == 't1':
                path = subject.dir_t1
                ref = subject.name_t1_ref
                if type_res == 'seg':
                    res = subject.name_t1_seg
            elif t == 't2':
                path = subject.dir_t2
                ref = subject.name_t2_ref
                if type_res == 'seg':
                    res = subject.name_t2_seg
            elif t == 't2star':
                path = subject.dir_t2star
                ref = subject.name_t2star_ref
                if type_res == 'seg':
                    res = subject.name_t2star_seg

            if type_res == 'reg' :
                res = 'label/template/MNI-Poly-AMU_cord.nii.gz'

            if ref is not '':
                try:
                    os.chdir(path)
                    cmd_dice = 'sct_dice_coefficient ' + res + ' ' + ref
                    sct.printv("\nComputing the dice coefficient for the " + t + " image(s) of subject " + subject.dir + "  ...", verbose=1, type="normal")
                    status, output = sct.runProcess(cmd_dice)

                    dice_dictionary[subject.dir] = float(output.split(' ')[-1][:-2])
                    res_file.write(subject.dir + ' : ' + str(dice_dictionary[subject.dir]) + '\n')
                    n_subjects += 1
                    sum_dice += dice_dictionary[subject.dir]

                except Exception, e:
                    sct.printv('WARNING: AN ERROR OCCURRED WHEN TRYING TO COMPUTE DICE BETWEEN ' + subject.dir + '/' + t + '/' + ref + ' AND ' + subject.dir + '/' + t + '/' + res + ' :',1,'warning')
                    print e
                    sct.printv('Continuing program ...\n', 1, 'warning')
                finally:
                    os.chdir('..')

            else:
                sct.printv('WARNING: no reference segmentation for the ' + t + ' image of subject ' + subject.dir + ', did not compute dice.', verbose=1, type='warning')
            os.chdir('..')
        mean_dice = sum_dice / n_subjects
        res_file.write('Mean Dice coefficient : ' + str(mean_dice) + '\n')
        res_file.close()
        return dice_dictionary, mean_dice

    def write_infos(self):
        time_now = time.localtime()
        now = str(time_now.tm_year) + '-' + str(time_now.tm_mon) + '-' + str(time_now.tm_mday) + '_' + str(time_now.tm_hour) + 'H' + str(time_now.tm_min)
        name_infos = now + '_parameters_info.txt'
        infos = open(name_infos, 'w')
        infos.write('Data treated using the sct_register_pipeline \n'
                    'Path to data : ' + self.path_data + '\n'
                    'Type of anatomic data : ' + self.t + '\n')
        infos.write('\n\nTreatments applied : \n'
                    ' - Segmentation of anatomic image : ' + self.yesNo(self.seg) + '\n')
        if self.seg :
            infos.write('   --> Parameters : ' + str(self.seg_params) + '\n')
        infos.write(' - Registration of anat to template and warping of template to anat : ' + self.yesNo(self.reg_template) + '\n')
        if self.reg_template:
            infos.write('   --> Parameters : ' + str(self.reg_template_params) + '\n')
        infos.write(' - Segmentation of T2star image : ' + self.yesNo(self.seg_t2star) + '\n')
        if self.seg_t2star:
            infos.write('   --> Parameters : ' + str(self.seg_t2star_params) + '\n')
        infos.write(' - Registration of T2star to Template2anat and warping of Template2anat to T2star : ' + self.yesNo(self.reg_multimodal) + '\n')
        if self.reg_multimodal:
            infos.write('   --> Parameters : ' + str(self.reg_multimodal_params) + '\n')
        if self.dice:
            infos.write(' - Computed Dice coefficient\n')
        infos.close()

    def yesNo(self, bool):
        if bool:
            return 'YES'
        else:
            return 'NO'

    #compute each step of the pipeline
    def compute(self):
        #segmentation of the anatomical image
        if self.seg:
            if self.t == 'both' or self.t == 'b':
                self.segmentation('t1', init=self.seg_params['init'], up=self.seg_params['up'], down=self.seg_params['down'], centerline=self.seg_params['centerline'], init_mask=self.seg_params['init-mask'], radius=self.seg_params['radius'])
                self.segmentation('t2', init=self.seg_params['init'], up=self.seg_params['up'], down=self.seg_params['down'], centerline=self.seg_params['centerline'], init_mask=self.seg_params['init-mask'], radius=self.seg_params['radius'])
                if self.dice:
                    self.compute_dice('t1', 'seg')
                    self.compute_dice('t2', 'seg')

            else:
                self.segmentation(self.t, init=self.seg_params['init'], up=self.seg_params['up'], down=self.seg_params['down'], centerline=self.seg_params['centerline'], init_mask=self.seg_params['init-mask'], radius=self.seg_params['radius'])
                if self.dice:
                    self.compute_dice(self.t, 'seg')

        #registration of anatomical image to template and warping of template to anat
        if self.reg_template:
            if self.t == 'both' or self.t == 'b':
                self.register_warp_to_template('t1')
                self.register_warp_to_template('t2')
                if self.dice:
                    self.compute_dice('t1', 'reg')
                    self.compute_dice('t2', 'reg')
            else:
                self.register_warp_to_template(self.t)
                if self.dice:
                    self.compute_dice(self.t, 'reg')

        #segmentation of the T2star image
        if self.seg_t2star:
            self.segmentation('t2star', init=self.seg_t2star_params['init'], up=self.seg_t2star_params['up'], down=self.seg_t2star_params['down'], centerline=self.seg_t2star_params['centerline'], init_mask=self.seg_t2star_params['init-mask'], radius=self.seg_t2star_params['radius'])
            if self.dice:
                self.compute_dice('t2star', 'seg')

        #registration of T2star on Template2anat image and warping of template2anat to T2star
        if self.reg_multimodal:
            if self.t == 'both' or self.t == 'b':
                self.register_warp_multimodal('t1')
                self.register_warp_multimodal('t2')
            else:
                self.register_warp_multimodal(self.t)
            if self.dice:
                self.compute_dice('t2star', 'reg')
        #compute dice on other results
        for arg in self.dice_on:
            type_res,type_image = arg.split(":")
            self.compute_dice(type_image,type_res)


        self.write_infos()









#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    begin = time.time()
    # initialize parameters
    param = Param()
    param_default = Param()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Pipeline to do Template --> Anatomical image registration on a lot of data\n '
                                 'Your data should be organized with one folder per subject\n'
                                 'each subject folder should contain t1, t2 and more folders with the appropriate data\n'
                                 't1 and t2 files should be named "your_prefix_t1.nii.gz" and "your_prefix_t2.nii.gz"\n')
    parser.add_option(name="-data",
                      type_value="folder",
                      description="Path to the data you want to register template on",
                      mandatory=True,
                      example='/home/jdoe/data_testing')
    parser.add_option(name="-t",
                      type_value="str",
                      description='Type of anatomic data : t1, t2 or both',
                      mandatory=True,
                      example='t2')
    parser.add_option(name="-seg",
                      description='Segmentation of the spinal cord on anatomic data using sct_propseg\n                                          '
                                  'If not used, a segmentation file should be provided in the anatomic data folder with a name containing "seg"',
                      mandatory=False)
    parser.add_option(name="-seg-params",
                      type_value=[[','], 'str'],
                      description='Parameters for sct_propseg (only if the -seg flag is used)\n                                          '
                                  'Parameters available : \n                                          '
                                  '                     - init : int, axial slice where the propagation starts, default is middle axial slice\n                                          '
                                  '                     - down : int, down limit of the propagation, default is 0\n                                          '
                                  '                     - up : int, up limit of the propagation, default is the higher slice of the image\n                                          '
                                  '                     - centerline : filename of centerline to use for the propagation, format .txt or .nii, see file structure in documentation\n                                          '
                                  '                     - init-mask : string, mask containing three center of the spinal cord, used to initiate the propagation\n                                          '
                                  '                     - radius : double, approximate radius of the spinal cord, default is 4 mm\n                                          ',
                      mandatory=False,
                      example='init:0.5,up:8,centerline:my_centerline.nii.gz,radius:6')
    parser.add_option(name="-reg-template",
                      description='Registration of anatomic data to template and warping template on anatomic data,\n                                          '
                                  'WARNING : OPTION NOT READY TO USE YET FOR T1 DATA \n                                          '
                                  'A landmarks file containing appropriate labels at spinal cord center should be in the anatomic data folder \n                                          '
                                  'See: http://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels/',
                      mandatory=False)
    parser.add_option(name="-reg-template-params",
                      type_value='str',
                      description='Parameters for the registration of anatomic data to template (only if the -reg-template flag is used),\n                                          '
                                  'ALL ITEMS MUST BE LISTED IN ORDER. Separate with comma WITHOUT WHITESPACE IN BETWEEN.'
                                  'Default=5,SyN,0.5,MI'
                                  '1) number of iterations for last stage.'
                                  '2) algo: {SyN, BSplineSyN}'
                                  '3) gradient step. The larger the more deformation.'
                                  '4) metric: {MI,MeanSquares}.'
                                  'If you find very large deformations, switching to MeanSquares can help.',
                      mandatory=False,
                      example='10,SyN,0.5,MeanSquares')
    parser.add_option(name="-seg-t2star",
                      description='Segmentation of the spinal cord on t2star using sct_propseg\n                                          '
                                  'If used, a 3 point mask can be used to help sct_propseg : the mask file should be in the t2star folder with a name containing "mask"\n                                          '
                                  'If not used, a segmentation file should be provided in the anatomic data folder with a name containing "seg"',
                      mandatory=False)
    parser.add_option(name="-seg-t2star-params",
                      type_value=[[','], 'str'],
                      description='Parameters for sct_propseg on T2star (only if the -seg-t2star flag is used)\n                                          '
                                  'Parameters available : \n                                          '
                                  '                     - init : int, axial slice where the propagation starts, default is middle axial slice\n                                          '
                                  '                     - down : int, down limit of the propagation, default is 0\n                                          '
                                  '                     - up : int, up limit of the propagation, default is the higher slice of the image\n                                          '
                                  '                     - centerline : filename of centerline to use for the propagation, format .txt or .nii, see file structure in documentation\n                                          '
                                  '                     - init-mask : string, mask containing three center of the spinal cord, used to initiate the propagation\n                                          '
                                  '                     - radius : double, approximate radius of the spinal cord, default is 4 mm\n                                          ',
                      mandatory=False,
                      example='init:0.5,up:8,centerline:my_centerline.nii.gz,radius:6')
    parser.add_option(name="-reg-multimodal",
                      description='Registration of template to to T2star and warping template on T2star',
                      mandatory=False)
    parser.add_option(name="-reg-multimodal-params",
                      type_value='str',
                      description='Parameters for the registration of template to to T2star (only if the -reg-multimodal flag is used),\n                                          '
                                  'ALL ITEMS MUST BE LISTED IN ORDER. Separate with comma WITHOUT WHITESPACE IN BETWEEN.'
                                  'Default=5,SyN,0.5,MI'
                                  '1) number of iterations for last stage.'
                                  '2) algo: {SyN, BSplineSyN}'
                                  '3) gradient step. The larger the more deformation.'
                                  '4) metric: {MI,MeanSquares}.'
                                  'If you find very large deformations, switching to MeanSquares can help.',
                      mandatory=False,
                      example='10,SyN,0.5,MeanSquares')
    parser.add_option(name="-dice",
                      description='Compute the Dice coefficient on the results of the operations you did',
                      mandatory=False)
    parser.add_option(name="-dice-on",
                      type_value=[[','], 'str'],
                      description='Compute the Dice coefficient on the specified results : usefull if you did\'nt just do any operations but want to compute dice on previous results\n'
                                  'Has to be type_results:type_image, separated with coma, without any white spaces',
                      mandatory=False,
                      example="seg:t2,reg:t2star")

    arguments = parser.parse(sys.argv[1:])
    input_path_data = arguments["-data"]
    input_t = arguments["-t"]
    input_seg = False
    input_seg_params = {'init':0.5,'up':None,'down':None,'centerline':None,'init-mask':None,'radius':4}
    input_reg_template = False
    input_reg_template_params = '5,SyN,0.5,MI'
    input_seg_t2star = False
    input_seg_t2star_params = {'init':0.5,'up':None,'down':None,'centerline':None,'init-mask':None,'radius':4}
    input_reg_multimodal = False
    input_reg_multimodal_params = '5,SyN,0.5,MI'
    input_dice = False
    input_dice_on = []
    if "-seg" in arguments:
        input_seg = arguments["-seg"]
    if "-seg-params" in arguments:
        for param in arguments["-seg-params"]:
            option, value = param.split(':')
            input_seg_params[option] = value
    if "-reg-template" in arguments:
        input_reg_template = arguments["-reg-template"]
    if "-reg-template-params" in arguments:
        input_reg_template_params = arguments["-reg-template-params"]
    if "-seg-t2star" in arguments:
        input_seg_t2star = arguments["-seg-t2star"]
    if "-seg-t2star-params" in arguments:
        for param in arguments["-seg-t2star-params"]:
            option, value = param.split(':')
            input_seg_t2star_params[option] = value
    if "-reg-multimodal" in arguments:
        input_reg_multimodal = arguments["-reg-multimodal"]
    if "-reg-multimodal-params" in arguments:
        input_reg_multimodal_params = arguments["-reg-multimodal-params"]
    if "-dice" in arguments:
        input_dice = arguments["-dice"]
    if "-dice-on" in arguments:
        input_dice_on = arguments["-dice-on"]


    pipeline_test = Pipeline(input_path_data, input_t, seg=input_seg, seg_params=input_seg_params,  reg_template=input_reg_template, reg_template_params=input_reg_template_params, seg_t2star=input_seg_t2star, seg_t2star_params=input_seg_t2star_params, reg_multimodal=input_reg_multimodal, reg_multimodal_params=input_reg_multimodal_params, dice=input_dice, dice_on=input_dice_on)
    pipeline_test.compute()

    elapsed_time = round(time.time() - begin, 2)

    sct.printv('Finished ! \nElapsed time : ' + str(elapsed_time) + ' sec', 1, 'info' )