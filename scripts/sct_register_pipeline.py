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
# .........................subject_02_manual_segmentation_t1.nii.gz
# .................t2/
# .........................subject_02_anything_t2.nii.gz
# .........................some_landmarks_of_vertebral_levels.nii.gz
# .........................subject_02_manual_segmentation_t2.nii.gz
# .................t2star/
# .........................subject_02_anything_t2star.nii.gz
# .........................subject_02_manual_segmentation_t2star.nii.gz
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
# TODO : compute the 'total' mean dice (T1, T2, T2star)
#########################################################################################
from msct_parser import Parser
import sys
import sct_utils as sct
import os
import time
import copy_reg
import types


def _pickle_method(method):
    """
    Author: Steven Bethard (author of argparse)
    http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    """
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    cls_name = ''
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
    if cls_name:
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    """
    Author: Steven Bethard
    http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    """
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


# DEFAULT PARAMETERS
class Param(object):
    """
    Parameters used for the pipeline
    """

    # The constructor
    def __init__(self):
        self.debug = 0  # type: int
        self.verbose = 1  # type: int
        self.path_data = ''  # type: folder
        self.t = 't2'  # type: string


class Subject(object):
    """
    A directory containing data for a subject
    The directory must be organized as follows :
    ......subject_xx/
    ....t1/
    ........subject_xx_anything_t1.nii.gz
    ........some_landmarks_of_vertebral_levels.nii.gz
    ........subject_xx_manual_segmentation_t1.nii.gz
    ....t2/
    ........subject_xx_anything_t2.nii.gz
    ........some_landmarks_of_vertebral_levels.nii.gz
    ........subject_xx_manual_segmentation_t2.nii.gz
    ....t2star/
    ........subject_xx_anything_t2star.nii.gz
    ........subject_xx_manual_segmentation_t2star.nii.gz
    """
    def __init__(self, dir_name, dir_t1, name_t1, name_t1_seg, name_t1_ref, name_landmarks_t1,
                 dir_t2, name_t2, name_t2_seg, name_t2_ref, name_landmarks_t2,
                 dir_t2star, name_t2star, name_t2star_seg, name_t2star_ref):
        self.dir_name = dir_name

        # data t1
        self.dir_t1 = dir_t1
        self.name_t1 = name_t1
        self.name_t1_seg = name_t1_seg
        self.name_t1_ref = name_t1_ref
        self.name_landmarks_t1 = name_landmarks_t1

        # data t2
        self.dir_t2 = dir_t2
        self.name_t2 = name_t2
        self.name_t2_ref = name_t2_ref
        self.name_t2_seg = name_t2_seg
        self.name_landmarks_t2 = name_landmarks_t2

        # data t2star
        self.dir_t2star = dir_t2star
        self.name_t2star = name_t2star
        self.name_t2star_seg = name_t2star_seg
        self.name_t2star_ref = name_t2star_ref


class Pipeline(object):
    """
    Segmentation and registration pipeline
    """

    # The constructor
    def __init__(self, path_data, t, seg=True, seg_params=None, reg_template=False, reg_template_params=None,
                 seg_t2star=False,  seg_t2star_params=None, reg_multimodal=False, reg_multimodal_params=None,
                 straightening=False, straightening_params=None, dice=False, dice_on=None, verbose=1):
        self.path_data = path_data  # type: folder
        self.old_dir = os.getcwd()
        os.chdir(self.path_data)
        self.t = t   # type: string

        # options
        self.t = t
        self.seg = seg  # type: boolean
        self.seg_params = seg_params  # type: dictionary
        self.reg_template = reg_template  # type: boolean
        self.reg_template_params = reg_template_params  # type: string
        self.seg_t2star = seg_t2star  # type: boolean
        self.seg_t2star_params = seg_t2star_params  # type: dictionary
        self.reg_multimodal = reg_multimodal  # type: boolean
        self.reg_multimodal_params = reg_multimodal_params  # type: string
        self.straightening = straightening
        self.straightening_params = straightening_params
        self.straightening_results = []
        self.dice = dice  # type: boolean
        self.dice_on = dice_on  # type: list
        if self.dice_on is None:
            self.dice_on = []

        self.cpu_count = None
        self.verbose = verbose

        # generating data
        self.data = self.generate_data_list()   # type: list

    def generate_data_list(self):
        """
        Construction of the data list from the data set
        :return data:
        """
        data = []

        # each directory in "path_data" is a subject
        for subject_dir in os.listdir('./'):
            if not os.path.isdir(subject_dir):
                sct.printv("WARNING : Found a file, data should be organized in folders ...", verbose=self.verbose, type="warning")
            else:
                os.chdir(subject_dir)

                dir_t1 = ''
                dir_t2 = ''
                dir_t2star = ''
                sct.printv("\n\n---------------------------------------------------------------------------", self.verbose)
                sct.printv("SUBJECT : " + subject_dir, self.verbose)
                sct.printv("---> containing : ", self.verbose)
                if self.verbose:
                    print os.listdir('./')
                for directory in os.listdir('./'):
                    if 't1' in directory.lower():
                        dir_t1 = directory
                    if 't2star' in directory.lower():
                        dir_t2star = directory
                    elif 't2' in directory.lower():
                        dir_t2 = directory

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

                # T1 data
                if dir_t1 is not '':
                    for file_name in os.listdir(dir_t1):
                        if "t1.nii" in file_name.lower() and check_nii_gz(file_name, self.verbose) and 'warp' not in file_name:
                            name_t1 = file_name
                        if "landmark" in file_name.lower() and check_nii_gz(file_name, self.verbose):
                            name_landmarks_t1 = file_name
                        if 'manual_seg' in file_name.lower() or 'manualseg' in file_name.lower()\
                                or 'ref' in file_name.lower() and not "csf" in file_name.lower():
                            if check_nii_gz(file_name, self.verbose):
                                name_t1_ref = file_name
                        elif not self.seg:
                            if "seg" in file_name.lower() and check_nii_gz(file_name, self.verbose):
                                name_t1_seg = file_name

                    if name_t1 == '':
                        sct.printv("WARNING: could not find t1 file in folder" + dir_t1, verbose=self.verbose, type='warning')
                    if name_landmarks_t1 == '':
                        sct.printv("WARNING: could not find landmarks file in folder" + dir_t1,
                                   verbose=self.verbose, type='warning')
                    if name_t1_ref == '':
                        sct.printv("WARNING: could not find t1 reference segmentation file in folder" + dir_t1,
                                   verbose=self.verbose, type='warning')
                    if not self.seg and name_t1_seg == '':
                        sct.printv("WARNING: could not find t1 segmentation file in folder " + dir_t1,
                                   verbose=self.verbose, type='warning')
                else:
                    sct.printv('WARNING: no t1 folder', self.verbose, 'warning')

                # T2 data
                if dir_t2 is not '':
                    for file_name in os.listdir(dir_t2):
                        if "t2.nii" in file_name.lower() and check_nii_gz(file_name, self.verbose) and 'warp' not in file_name:
                            name_t2 = file_name
                        if "landmark" in file_name.lower() and check_nii_gz(file_name, self.verbose):
                            name_landmarks_t2 = file_name
                        if 'manual_seg' in file_name.lower() or 'manualseg' in file_name.lower()\
                                or 'ref' in file_name.lower():
                            if check_nii_gz(file_name, self.verbose):
                                name_t2_ref = file_name
                        elif not self.seg:
                            if "seg" in file_name.lower() and 'mask' not in file_name and 'warp' not in file_name \
                                    and check_nii_gz(file_name, self.verbose):
                                name_t2_seg = file_name

                    if name_t2 == '':
                        sct.printv("WARNING: could not find t2 file in folder " + dir_t2, verbose=self.verbose, type='warning')
                    if name_landmarks_t2 == '':
                        sct.printv("WARNING: could not find landmarks file in folder " + dir_t2,
                                   verbose=self.verbose, type='warning')
                    if name_t2_ref == '':
                        sct.printv("WARNING: could not find t2 reference segmentation file in folder " + dir_t2,
                                   verbose=self.verbose, type='warning')
                    if not self.seg and name_t2_seg == '':
                        sct.printv("WARNING: could not find t2 segmentation file in folder " + dir_t2,
                                   verbose=self.verbose, type='warning')
                else:
                    sct.printv('WARNING: no t2 folder', self.verbose, 'warning')

                # T2star data
                if dir_t2star is not '':
                    for file_name in os.listdir(dir_t2star):
                        if 't2star.nii' in file_name and check_nii_gz(file_name, self.verbose) and 'warp' not in file_name:
                            name_t2star = file_name
                        if 'manual_seg' in file_name.lower() or 'manualseg' in file_name.lower()\
                                or 'ref' in file_name.lower():
                            if check_nii_gz(file_name, self.verbose):
                                name_t2star_ref = file_name
                        elif not self.seg_t2star:
                            if "seg" in file_name.lower() and 'mask' not in file_name and 'seg_in' not in file_name\
                                    and 'warp' not in file_name and check_nii_gz(file_name, self.verbose):
                                name_t2star_seg = file_name

                    if name_t2star == '':
                        sct.printv("WARNING: could not find t2star file in folder " + dir_t2star,
                                   verbose=self.verbose, type='warning')
                    if name_t2star_ref == '':
                        sct.printv("WARNING: could not find t2star reference segmentation file in folder " + dir_t2star,
                                   verbose=self.verbose, type='warning')
                    if not self.seg_t2star and name_t2star_seg == '':
                        sct.printv("WARNING: could not find t2star segmentation file in folder " + dir_t2star,
                                   verbose=self.verbose, type='warning')
                else:
                    sct.printv('WARNING: no t2star folder', self.verbose, 'warning')

                # creation of a subject for each directory
                data.append(Subject(subject_dir, dir_t1, name_t1, name_t1_seg, name_t1_ref, name_landmarks_t1,
                                    dir_t2, name_t2, name_t2_seg, name_t2_ref, name_landmarks_t2,
                                    dir_t2star, name_t2star, name_t2star_seg, name_t2star_ref))
                sct.printv('\nFiles for subject ' + subject_dir +
                           '\nT1: .................. ' + name_t1 +
                           '\nT1 seg: .............. ' + name_t1_seg +
                           '\nT1 ref: .............. ' + name_t1_ref +
                           '\nT1 landmarks: ........ ' + name_landmarks_t1 +
                           '\nT2: .................. ' + name_t2 +
                           '\nT2 seg: .............. ' + name_t2_seg +
                           '\nT2 ref: .............. ' + name_t2_ref +
                           '\nT2 landmarks: ........ ' + name_landmarks_t2 +
                           '\nT2star: .............. ' + name_t2star +
                           '\nT2star seg: .......... ' + name_t2star_seg +
                           '\nT2star ref: .......... ' + name_t2star_ref, self.verbose, "normal")
                os.chdir('..')
        if not data:
            sct.printv("ERROR : Data should be organized in folders ... ", verbose=self.verbose, type="error")

        return data

    def segmentation(self, t, init=0.5, up=None, down=None, centerline=None, init_mask=None, radius=None):
        """
        segmentation of the spinal cord using sct_propseg for all the subjects of the data set
        :param t: type of image to segment
        :param init:
        :param up:
        :param down:
        :param centerline:
        :param init_mask:
        :param radius:
        :return:
        """
        for subject in self.data:
            os.chdir(subject.dir_name)
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
                init_mask = ''
                for file_name in os.listdir(path):
                    if 'mask' in file_name:
                        self.seg_t2star_params['init-mask'] = file_name
                        init_mask = file_name

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

                sct.printv("\nDoing segmentation on " + subject.dir_name + '/' + path + "/" + name +
                           " using sct_propseg ...", verbose=self.verbose, type="normal")
                sct.run(cmd)
                for file_name in os.listdir('./'):
                            if "_seg.nii" in file_name:
                                name_seg = file_name
                                break
                if t == 't1':
                    subject.name_t1_seg = name_seg
                if t == 't2':
                    subject.name_t2_seg = name_seg
                if t == 't2star':
                    subject.name_t2star_seg = name_seg
                os.chdir('..')
            else:
                sct.printv("WARNING: no file to do segmentation on in folder " + path, verbose=self.verbose, type='warning')
            os.chdir('..')

    def worker_straightening(self, subject):
        os.chdir(subject.dir_name)

        if self.t == 't1':
            path = subject.dir_t1
            name = subject.name_t1
            name_seg = subject.name_t1_ref
        elif self.t == 't2':
            path = subject.dir_t2
            name = subject.name_t2
            name_seg = subject.name_t2_ref

        if path is '' or name is '' or name_seg is '':
            sct.printv(
                'WARNING: AN ERROR OCCURRED WHEN TRYING TO STRAIGHTEN THE SPINAL CORD. The images seem to be missing.', self.verbose)
            os.chdir('..')
            from numpy import nan
            return [subject.dir_name, nan, nan]
        else:
            try:
                os.chdir(path)

                cmd_straightening = 'sct_straighten_spinalcord -i ' + name + ' -c ' + name_seg
                sct.printv("\nStraightening " + subject.dir_name + '/' + path + '/'
                                   + name + " using sct_straighten_spinalcord ...", verbose=self.verbose, type="normal")

                from sct_straighten_spinalcord import SpinalCordStraightener

                sc_straight = SpinalCordStraightener(name, name_seg)

                if self.straightening_params is not None:
                    cmd_straightening += ' ' + self.straightening_params
                    dict_params_straightening = dict([param.split('=') for param in self.straightening_params.split(',')])
                    if "algo_fitting" in dict_params_straightening:
                        sc_straight.algo_fitting = str(dict_params_straightening["algo_fitting"])
                    if "bspline_meshsize" in dict_params_straightening:
                        sc_straight.bspline_meshsize = str(dict_params_straightening["bspline_meshsize"])
                    if "bspline_numberOfLevels" in dict_params_straightening:
                        sc_straight.bspline_numberOfLevels = str(dict_params_straightening["bspline_numberOfLevels"])
                    if "bspline_order" in dict_params_straightening:
                        sc_straight.bspline_order = str(dict_params_straightening["bspline_order"])
                    if "algo_landmark_rigid" in dict_params_straightening:
                        sc_straight.algo_landmark_rigid = str(dict_params_straightening["algo_landmark_rigid"])
                    if "all_labels" in dict_params_straightening:
                        sc_straight.all_labels = int(dict_params_straightening["all_labels"])
                    if "use_continuous_labels" in dict_params_straightening:
                        sc_straight.use_continuous_labels = int(dict_params_straightening["use_continuous_labels"])
                    if "cpu-nb" in dict_params_straightening:
                        sc_straight.cpu_number = int(dict_params_straightening["cpu-nb"])

                sct.printv(cmd_straightening, self.verbose)
                sc_straight.remove_temp_files = 0
                sc_straight.verbose = self.verbose  # for visualization purpose
                sc_straight.straighten()

                os.chdir('../..')

                return [subject.dir_name, sc_straight.mse_straightening, sc_straight.max_distance_straightening]

            except Exception, e:
                sct.printv('WARNING: AN ERROR OCCURRED WHEN TRYING TO STRAIGHTEN THE SPINAL CORD ON ' + self.t.upper() + ': ',
                           self.verbose, 'warning')
                print e
                sct.printv('Continuing program ...', self.verbose, 'warning')

                os.chdir('../..')
                from numpy import nan
                return [subject.dir_name, nan, nan]

            except KeyboardInterrupt:
                os.chdir('../..')
                return

    def worker_straightening_results(self, results):
        self.straightening_results = [result for result in results if result is not None]
        sorted(self.straightening_results, key=lambda l: l[0])

    def straighten_spinalcord(self):
        """
        straighten image based on segmentation and/or manual labels
        :param t: type of anatomical image to register
        :return:
        """
        from multiprocessing import Pool

        pool = Pool(processes=self.cpu_count)
        pool.map_async(self.worker_straightening, self.data, callback=self.worker_straightening_results)

        pool.close()
        try:
            pool.join()  # waiting for all the jobs to be done
        except KeyboardInterrupt:
            print "\nWarning: Caught KeyboardInterrupt, terminating workers"
            pool.terminate()
            sys.exit(2)
        except Exception as e:
            print e
            sys.exit(2)

    def register_warp_to_template(self, t):
        """
        register anatomical image to template for all the subjects of the data set
        :param t: type of anatomical image to register
        :return:
        """
        for subject in self.data:
            os.chdir(subject.dir_name)
            name_seg = None
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
                cmd_register = 'sct_register_to_template -i ' + name + ' -s ' + name_seg + ' -l ' \
                               + subject.name_landmarks_t2

                if self.reg_template_params is not None:
                    cmd_register = cmd_register + " -p " + str(self.reg_template_params)

                sct.printv("\nDoing registration to template on " + subject.dir_name + '/' + path + '/'
                           + subject.name_t2 + " using sct_register_to_template ...", verbose=self.verbose, type="normal")
                sct.run(cmd_register)
            except Exception, e:
                    sct.printv('WARNING: AN ERROR OCCURRED WHEN TRYING TO REGISTER TEMPLATE TO' + t.upper() + ' : ',
                               self.verbose, 'warning')
                    print e
                    sct.printv('Continuing program ...', self.verbose, 'warning')
            else:
                try:
                    # TODO change name of the warping field by a Subject attribute ??
                    cmd_warp = 'sct_warp_template -d ' + name + ' -w warp_template2anat.nii.gz'

                    sct.printv("\nWarping Template to T2 on " + subject.dir_name + '/' + path + "/"
                               + name + " using sct_warp_template ...", verbose=self.verbose, type="normal")
                    sct.run(cmd_warp)
                except Exception, e:
                    sct.printv('WARNING: AN ERROR OCCURRED WHEN TRYING TO WARP TEMPLATE TO ' + t.upper() + ' : ',
                               self.verbose, 'warning')
                    print e
                    sct.printv('Continuing program ...', self.verbose, 'warning')
            os.chdir('../..')

    def register_warp_multimodal(self, t, src='template2anat.nii.gz'):
        """
        register metric image (T2 star) to template for all the subjects of the data set
        :param t: type of anatomical image to register
        :param src: source file to use for registration
        :return:
        """
        for subject in self.data:
            os.chdir(subject.dir_name)
            if subject.dir_t2star is not '':
                if t == 't1':
                    path_anat = '../' + subject.dir_t1
                elif t == 't2':
                    path_anat = '../' + subject.dir_t2

                try:
                    os.chdir(subject.dir_t2star)

                    # Register
                    cmd_register = 'sct_register_multimodal -i ' + path_anat + '/' + src + ' -d ' + subject.name_t2star \
                                   + ' -iseg ' + path_anat + '/label/template/MNI-Poly-AMU_cord.nii.gz' \
                                   ' -dseg ' + subject.name_t2star_seg
                    if self.reg_multimodal_params is not None:
                        cmd_register = cmd_register + " -p " + str(self.reg_multimodal_params)

                    sct.printv("\nDoing multimodal registration of " + src + " to " + subject.dir_name + '/'
                               + subject.dir_t2star + "/" + subject.name_t2star + " using sct_register_multimodal ...",
                               verbose=self.verbose, type="normal")
                    sct.run(cmd_register)
                except Exception, e:
                    sct.printv('WARNING: AN ERROR OCCURRED WHEN TRYING TO REGISTER TEMPLATE TO T2STAR : ', 1, 'warning')
                    print e
                    sct.printv('Continuing program ...', 1, 'warning')
                else:

                    try:
                        # concatenate transformations
                        src_name = sct.extract_fname(src)[1]
                        multimodal_warp_name = 'warp_' + src_name + '2' + subject.name_t2star
                        total_warp_name = 'warp_template2t2star.nii.gz'

                        cmd_concat = 'sct_concat_transfo -w ' + path_anat + '/warp_template2anat.nii.gz,'\
                                     + multimodal_warp_name + ' -d ' + subject.name_t2star + ' -o ' + total_warp_name
                        sct.printv("\nConcatenate transformations using sct_concat_transfo ...",
                                   verbose=self.verbose, type="normal")
                        sct.run(cmd_concat)
                    except Exception, e:
                        sct.printv('WARNING: AN ERROR OCCURRED WHEN TRYING TO CONCATENATE TRANSFORMATIONS : ',
                                   self.verbose, 'warning')
                        print e
                        sct.printv('Continuing program ...', self.verbose, 'warning')

                    else:
                        try:
                            # warp template to t2star
                            cmd_warp = 'sct_warp_template -d ' + subject.name_t2star + ' -w ' + total_warp_name

                            sct.printv("\nWarping Template to T2star on " + subject.dir_name + "/" + subject.dir_t2star
                                       + "/" + subject.name_t2star + " using sct_warp_template ...",
                                       verbose=self.verbose, type="normal")
                            sct.run(cmd_warp)
                        except Exception, e:
                            sct.printv('WARNING: AN ERROR OCCURRED WHEN TRYING TO WARP TEMPLATE TO T2STAR : ',
                                       self.verbose, 'warning')
                            print e
                            sct.printv('Continuing program ...', self.verbose, 'warning')
                os.chdir('..')

            else:
                sct.printv('WARNING: no t2star folder, did not apply sct_register_multimodal to subject '
                           + subject.dir_name, self.verbose, 'warning')
            os.chdir('..')

    def compute_dice(self, t, type_res):
        """
        compute the dice coefficient of all subjects in the data folder for the type of image in input(t1, t2 or t2star)
        :param t: type of data to compute dice on
        :param type_res: type of results to validate by computing dice (segmentation or registration)
        :return:
        """
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
            sct.printv('WARNING: input type of results to compute dice coefficient on is unrecognized ...',
                       self.verbose, 'warning')
        res_file.write('Dice coefficient for ' + t.upper() + name_res + ' data in file ' + self.path_data + '\n')
        for subject in self.data:
            os.chdir(subject.dir_name)
            path = ''
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

            if type_res == 'reg':
                res = './label/template/MNI-Poly-AMU_cord.nii.gz'

            if ref is not '':
                try:
                    os.chdir(path)
                    cmd_dice = 'sct_dice_coefficient ' + res + ' ' + ref
                    sct.printv("\nComputing the dice coefficient for the " + t + " image(s) of subject "
                               + subject.dir_name + "  ...", verbose=self.verbose, type="normal")
                    status, output = sct.run(cmd_dice)

                    dice_dictionary[subject.dir_name] = float(output.split(' ')[-1][:-2])
                    res_file.write(subject.dir_name + ' : ' + str(dice_dictionary[subject.dir_name]) + '\n')
                    n_subjects += 1
                    sum_dice += dice_dictionary[subject.dir_name]

                except Exception, e:
                    sct.printv('WARNING: AN ERROR OCCURRED WHEN TRYING TO COMPUTE DICE BETWEEN ' + subject.dir_name
                               + '/' + t + '/' + ref + ' AND ' + subject.dir_name + '/' + t + '/' + res + ' :',
                               self.verbose, 'warning')
                    print e
                    sct.printv('Continuing program ...\n', self.verbose, 'warning')
                finally:
                    os.chdir('..')

            else:
                sct.printv('WARNING: no reference segmentation for the ' + t + ' image of subject ' + subject.dir_name
                           + ', did not compute dice.', verbose=self.verbose, type='warning')
            os.chdir('..')
        mean_dice = sum_dice / n_subjects
        res_file.write('Mean Dice coefficient : ' + str(mean_dice) + '\n')
        res_file.close()
        return dice_dictionary, mean_dice

    def write_infos(self):
        time_now = time.localtime()
        now = str(time_now.tm_year) + '-' + str(time_now.tm_mon) + '-' + str(time_now.tm_mday) + '_' \
            + str(time_now.tm_hour) + 'H' + str(time_now.tm_min)
        name_infos = now + '_parameters_info.txt'
        infos = open(name_infos, 'w')
        infos.write('Data treated using the sct_register_pipeline \n'
                    'Path to data : ' + self.path_data + '\n'
                    'Type of anatomic data : ' + self.t + '\n')
        infos.write('\n\nTreatments applied : \n'
                    ' - Segmentation of anatomic image : ' + yes_no(self.seg) + '\n')
        if self.seg:
            infos.write('   --> Parameters : ' + str(self.seg_params) + '\n')
        infos.write(' - Registration of anat to template and warping of template to anat : '
                    + yes_no(self.reg_template) + '\n')
        if self.reg_template:
            infos.write('   --> Parameters : ' + str(self.reg_template_params) + '\n')
        infos.write(' - Segmentation of T2star image : ' + yes_no(self.seg_t2star) + '\n')
        if self.seg_t2star:
            infos.write('   --> Parameters : ' + str(self.seg_t2star_params) + '\n')
        infos.write(' - Registration of T2star to Template2anat and warping of Template2anat to T2star : '
                    + yes_no(self.reg_multimodal) + '\n')
        if self.reg_multimodal:
            infos.write('   --> Parameters : ' + str(self.reg_multimodal_params) + '\n')
        if self.dice:
            infos.write(' - Computed Dice coefficient\n')
        infos.close()

    def compute(self):
        """
        compute each step of the pipeline according the input parameters
        :return:
        """
        # segmentation of the anatomical image
        if self.seg:
            if self.t == 'both' or self.t == 'b':
                self.segmentation('t1', init=self.seg_params['init'], up=self.seg_params['up'],
                                  down=self.seg_params['down'], centerline=self.seg_params['centerline'],
                                  init_mask=self.seg_params['init-mask'], radius=self.seg_params['radius'])
                self.segmentation('t2', init=self.seg_params['init'], up=self.seg_params['up'],
                                  down=self.seg_params['down'], centerline=self.seg_params['centerline'],
                                  init_mask=self.seg_params['init-mask'], radius=self.seg_params['radius'])
                if self.dice:
                    self.compute_dice('t1', 'seg')
                    self.compute_dice('t2', 'seg')

            else:
                self.segmentation(self.t, init=self.seg_params['init'], up=self.seg_params['up'],
                                  down=self.seg_params['down'], centerline=self.seg_params['centerline'],
                                  init_mask=self.seg_params['init-mask'], radius=self.seg_params['radius'])
                if self.dice:
                    self.compute_dice(self.t, 'seg')

        # registration of anatomical image to template and warping of template to anat
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

        # segmentation of the T2star image
        if self.seg_t2star:
            self.segmentation('t2star', init=self.seg_t2star_params['init'], up=self.seg_t2star_params['up'],
                              down=self.seg_t2star_params['down'], centerline=self.seg_t2star_params['centerline'],
                              init_mask=self.seg_t2star_params['init-mask'], radius=self.seg_t2star_params['radius'])
            if self.dice:
                self.compute_dice('t2star', 'seg')

        # registration of T2star on Template2anat image and warping of template2anat to T2star
        if self.reg_multimodal:
            if self.t == 'both' or self.t == 'b':
                self.register_warp_multimodal('t1')
                self.register_warp_multimodal('t2')
            else:
                self.register_warp_multimodal(self.t)
            if self.dice:
                self.compute_dice('t2star', 'reg')

        if self.straightening:
            self.straighten_spinalcord()
            from numpy import array, mean, std
            if self.straightening_results:
                mse_results = array([itm[1] for itm in self.straightening_results])
                max_distance = array([itm[2] for itm in self.straightening_results])
                sct.printv('\n', self.verbose)
                if self.verbose:
                    print self.straightening_results
                self.straightening_results_mse = [mean(mse_results), std(mse_results)]
                self.straightening_results_dist_max = [mean(max_distance), std(max_distance)]
                sct.printv('MSE          = ' + str(self.straightening_results_mse[0]) + ' +- ' + str(
                    self.straightening_results_mse[1]), self.verbose)
                sct.printv('Max distance = ' + str(self.straightening_results_dist_max[0]) + ' +- ' + str(
                    self.straightening_results_dist_max[1]), self.verbose)
            else:
                print self.straightening_results
                sct.printv('WARNING: there are no results for straightening... Please check your data or contact the SCToolbox administrators.', self.verbose, 'warning')

        # compute dice on other results
        for arg in self.dice_on:
            type_res, type_image = arg.split(":")
            self.compute_dice(type_image, type_res)

        self.write_infos()

        os.chdir(self.old_dir)


# =======================================================================================================================
# Static functions
# =======================================================================================================================#
def yes_no(bool_param):
    if bool_param:
        return 'YES'
    else:
        return 'NO'


def check_nii_gz(full_file, verbose=1):
    """
    Check if file extension is .nii.gz, if yes, return True, else, print a warning message and return False
    :param full_file:
    :return Boolean:
    """
    file_path, file_name, file_ext = sct.extract_fname(full_file)
    if file_ext != '.nii.gz':
        sct.printv('WARNING: File ' + file_name + ' should be .nii.gz instead of ' + file_ext + '...', verbose, 'warning')
        return False
    else:
        return True

#  =======================================================================================================================
# Start program
# =======================================================================================================================
if __name__ == "__main__":
    begin = time.time()
    # initialize parameters
    param = Param()
    param_default = Param()

    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Pipeline to do Template --> Anatomical image registration on a lot of data\n '
                                 'Your data should be organized with one folder per subject\n'
                                 'each subject folder should contain a folder per data type (t1, t2, etc)\n'
                                 't1 and t2 files should be named "anything_t1.nii.gz" and "anything_t2.nii.gz"\n')
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
                      description='Segmentation of the spinal cord on anatomic data using sct_propseg\n'
                                  'If not used, a segmentation file should be provided in the anatomic data folder'
                                  ' with a name containing "seg"',
                      mandatory=False)
    parser.add_option(name="-seg-params",
                      type_value=[[','], 'str'],
                      description='Parameters for sct_propseg (only if the -seg flag is used)\n'
                                  'Parameters available : \n'
                                  '- init : int, axial slice where the propagation starts,'
                                  ' default is middle axial slice\n'
                                  '- down : int, down limit of the propagation, default is 0\n'
                                  '- up : int, up limit of the propagation, default is the higher slice of the image\n'
                                  '- centerline : filename of centerline to use for the propagation, '
                                  'format .txt or .nii, see file structure in documentation\n'
                                  '- init-mask : string, mask containing three center of the spinal cord,'
                                  ' used to initiate the propagation\n'
                                  '- radius : double, approximate radius of the spinal cord, default is 4 mm\n',
                      mandatory=False,
                      example='init:0.5,up:8,centerline:my_centerline.nii.gz,radius:6')
    parser.add_option(name="-reg-template",
                      description='Registration of anatomic data to template and warping template on anatomic data,\n'
                                  'WARNING : OPTION NOT READY TO USE YET FOR T1 DATA \n'
                                  'A landmarks file containing appropriate labels at spinal cord center should be'
                                  ' in the anatomic data folder \n'
                                  'See: http://sourceforge.net/p/spinalcordtoolbox/wiki/create_labels/',
                      mandatory=False)
    parser.add_option(name="-reg-template-params",
                      type_value='str',
                      description='Parameters for the registration of anatomic data to template'
                                  ' (only if the -reg-template flag is used).'
                                  'step: number og the step this parameters are for'
                                  'type: {im, seg} type of data used for this step'
                                  'algo: {syn, bsplinesyn, slicereg}'
                                  'metric: {MI,MeanSquares}'
                                  'iter: number of iterations'
                                  'shrink: shrinkage factor, >1 may fasten the computation'
                                  'smooth: smoothing factor'
                                  'gradStep: gradient step, the larger the more deformation'
                                  'If you find very large deformations, switching to MeanSquares can help.',
                      mandatory=False,
                      example='step=1,type=seg,algo=syn,metric=MeanSquares,iter=5:step=2,type=im,algo=slicereg,'
                              'metric=MeanSquares,iter=5')
    parser.add_option(name="-seg-t2star",
                      description='Segmentation of the spinal cord on t2star using sct_propseg\n'
                                  'If used, a 3 point mask can be used to help sct_propseg : the mask file should be '
                                  'in the t2star folder with a name containing "mask"\n'
                                  'If not used, a segmentation file should be provided in the anatomic data folder'
                                  ' with a name containing "seg"',
                      mandatory=False)
    parser.add_option(name="-seg-t2star-params",
                      type_value=[[','], 'str'],
                      description='Parameters for sct_propseg on T2star (only if the -seg-t2star flag is used)\n'
                                  'Parameters available : \n'
                                  '- init : int, axial slice where the propagation starts,'
                                  ' default is middle axial slice\n'
                                  '- down : int, down limit of the propagation, default is 0\n'
                                  '- up : int, up limit of the propagation, default is the higher slice of the image\n'
                                  '- centerline : filename of centerline to use for the propagation,'
                                  ' format .txt or .nii, see file structure in documentation\n'
                                  '- init-mask : string, mask containing three center of the spinal cord, '
                                  'used to initiate the propagation\n'
                                  '- radius : double, approximate radius of the spinal cord, default is 4 mm\n',
                      mandatory=False,
                      example='init:0.5,up:8,centerline:my_centerline.nii.gz,radius:6')
    parser.add_option(name="-reg-multimodal",
                      description='Registration of template to to T2star and warping template on T2star',
                      mandatory=False)
    parser.add_option(name="-reg-multimodal-params",
                      type_value='str',
                      description='Parameters for the registration of template to to T2star'
                                  ' (only if the -reg-multimodal flag is used),\n'
                                  'step: number og the step this parameters are for'
                                  'type: {im, seg} type of data used for this step'
                                  'algo: {syn, bsplinesyn, slicereg}'
                                  'metric: {MI,MeanSquares}'
                                  'iter: number of iterations'
                                  'shrink: shrinkage factor, >1 may fasten the computation (only for SyN)'
                                  'smooth: smoothing factor (only for SyN)'
                                  'gradStep: gradient step, the larger the more deformation (only for SyN)'
                                  'poly: polynomial degree (only for slicereg)'
                                  'If you find very large deformations, switching to MeanSquares can help.',
                      mandatory=False,
                      example='step=1,type=seg,algo=syn,metric=MeanSquares,iter=5:step=2,type=im,algo=slicereg,'
                              'metric=MeanSquares,iter=5')
    parser.add_option(name="-straightening",
                      description='Straighten the spinal cord',
                      mandatory=False)
    parser.add_option(name="-straightening-params",
                      type_value='str',
                      description='Parameters for the spinal cord straightening\nalgo_fitting: {hanning,nurbs} algorithm for curve fitting. Default=hanning\nbspline_meshsize: <int>x<int>x<int> size of mesh for B-Spline registration. Default=5x5x10\nbspline_numberOfLevels: <int> number of levels for BSpline interpolation. Default=3\nbspline_order: <int> Order of BSpline for interpolation. Default=2\nalgo_landmark_rigid {rigid,xy,translation,translation-xy,rotation,rotation-xy} constraints on landmark-based rigid pre-registration',
                      mandatory=False,
                      example='algo_fitting=nurbs,bspline_meshsize=5x5x12,algo_landmark_rigid=xy')
    parser.add_option(name="-dice",
                      description='Compute the Dice coefficient on the results of the operations you did',
                      mandatory=False)
    parser.add_option(name="-dice-on",
                      type_value=[[','], 'str'],
                      description='Compute the Dice coefficient on the specified results : usefull if you didn\'t '
                                  'just do any operations but want to compute dice on previous results\n'
                                  'Has to be type_results:type_image, separated with coma, without any white spaces',
                      mandatory=False,
                      example="seg:t2,reg:t2star")
    from multiprocessing import cpu_count
    parser.add_option(name="-cpu-count",
                      type_value="int",
                      description='number of cores used by the pipeline. Default='+str(cpu_count()),
                      mandatory=False,
                      example="6")

    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="Verbose. 0: nothing, 1: basic, 2: extended.",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value=1)

    arguments = parser.parse(sys.argv[1:])
    input_path_data = arguments["-data"]
    input_t = arguments["-t"]
    input_seg = False
    input_seg_params = {'init': 0.5, 'up': None, 'down': None, 'centerline': None, 'init-mask': None, 'radius': 4}
    input_reg_template = False
    input_reg_template_params = 'step=1,type=seg,algo=syn,metric=MeanSquares,iter=5:step=2,type=im,algo=slicereg,' \
                                'metric=MeanSquares,iter=5'
    input_seg_t2star = False
    input_seg_t2star_params = {'init': 0.5, 'up': None, 'down': None, 'centerline': None, 'init-mask': None,
                               'radius': 4}
    input_reg_multimodal = False
    input_reg_multimodal_params = 'step=1,type=seg,algo=syn,metric=MeanSquares,iter=5:step=2,type=im,algo=slicereg,' \
                                  'metric=MeanSquares,iter=5'
    input_straightening = False
    input_straightening_params = 'algo=hanning'

    input_verbose = 0

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
    if "-straightening" in arguments:
        input_straightening = arguments["-straightening"]
    if "-straightening-params" in arguments:
        input_straightening_params = arguments["-straightening-params"]
    if "-dice" in arguments:
        input_dice = arguments["-dice"]
    if "-dice-on" in arguments:
        input_dice_on = arguments["-dice-on"]

    pipeline_test = Pipeline(input_path_data, input_t, seg=input_seg, seg_params=input_seg_params,
                             reg_template=input_reg_template, reg_template_params=input_reg_template_params,
                             seg_t2star=input_seg_t2star, seg_t2star_params=input_seg_t2star_params,
                             reg_multimodal=input_reg_multimodal, reg_multimodal_params=input_reg_multimodal_params,
                             straightening=input_straightening,straightening_params=input_straightening_params,
                             dice=input_dice, dice_on=input_dice_on)
    if "-cpu-count" in arguments:
        pipeline_test.cpu_count = arguments["-cpu-count"]
    if "-v" in arguments:
        pipeline_test.verbose = int(arguments["-v"])
        input_verbose = int(arguments["-v"])
    pipeline_test.compute()

    elapsed_time = round(time.time() - begin, 2)

    sct.printv('Finished ! \nElapsed time : ' + str(elapsed_time) + ' sec', input_verbose, 'info')