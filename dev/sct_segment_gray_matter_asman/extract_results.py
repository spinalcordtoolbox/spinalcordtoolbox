#!/usr/bin/env python
import xlsxwriter as xl
import os
from msct_gmseg_utils import *

path = '.'

'''
# ############################# MULTIPLE LOOCV FROM MAGMA #############################
# Create a workbook and add a worksheet.
workbook = xl.Workbook('results_wmseg_dices.xlsx')
worksheet_36 = workbook.add_worksheet('36subjects')
worksheet_28 = workbook.add_worksheet('28subjects')
worksheet_20 = workbook.add_worksheet('20subjects')
worksheet_15 = workbook.add_worksheet('15subjects')
worksheet_10 = workbook.add_worksheet('10subjects')
worksheet_5 = workbook.add_worksheet('5subjects')
worksheet_2 = workbook.add_worksheet('2subjects')

bold = workbook.add_format({'bold': True})


worksheets = {'36subjects': [worksheet_36, 2, {}], '28subjects': [worksheet_28, 2, {}], '20subjects': [worksheet_20, 2, {}], '15subjects': [worksheet_15, 2, {}], '10subjects': [worksheet_10, 2, {}], '5subjects': [worksheet_5, 2, {}], '2subjects': [worksheet_2, 2, {}]}

for w in worksheets.values():
    w[0].write(1, 0, 'Subject', bold)
    w[0].write(1, 1, 'Slice #', bold)
    w[0].write(1, 2, 'Slice level', bold)

    for mod in range(8):
        w[0].write(1, 3+3*mod, 'Dice - gamma 1.2', bold)
        w[0].write(1, 3+3*mod+1, 'Dice - gamma 0', bold)
        w[0].write(1, 3+3*mod+2, 'Number of model slices', bold)
        w[0].merge_range(0, 3+3*mod, 0, 3+3*mod+2, str(mod), bold)


for loocv_dir in os.listdir(path):
    if os.path.isdir(path + '/' + loocv_dir) and 'dictionary' not in loocv_dir:
        words = loocv_dir.split('_')
        n_sub = words[0]
        mod = int(words[1])
        gamma = words[-1]
        if gamma == '1.2':
            to_add_col = 0
        else:
            to_add_col = 1

        dice_file = open(path + '/' + loocv_dir + '/wm_dice_coeff.txt', 'r')
        dice_lines = dice_file.readlines()
        dice_file.close()

        dice_lines = [line.split(' ') for line in dice_lines]

        col = 0
        for line in dice_lines:
            line[2] = line[2][:-1]
            line[-1] = line[-1][:-1]
            line.pop(4)
            line.pop(4)
            slice_id = '_'.join(line[:2])
            if slice_id not in worksheets[n_sub][2].keys():
                worksheets[n_sub][2][slice_id] = worksheets[n_sub][1]
                for i, v in enumerate(line[:3]):
                    worksheets[n_sub][0].write(worksheets[n_sub][1], col+i, v)
                worksheets[n_sub][0].write(worksheets[n_sub][1], 3+3*mod+to_add_col, float(line[3]))
                worksheets[n_sub][0].write(worksheets[n_sub][1], 3+3*mod+2, float(line[-1]))
                worksheets[n_sub][1] += 1
            else:
                worksheets[n_sub][0].write(worksheets[n_sub][2][slice_id], 3+3*mod+to_add_col, float(line[3]))
                worksheets[n_sub][0].write(worksheets[n_sub][2][slice_id], 3+3*mod+2, float(line[-1]))

workbook.close()

# ############################# MULTIPLE LOOCV FROM MAGMA :  get n_sice data #############################
workbook2 = xl.Workbook('results_wmseg_dices_by_n_slices.xlsx')
worksheet_36 = workbook2.add_worksheet('36subjects')
worksheet_28 = workbook2.add_worksheet('28subjects')
worksheet_20 = workbook2.add_worksheet('20subjects')
worksheet_15 = workbook2.add_worksheet('15subjects')
worksheet_10 = workbook2.add_worksheet('10subjects')
worksheet_5 = workbook2.add_worksheet('5subjects')
worksheet_2 = workbook2.add_worksheet('2subjects')

bold = workbook2.add_format({'bold': True})

init_row = 2
init_col = 1
worksheets_n_slices = {'36subjects': [worksheet_36, init_col, {}], '28subjects': [worksheet_28, init_col, {}], '20subjects': [worksheet_20, init_col, {}], '15subjects': [worksheet_15, init_col, {}], '10subjects': [worksheet_10, init_col, {}], '5subjects': [worksheet_5, init_col, {}], '2subjects': [worksheet_2, init_col, {}]}

for w in worksheets_n_slices.values():
    w[0].write(0, 0, 'Number of slices in model', bold)
    w[0].write(1, 0, 'Gamma', bold)

for loocv_dir in os.listdir(path):
    if os.path.isdir(path + '/' + loocv_dir) and 'dictionary' not in loocv_dir:
        words = loocv_dir.split('_')
        n_sub = words[0]
        mod = int(words[1])
        gamma = words[-1]
        if gamma == '1.2':
            to_add_col = 0
        else:
            to_add_col = 1

        dice_file = open(path + '/' + loocv_dir + '/wm_dice_by_n_slice.txt', 'r')
        dice_lines = dice_file.readlines()
        dice_file.close()

        dice_lines = [line.split(':') for line in dice_lines]

        for line in dice_lines:
            if line[0] != '\n':
                n_slices = line[0]
                n_slices = n_slices.replace(' ', '')

                if n_slices not in worksheets_n_slices[n_sub][2].keys():
                    worksheets_n_slices[n_sub][2][n_slices] = [init_row, init_row, worksheets_n_slices[n_sub][1]]
                    worksheets_n_slices[n_sub][0].write(0, worksheets_n_slices[n_sub][1], int(n_slices))
                    worksheets_n_slices[n_sub][0].write(0, worksheets_n_slices[n_sub][1]+1, int(n_slices))

                    worksheets_n_slices[n_sub][0].write(1, worksheets_n_slices[n_sub][1], 1.2)
                    worksheets_n_slices[n_sub][0].write(1, worksheets_n_slices[n_sub][1]+1, 0)

                    worksheets_n_slices[n_sub][1] += 2
                    for i, dc in enumerate(line[1].split(',')):
                        worksheets_n_slices[n_sub][0].write(worksheets_n_slices[n_sub][2][n_slices][to_add_col], worksheets_n_slices[n_sub][2][n_slices][2]+to_add_col, float(dc))
                        worksheets_n_slices[n_sub][2][n_slices][to_add_col] += 1
                else:
                    for i, dc in enumerate(line[1].split(',')):
                        worksheets_n_slices[n_sub][0].write(worksheets_n_slices[n_sub][2][n_slices][to_add_col], worksheets_n_slices[n_sub][2][n_slices][2]+to_add_col, float(dc))
                        worksheets_n_slices[n_sub][2][n_slices][to_add_col] += 1

workbook2.close()


# ############################# MULTIPLE LOOCV FROM MAGMA :  get level data #############################
workbook3 = xl.Workbook('results_wmseg_dices_by_level.xlsx')
worksheet_36 = workbook3.add_worksheet('36subjects')
worksheet_28 = workbook3.add_worksheet('28subjects')
worksheet_20 = workbook3.add_worksheet('20subjects')
worksheet_15 = workbook3.add_worksheet('15subjects')
worksheet_10 = workbook3.add_worksheet('10subjects')
worksheet_5 = workbook3.add_worksheet('5subjects')
worksheet_2 = workbook3.add_worksheet('2subjects')

bold = workbook3.add_format({'bold': True})

init_row = 2
init_col = 1
worksheets_levels = {'36subjects': [worksheet_36, init_col, {}], '28subjects': [worksheet_28, init_col, {}], '20subjects': [worksheet_20, init_col, {}], '15subjects': [worksheet_15, init_col, {}], '10subjects': [worksheet_10, init_col, {}], '5subjects': [worksheet_5, init_col, {}], '2subjects': [worksheet_2, init_col, {}]}

for w in worksheets_levels.values():
    w[0].write(0, 0, 'Number of slices in model', bold)
    w[0].write(1, 0, 'Gamma', bold)

for loocv_dir in os.listdir(path):
    if os.path.isdir(path + '/' + loocv_dir) and 'dictionary' not in loocv_dir:
        words = loocv_dir.split('_')
        n_sub = words[0]
        print n_sub
        mod = int(words[1])
        gamma = words[-1]
        if gamma == '1.2':
            to_add_col = 0
        else:
            to_add_col = 1

        dice_file = open(path + '/' + loocv_dir + '/wm_dice_by_level.txt', 'r')
        dice_lines = dice_file.readlines()
        dice_file.close()

        dice_lines = [line.split(':') for line in dice_lines]

        for line in dice_lines:
            if line[0] != '\n' and line[1] != ' \n':
                level = line[0]
                level = level.replace(' ', '')
                if level not in worksheets_levels[n_sub][2].keys():
                    worksheets_levels[n_sub][2][level] = [init_row, init_row, worksheets_levels[n_sub][1]]
                    worksheets_levels[n_sub][0].write(0, worksheets_levels[n_sub][1], level)
                    worksheets_levels[n_sub][0].write(0, worksheets_levels[n_sub][1]+1, level)

                    worksheets_levels[n_sub][0].write(1, worksheets_levels[n_sub][1], 1.2)
                    worksheets_levels[n_sub][0].write(1, worksheets_levels[n_sub][1]+1, 0)

                    worksheets_levels[n_sub][1] += 2
                    print 'levels line dice', line
                    for i, dc in enumerate(line[1].split(',')):
                        worksheets_levels[n_sub][0].write(worksheets_levels[n_sub][2][level][to_add_col], worksheets_levels[n_sub][2][level][2]+to_add_col, float(dc))
                        worksheets_levels[n_sub][2][level][to_add_col] += 1
                else:
                    for i, dc in enumerate(line[1].split(',')):
                        worksheets_levels[n_sub][0].write(worksheets_levels[n_sub][2][level][to_add_col], worksheets_levels[n_sub][2][level][2]+to_add_col, float(dc))
                        worksheets_levels[n_sub][2][level][to_add_col] += 1

workbook3.close()
'''

# ############################# MULTIPLE LOOCV FROM MAGMA - HAUSDORFF DISTANCE #############################
# Create a workbook and add a worksheet.
workbook = xl.Workbook('results_hausdorff_dist.xlsx')
worksheet_36 = workbook.add_worksheet('36subjects')
worksheet_28 = workbook.add_worksheet('28subjects')
worksheet_20 = workbook.add_worksheet('20subjects')
worksheet_15 = workbook.add_worksheet('15subjects')
worksheet_10 = workbook.add_worksheet('10subjects')
worksheet_5 = workbook.add_worksheet('5subjects')
worksheet_2 = workbook.add_worksheet('2subjects')

bold = workbook.add_format({'bold': True})


worksheets = {'36subjects': [worksheet_36, 2, {}], '28subjects': [worksheet_28, 2, {}], '20subjects': [worksheet_20, 2, {}], '15subjects': [worksheet_15, 2, {}], '10subjects': [worksheet_10, 2, {}], '5subjects': [worksheet_5, 2, {}], '2subjects': [worksheet_2, 2, {}]}

for w in worksheets.values():
    w[0].write(1, 0, 'Subject', bold)
    w[0].write(1, 1, 'Slice #', bold)
    w[0].write(1, 2, 'Slice level', bold)

    for mod in range(8):
        w[0].write(1, 3+3*mod, 'HD - gamma 1.2', bold)
        w[0].write(1, 3+3*mod+1, 'HD - gamma 0', bold)
        w[0].write(1, 3+3*mod+2, 'Number of model slices', bold)
        w[0].merge_range(0, 3+3*mod, 0, 3+3*mod+2, str(mod), bold)


for loocv_dir in os.listdir(path):
    if os.path.isdir(path + '/' + loocv_dir) and 'dictionary' not in loocv_dir:
        words = loocv_dir.split('_')
        n_sub = words[0]
        mod = int(words[1])
        gamma = words[-1]
        if gamma == '1.2':
            to_add_col = 0
        else:
            to_add_col = 1

        # something = get_hausdorff_dists(path + '/' + loocv_dir)
        # TODO : continuer result extraction ici
        dice_file = open(path + '/' + loocv_dir + '/wm_dice_coeff.txt', 'r')
        dice_lines = dice_file.readlines()
        dice_file.close()

        dice_lines = [line.split(' ') for line in dice_lines]

        col = 0
        for line in dice_lines:
            line[2] = line[2][:-1]
            line[-1] = line[-1][:-1]
            line.pop(4)
            line.pop(4)
            slice_id = '_'.join(line[:2])
            if slice_id not in worksheets[n_sub][2].keys():
                worksheets[n_sub][2][slice_id] = worksheets[n_sub][1]
                for i, v in enumerate(line[:3]):
                    worksheets[n_sub][0].write(worksheets[n_sub][1], col+i, v)
                worksheets[n_sub][0].write(worksheets[n_sub][1], 3+3*mod+to_add_col, float(line[3]))
                worksheets[n_sub][0].write(worksheets[n_sub][1], 3+3*mod+2, float(line[-1]))
                worksheets[n_sub][1] += 1
            else:
                worksheets[n_sub][0].write(worksheets[n_sub][2][slice_id], 3+3*mod+to_add_col, float(line[3]))
                worksheets[n_sub][0].write(worksheets[n_sub][2][slice_id], 3+3*mod+2, float(line[-1]))

workbook.close()


def get_hausdorff_dists(mod_path):
    from sct_asman import Param, Model
    hd_dic = {}
    original_path = os.path.abspath('.')
    os.chdir(mod_path)
    for sub_dir in os.listdir('.'):
        if os.path.isdir(sub_dir) and 'dictionary' not in sub_dir:
            os.chdir(sub_dir)
            model_dir = 'gm_seg_model_data/'
            par = Param()
            par.path_dictionary = './' + model_dir
            par.todo_model = 'load'
            model = Model(model_param=par)

            subject = '_'.join(sub_dir.split('_')[1:3])
            if 'pilot' in subject:
                subject = subject.split('_')[0]

            for file_name in os.listdir('./' + subject + '/'):
                if 'im' in file_name:
                    slice_name = sct.extract_fname(file_name)[1][:-2]
                    slice_level = slice_name.split('_')[-2]
                    ref = slice_name + '_seg' + sct.extract_fname(file_name)[2]

                    #todo: read selected slice file
                    target_seg = TargetSegmentationPairwiseNOTFULL(model, target_image=Image(file_name), levels_image=slice_level)
                    res_im = Image(param=target_seg.target[0].gm_seg, absolutepath='./' + slice_name + '_res_gmseg.nii.gz')
                    res_im.save()
                    res = res_im.file_name + res_im.ext

                    hausdorff_fname = 'hausdorff_distance_' + slice_name
                    sct.run('sct_compute_hausdorff_distance.py  -i ' + res + ' -r ' + ref + ' -o ' + hausdorff_fname)
                    hd_fic = open(hausdorff_fname, 'r')
                    hd = hd_fic.readline().split(' ')[-2]
                    hd_fic.readline()
                    hd_fic.close()
                    med1 = hd_fic.readline().split(' ')[-2]
                    med2 = hd_fic.readline().split(' ')[-2]
                    med = max(med1, med2)
                    hd_dic[slice_name] = (hd, med)

            os.chdir('..')
    os.chdir(original_path)
    return hd_dic


# ----------------------------------------------------------------------------------------------------------------------
# TARGET SEGMENTATION PAIRWISE -----------------------------------------------------------------------------------------
class TargetSegmentationPairwiseNOTFULL:
    """
    Contains all the function to segment the gray matter an a target image given a model

        - registration of the target to the model space

        - projection of the target slices on the reduced model space

        - selection of the model slices most similar to the target slices

        - computation of the resulting target segmentation by label fusion of their segmentation
    """
    def __init__(self, model, target_image=None, levels_image=None, selected_slices_id=None):
        """
        Target gray matter segmentation constructor

        :param model: Model used to compute the segmentation, type: Model

        :param target_image: Target image to segment gray matter on, type: Image

        """
        self.model = model

        # Initialization of the target image
        self.target = [Slice(slice_id=0, im=target_image.data, reg_to_m=[])]
        self.target_dim = 2

        self.target[0].set(level=get_key_from_val(self.model.dictionary.level_label, levels_image.upper()))

        self.target_pairwise_registration()

        self.selected_k_slices = [False]*self.model.dictionary.J

        for i in range(self.model.dictionary.J):
            if i in selected_slices_id:
                self.selected_k_slices[i] = True

        self.model.label_fusion(self.target, self.selected_k_slices, type=self.model.param.res_type)

        sct.printv('\nRegistering the result gray matter segmentation back into the target original space...',
                   model.param.verbose, 'normal')
        self.target_pairwise_registration(inverse=True)

    # ------------------------------------------------------------------------------------------------------------------
    def target_pairwise_registration(self, inverse=False):
        """
        Register the target image into the model space

        Affine (or rigid + affine) registration of the target on the mean model image --> pairwise

        :param inverse: if True, apply the inverse warping field of the registration target -> model space
        to the result gray matter segmentation of the target
        (put it back in it's original space)

        :return None: the target attributes are set in the function
        """
        if not inverse:
            # Registration target --> model space
            mean_dic_im = self.model.pca.mean_image
            for i, target_slice in enumerate(self.target):
                if not self.model.param.first_reg:
                    moving_target_slice = target_slice.im
                else:
                    moving_target_slice = target_slice.im_M
                for transfo in self.model.dictionary.coregistration_transfos:
                    transfo_name = transfo + '_transfo_target2model_space_slice_' + str(i) + find_ants_transfo_name(transfo)[0]
                    target_slice.reg_to_M.append((transfo, transfo_name))

                    moving_target_slice = apply_ants_transfo(mean_dic_im, moving_target_slice, binary=False, transfo_type=transfo, transfo_name=transfo_name)
                self.target[i].set(im_m=moving_target_slice)

        else:
            # Inverse registration result in model space --> target original space
            for i, target_slice in enumerate(self.target):
                moving_wm_seg_slice = target_slice.wm_seg_M
                moving_gm_seg_slice = target_slice.gm_seg_M

                for transfo in target_slice.reg_to_M:
                    if self.model.param.res_type == 'binary':
                        bin = True
                    else:
                        bin = False
                    moving_wm_seg_slice = apply_ants_transfo(self.model.dictionary.mean_seg, moving_wm_seg_slice, search_reg=False, binary=bin, inverse=1, transfo_type=transfo[0], transfo_name=transfo[1])
                    moving_gm_seg_slice = apply_ants_transfo(self.model.dictionary.mean_seg, moving_gm_seg_slice, search_reg=False, binary=bin, inverse=1, transfo_type=transfo[0], transfo_name=transfo[1])

                target_slice.set(wm_seg=moving_wm_seg_slice)
                target_slice.set(gm_seg=moving_gm_seg_slice)



'''
# ############################# CLASSICAL LOOCV #############################

workbook = xl.Workbook('results_loocv_subjects_resampled.xlsx')

worksheet_wm = workbook.add_worksheet('wm_dice')
worksheet_gm = workbook.add_worksheet('gm_dice')

worksheet_csa = workbook.add_worksheet('CSA')

bold = workbook.add_format({'bold': True, 'text_wrap': True})

init_col = 3
dic_worksheets = {worksheet_wm: init_col, worksheet_gm: init_col, worksheet_csa: init_col}

for w in dic_worksheets.keys():
    w.write(1, 0, 'Subject', bold)
    w.write(1, 1, 'Slice #', bold)
    w.write(1, 2, 'Slice level', bold)

len_gm_lines = 0
len_wm_lines = 0

for loocv_dir in os.listdir(path):
    if os.path.isdir(path + '/' + loocv_dir):
        words = loocv_dir.split('_')
        levels = words[0] + ' ' + words[1]
        gamma = words[2]
        reg = words[-1]
        info = levels + ' - ' + gamma + ' - ' + reg

        # GM DICE
        gm_dice = open(path + '/' + loocv_dir + '/gm_dice_coeff.txt')
        gm_dice_lines = gm_dice.readlines()
        gm_dice.close()

        gm_dice_lines = [line.split(' ') for line in gm_dice_lines]

        if dic_worksheets[worksheet_gm] == init_col:
            worksheet_gm.write(1, dic_worksheets[worksheet_gm], 'Number of slices in model', bold)
            dic_worksheets[worksheet_gm] += 1
            for i, line in enumerate(gm_dice_lines):
                worksheet_gm.write(i+2, 0, line[0])
                worksheet_gm.write(i+2, 1, line[1])
                worksheet_gm.write(i+2, 2, line[2][:-1])
                worksheet_gm.write(i+2, 3, int(line[-1][:-1]))
                res = float(line[3])
                worksheet_gm.write(i+2, dic_worksheets[worksheet_gm], res)  # RESULT
            len_gm_lines = len(gm_dice_lines)
        else:
            for i, line in enumerate(gm_dice_lines):
                res = float(line[3])
                worksheet_gm.write(i+2, dic_worksheets[worksheet_gm], res)  # RESULT
        worksheet_gm.write(0, dic_worksheets[worksheet_gm], info, bold)
        dic_worksheets[worksheet_gm] += 1

        # WM DICE
        wm_dice = open(path + '/' + loocv_dir + '/wm_dice_coeff.txt')
        wm_dice_lines = wm_dice.readlines()
        wm_dice.close()

        wm_dice_lines = [line.split(' ') for line in wm_dice_lines]

        if dic_worksheets[worksheet_wm] == init_col:
            worksheet_wm.write(1, dic_worksheets[worksheet_wm], 'Number of slices in model', bold)
            dic_worksheets[worksheet_wm] += 1
            for i, line in enumerate(wm_dice_lines[:-1]):
                worksheet_wm.write(i+2, 0, line[0])
                worksheet_wm.write(i+2, 1, line[1])
                worksheet_wm.write(i+2, 2, line[2][:-1])
                worksheet_wm.write(i+2, 3, int(line[-1][:-1]))
                if line[3] != '':
                    res = float(line[3])
                else:
                    res = 0
                worksheet_wm.write(i+2, dic_worksheets[worksheet_wm], res)  # RESULT
            len_wm_lines = len(wm_dice_lines)

        else:
            for i, line in enumerate(wm_dice_lines[:-1]):
                if line[3] != '':
                    res = float(line[3])
                else:
                    res = 0
                worksheet_wm.write(i+2, dic_worksheets[worksheet_wm], res)  # RESULT
        worksheet_wm.write(0, dic_worksheets[worksheet_wm], info, bold)
        dic_worksheets[worksheet_wm] += 1

        # CSA
        wm_csa = open(path + '/' + loocv_dir + '/wm_csa.txt')
        wm_csa_lines = wm_csa.readlines()
        wm_csa.close()
        wm_csa_lines = [line.split(' ') for line in wm_csa_lines]

        gm_csa = open(path + '/' + loocv_dir + '/gm_csa.txt')
        gm_csa_lines = gm_csa.readlines()
        gm_csa.close()
        gm_csa_lines = [line.split(' ') for line in gm_csa_lines]

        worksheet_csa.write(1, dic_worksheets[worksheet_csa], 'WM', bold)
        worksheet_csa.write(1, dic_worksheets[worksheet_csa]+1, 'GM', bold)

        if dic_worksheets[worksheet_csa] == init_col:
            for i, csa_lines in enumerate(zip(wm_csa_lines, gm_csa_lines)):
                wm_line, gm_line = csa_lines
                worksheet_csa.write(i+2, 0, wm_line[0])
                worksheet_csa.write(i+2, 1, wm_line[1])
                worksheet_csa.write(i+2, 2, wm_line[2][:-1])

                res_wm = float(wm_line[3])
                res_gm = float(gm_line[3])

                worksheet_csa.write(i+2, dic_worksheets[worksheet_csa], res_wm)
                worksheet_csa.write(i+2, dic_worksheets[worksheet_csa]+1, res_gm)
        else:
            for i, csa_lines in enumerate(zip(wm_csa_lines, gm_csa_lines)):
                wm_line, gm_line = csa_lines
                res_wm = float(wm_line[3])
                res_gm = float(gm_line[3])

                worksheet_csa.write(i+2, dic_worksheets[worksheet_csa], res_wm)
                worksheet_csa.write(i+2, dic_worksheets[worksheet_csa]+1, res_gm)

        worksheet_csa.merge_range(0,dic_worksheets[worksheet_csa], 0, dic_worksheets[worksheet_csa]+1, info, bold)
        dic_worksheets[worksheet_csa] += 2

# ##  CONDITIONAL FORMATTING  ##
good = workbook.add_format({'font_color': '#006600', 'bg_color': '#CCFFCC'})
med = workbook.add_format({'font_color': '#CC6600', 'bg_color': '#FDED83'})
bad = workbook.add_format({'font_color': '#990000', 'bg_color': '#FFCCCC'})

gm_lim = chr(dic_worksheets[worksheet_gm] + 64) + str(len_gm_lines)
worksheet_gm.conditional_format('E3:' + gm_lim, {'type':     'cell',
                                        'criteria': '>=',
                                        'value':    0.9,
                                        'format':   good})
worksheet_gm.conditional_format('E3:' + gm_lim, {'type':     'cell',
                                        'criteria': 'between',
                                        'minimum': 0.8,
                                        'maximum': 0.85,
                                        'format':   med})
worksheet_gm.conditional_format('E3:' + gm_lim, {'type':     'cell',
                                        'criteria': '<',
                                        'value':    0.8,
                                        'format':   bad})


wm_lim = chr(dic_worksheets[worksheet_wm] + 64) + str(len_wm_lines)
worksheet_wm.conditional_format('E3:' + wm_lim, {'type':     'cell',
                                        'criteria': '>=',
                                        'value':    0.9,
                                        'format':   good})
worksheet_wm.conditional_format('E3:' + wm_lim, {'type':     'cell',
                                        'criteria': 'between',
                                        'minimum': 0.8,
                                        'maximum': 0.85,
                                        'format':   med})
worksheet_wm.conditional_format('E3:' + wm_lim, {'type':     'cell',
                                        'criteria': '<',
                                        'value':    0.8,
                                        'format':   bad})
workbook.close()

'''
