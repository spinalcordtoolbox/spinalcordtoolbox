#!/usr/bin/env python
import xlsxwriter as xl
import os

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
