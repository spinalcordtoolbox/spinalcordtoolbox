#!/usr/bin/env python
import xlsxwriter as xl
import os

path = '.'

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
    if os.path.isdir(path + '/' + loocv_dir):
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
                worksheets[n_sub][0].write(worksheets[n_sub][1], 3+3*mod+to_add_col, line[3])
                worksheets[n_sub][0].write(worksheets[n_sub][1], 3+3*mod+2, line[-1])
                worksheets[n_sub][1] += 1
            else:
                worksheets[n_sub][0].write(worksheets[n_sub][2][slice_id], 3+3*mod+to_add_col, line[3])
                worksheets[n_sub][0].write(worksheets[n_sub][2][slice_id], 3+3*mod+2, line[-1])




workbook.close()