#!/usr/bin/env python
import sys
'''
dice_file = open('./dice_coeffs.txt', 'r')
data_lines = dice_file.readline().split('\r')
dice_file.close()

seg_type = data_lines[0].split('\t')
reg = data_lines[1].split('\t')
use_levels = data_lines[2].split('\t')
model_type = data_lines[3].split('\t')

case1_dic = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': []}
case2_dic = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': []}
case3_dic = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': []}
case4_dic = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': []}
case5_dic = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': []}
case6_dic = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': []}


for slice_line in data_lines[4:]:
    slice_line = slice_line.split('\t')

    level = slice_line[2]
    case1_dic[level].append(slice_line[3])
    case2_dic[level].append(slice_line[4])
    case3_dic[level].append(slice_line[5])
    case4_dic[level].append(slice_line[6])
    case5_dic[level].append(slice_line[7])
    case6_dic[level].append(slice_line[8])


data_by_level = open('data_by_level.txt', 'w')

data_by_level.write('WM SEG - RIGID_AFFINE - NO LEVELS - MODEL WM\n')
for level in case1_dic.keys():
    s_dices = ''
    for dice in case1_dic[level]:
        s_dices += dice + ' , '
    data_by_level.write(level + ' : ' + s_dices[:-2] + '\n')
data_by_level.write('\n\n')


data_by_level.write('WM SEG - RIGID_AFFINE - WITH LEVELS - MODEL WM\n')
for level in case2_dic.keys():
    s_dices = ''
    for dice in case2_dic[level]:
        s_dices += dice + ' , '
    data_by_level.write(level + ' : ' + s_dices[:-2] + '\n')
data_by_level.write('\n\n')


data_by_level.write('WM SEG - AFFINE - NO LEVELS - MODEL WM\n')
for level in case3_dic.keys():
    s_dices = ''
    for dice in case3_dic[level]:
        s_dices += dice + ' , '
    data_by_level.write(level + ' : ' + s_dices[:-2] + '\n')
data_by_level.write('\n\n')


data_by_level.write('WM SEG - AFFINE - WITH LEVELS - MODEL WM\n')
for level in case4_dic.keys():
    s_dices = ''
    for dice in case4_dic[level]:
        s_dices += dice + ' , '
    data_by_level.write(level + ' : ' + s_dices[:-2] + '\n')
data_by_level.write('\n\n')


data_by_level.write('GM SEG - AFFINE - WITH LEVELS - MODEL GM\n')
for level in case5_dic.keys():
    s_dices = ''
    for dice in case5_dic[level]:
        s_dices += dice + ' , '
    data_by_level.write(level + ' : ' + s_dices[:-2] + '\n')
data_by_level.write('\n\n')


data_by_level.write('GM SEG - AFFINE - WITH LEVELS - MODEL WM\n')
for level in case6_dic.keys():
    s_dices = ''
    for dice in case6_dic[level]:
        s_dices += dice + ' , '
    data_by_level.write(level + ' : ' + s_dices[:-2] + '\n')
data_by_level.write('\n\n')

data_by_level.close()
'''

## USUAL CASE THAT WORKS ##
'''
dic = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': []}

dice_file = open('./wm_dice_coeff.txt', 'r')
data_line = dice_file.readline()
# data_line = data_line.split('\r')
while data_line != '\n' and data_line != '':
    # for slice_line in data_line[:-2]:
    slice_line = data_line.split(' ')

    if len(slice_line) == 1:
        slice_line = slice_line[0].split('\t')

    # level = slice_line[-1][1:-2]
    level = slice_line[2][:-1]
    dic[level].append(slice_line[3])
    data_line = dice_file.readline()
dice_file.close()


data_by_level = open('wm_dice_by_level.txt', 'w')

for level in dic.keys():
    s_dices = ''
    for dice in dic[level]:
        s_dices += dice + ' , '
    data_by_level.write(level + ' : ' + s_dices[:-2] + '\n')
data_by_level.write('\n\n')


data_by_level.close()
'''


# #################  WM DICE COEFF
"""
path = sys.argv[1]
n_slice_dic = {}
level_dic = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': []}

dice_file = open(path + '/wm_dice_coeff.txt', 'r')
data_line = dice_file.readline()
# data_line = data_line.split('\r')
while data_line != '\n' and data_line != '':
    # for slice_line in data_line[:-2]:
    slice_line = data_line.split(' ')

    if len(slice_line) == 1:
        slice_line = slice_line[0].split('\t')

    n_slices = slice_line[-1][:-1]
    level = slice_line[2][:-1]

    level_dic[level].append(slice_line[3])
    if n_slices not in n_slice_dic.keys():
        n_slice_dic[n_slices] = [slice_line[3]]
    else:
        n_slice_dic[n_slices].append(slice_line[3])

    data_line = dice_file.readline()
dice_file.close()


'''
for slice_line in data_lines:
    slice_line = slice_line.split('\t')

    level = slice_line[-1][1:-1]
    dic[level].append(slice_line[1])
'''

data_by_level = open(path + '/wm_dice_by_level.txt', 'w')

for level in level_dic.keys():
    s_dices = ''
    for dice in level_dic[level]:
        s_dices += dice + ' , '
    data_by_level.write(level + ' : ' + s_dices[:-2] + '\n')
data_by_level.write('\n\n')


data_by_level.close()

data_by_n_slice = open(path + '/wm_dice_by_n_slice.txt', 'w')

for n_slice in n_slice_dic.keys():
    s_dices = ''
    for dice in n_slice_dic[n_slice]:
        s_dices += dice + ' , '
    data_by_n_slice.write(str(n_slice) + ' : ' + s_dices[:-2] + '\n')
data_by_n_slice.write('\n\n')


data_by_n_slice.close()
"""

'''
# #################  CSA
path = sys.argv[1]
level_dic = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': []}

sc_part = 'wm'

dice_file = open(path + '/' + sc_part + '_csa.txt', 'r')
data_line = dice_file.readline()
# data_line = data_line.split('\r')
while data_line != '\n' and data_line != '':
    # for slice_line in data_line[:-2]:
    slice_line = data_line.split(' ')

    if len(slice_line) == 1:
        slice_line = slice_line[0].split('\t')

    # n_slices = slice_line[-1][:-1]
    level = slice_line[2][:-1]

    level_dic[level].append(slice_line[3][:-1])

    data_line = dice_file.readline()
dice_file.close()




data_by_level = open(path + '/' + sc_part + '_csa_by_level.txt', 'w')

for level in level_dic.keys():
    s_dices = ''
    for dice in level_dic[level]:
        s_dices += dice + ' , '
    data_by_level.write(level + ' : ' + s_dices[:-2] + '\n')
data_by_level.write('\n\n')


data_by_level.close()
'''

# #################  HAUSDORFF DISTANCE RESULTS SAVED AS TXT
fic_name = sys.argv[1]


fic = open(fic_name, 'r')
col_title = fic.readline().split('\t')
lines = fic.readline().split('\r')
fic.close()

data_dic = {}
mod_list = col_title[3:]
mod_list[-1] = mod_list[-1][:-2]
for modality in mod_list:
    data_dic[modality] = {'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [], 'C6': [], 'C7': [], 'T1': [], 'T2': []}

for j, line in enumerate(lines):
    l = line.split('\t')
    level = l[2]
    for i, mod in enumerate(mod_list):
        data_dic[mod][level].append(l[i+3])

for mod in mod_list:
    data_by_level = open(mod + '_hausdorff_dist_by_level.txt', 'w')

    for level in data_dic[mod].keys():
        str_hd = ''
        for hd in data_dic[mod][level]:
            str_hd += hd + ' , '
        data_by_level.write(level + ' : ' + str_hd[:-2] + '\n')
    data_by_level.close()






