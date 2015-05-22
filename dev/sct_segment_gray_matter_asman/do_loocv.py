#!/usr/bin/env python
import sct_utils as sct
import sys
from msct_parser import Parser
from msct_gmseg_utils import leave_one_out_by_slice
import os

if __name__ == '__main__':
    parser = Parser(__file__)
    parser.usage.set_description('Gray matter segmentation with several types of registration')
    parser.add_option(name="-dic",
                      type_value="folder",
                      description="Path to the dictionary of images",
                      mandatory=True,
                      example='/home/jdoe/data/dictionary')
    parser.add_option(name="-target-reg",
                      type_value='multiple_choice',
                      description="type of registration of the target to the model space "
                                  "(if pairwise, the registration applied to the target are the same as"
                                  " those of the -reg flag)",
                      mandatory=False,
                      default_value='pairwise',
                      example=['pairwise', 'groupwise'])

    arguments = parser.parse(sys.argv[1:])
    target_reg = 'pairwise'
    cmd_gmseg = ''
    if "-target-reg" in arguments:
        target_reg = arguments["-target-reg"]
    path_dictionary = arguments["-dic"]

    registration = 'Affine'

    weights = [1.1, 1.2, 1.5, 1.8, 2.1]

    for w in weights:
        sct.run('mkdir ./' + registration + '_with_levels_weight' + str(w))
        sct.run('cp -r ' + path_dictionary + ' ./' + registration + '_with_levels_weight' + str(w) + '/dictionary')
        os.chdir('./' + registration + '_with_levels_weight' + str(w))
        leave_one_out_by_slice('dictionary/', reg=registration, target_reg=target_reg, use_levels=True, weight=w)
        os.chdir('..')

        sct.run('mkdir ./' + registration + '_without_levels' + str(w))
        sct.run('cp -r ' + path_dictionary + ' ./' + registration + '_without_levels' + str(w) + '/dictionary')
        os.chdir('./' + registration + '_without_levels' + str(w))
        leave_one_out_by_slice('dictionary/', reg=registration, target_reg=target_reg, use_levels=False, weight=w)
        os.chdir('..')
