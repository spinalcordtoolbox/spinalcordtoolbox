#!/usr/bin/env python
import sct_utils as sct
import sys
from msct_parser import Parser
from msct_gmseg_utils import leave_one_out_by_subject
import os
import time
import multiprocessing as mp


def loocv(param):
    use_level, weight = param
    sct.run('mkdir ./' + registration + '_levels_' + str(use_level) + '_weight' + str(weight) )
    sct.run('cp -r ' + path_dictionary + ' ./' + registration + '_levels_' + str(use_level) + '_weight' + str(weight) + '/dictionary')
    os.chdir('./' +registration + '_levels_' + str(use_level) + '_weight' + str(weight))
    # leave_one_out_by_slice('dictionary/', reg=registration, target_reg=target_reg, use_levels=use_level, weight=weight)
    leave_one_out_by_subject('dictionary/', use_levels=use_level, weight=weight)
    os.chdir('..')

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

    weights = [1.2]


    before = time.time()
    for w in weights:
        loocv((True, w))
    loocv((False, 0))
    t1 = time.time() - before

    '''
    before = time.time()
    par = zip([True]*(len(weights) - 1)+[False], weights)
    pool = mp.Pool(8)
    pool.map(loocv, par)
    t2 = time.time() - before
    '''
    print 'normal : ', t1
    # print 'with mp: ', t2
