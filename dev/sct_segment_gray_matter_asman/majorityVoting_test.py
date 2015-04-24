#!/usr/bin/env python

from msct_gmseg_utils import *
from sct_asman import ModelDictionary
import sys
import os
import time

seg_dataset = []

path = sys.argv[1]

for file_name in os.listdir(path):
    seg_dataset.append(Image(path + file_name).data)


before_old_version = time.time()

dic = ModelDictionary()
dic.N = 2025
dic.L = [0, 1]
res_old_version = dic.compute_majority_vote_mean_seg_old_version(seg_dataset)

Image(param=res_old_version, absolutepath='./res_old_version.nii.gz').save()
print 'Old version in ', time.time() - before_old_version, ' sec'


before_new_version = time.time()

res_new_version = compute_majority_vote_mean_seg(seg_dataset)

Image(param=res_new_version, absolutepath='./res_new_version.nii.gz').save()

print 'New version in ', time.time() - before_new_version, ' sec'

