#!/usr/bin/env python
#########################################################################################
#
# Validation of WM atlas
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Charles Naaman, Julien Cohen-Adad
# Modified: 2014-11-25
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import shutil

# param for validation
bootstrap_iter = 2
path_atlas = '../WM_atlas_generation/WMtracts_outputs/final_results/'  # add / at the end
user_tract = 'charles_tract_,julien_tract_,tanguy_tract_,simon_tract_'
std_noise_list = [0.0000000001 1 5 10 20 50]
range_tract = [0 5 10 20 50]
fixed_range = 10
fixed_noise = 10
results_folder = 'results/'  # add / at the end

# create output folder
create_folder(results_folder)

for std_noise in std_noise_list:
    validate_atlas(path_atlas, bootstrap_iter, std_noise, fixed_range, results_folder+str(std_noise), user_tract)
#    cat $results_folder_npv/tract_noise$i.txt >> $results_folder/nopvnoise.txt

for i in "${range_tract[@]}"
do
    python validate_atlas.py -a $path_atlas -b $bootstrap_iter -n $fixed_noise -s $i -r $results_folder_npv/noise_tract$i.txt -d $user_tract
    python validate_atlas.py -a $path_atlas -b $bootstrap_iter -n $fixed_noise -s $i -r $results_folder_pv/noise_tract$i.txt -d $user_tract -p
    cat $results_folder_npv/noise_tract$i.txt >> $results_folder/nopvtract.txt
    cat $results_folder_pv/noise_tract$i.txt >> $results_folder/pvtract.txt
done
echo "no partial volume correction, noise variation"
cat $results_folder/nopvnoise.txt
echo "partial volume correction, noise variation"
cat $results_folder/pvnoise.txt
echo "no partial volume correction, tract range variation"
cat $results_folder/nopvtract.txt
echo "partial volume correction, tract range variation"
cat $results_folder/pvtract.txt


def create_folder(folder):
    """create folder-- delete if already exists"""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)