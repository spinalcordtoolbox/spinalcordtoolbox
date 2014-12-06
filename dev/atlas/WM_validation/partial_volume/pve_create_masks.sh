#!/usr/bin/env python
# create masks for partial volume estimation: CSF, WM and GM.
# author: Julien Cohen-Adad
# 2014-11-26

import glob
import os 
import sys
import commands
import re
import numpy as np
import math
status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
import sct_utils as sct

# param for validation
bootstrap_iter=2
path_atlas="../WM_atlas_generation/WMtracts_outputs/final_results"
user_tract=charles_tract_,julien_tract_,tanguy_tract_,simon_tract_
std_noise=(0.0000000001 1 5 10 20 50)
range_tract=(0 5 10 20 50)
fixed_range=10
fixed_noise=10
results_folder="results"
results_folder_npv=$results_folder/no_partial_vol_correction
results_folder_pv=$results_folder/partial_vol_correction

# start batch
mkdir $results_folder
mkdir $results_folder_npv
mkdir $results_folder_pv

for i in "${std_noise[@]}"
do
    python validate_atlas.py -a $path_atlas -b $bootstrap_iter -n $i -s $fixed_range -r $results_folder_npv/tract_noise$i.txt -d $user_tract
    python validate_atlas.py -a $path_atlas -b $bootstrap_iter -n $i -s $fixed_range -r $results_folder_pv/tract_noise$i.txt -d $user_tract -p
    cat $results_folder_npv/tract_noise$i.txt >> $results_folder/nopvnoise.txt
    cat $results_folder_pv/tract_noise$i.txt >> $results_folder/pvnoise.txt
done

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
