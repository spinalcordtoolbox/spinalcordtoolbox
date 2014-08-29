#!/bin/bash
bootstrap_iter=2
user_tract=charles_tract_,julien_tract_,tanguy_tract_,simon_tract_
results_folder=atlas_validation
results_folder_npv=$results_folder/no_partial_vol_correction
results_folder_pv=$results_folder/partial_vol_correction
#****This batch_man_test code will work only if there is 4 users exactly, otherwise use batch_atlas_validation.py
code_validation=batch_man_test__julien.py
mkdir $results_folder
mkdir $results_folder_npv
mkdir $results_folder_pv
std_noise=(0.0000000001 1 5 10 20 50)
range_tract=(0 5 10 20 50)
fixed_range=10
fixed_noise=10

for i in "${std_noise[@]}"
do
    python $code_validation -b $bootstrap_iter -n $i -s $fixed_range -r $results_folder_npv/tract_noise$i.txt -d $user_tract
    python $code_validation -b $bootstrap_iter -n $i -s $fixed_range -r $results_folder_pv/tract_noise$i.txt -d $user_tract -p
    cat $results_folder_npv/tract_noise$i.txt >> $results_folder/nopvnoise.txt
    cat $results_folder_pv/tract_noise$i.txt >> $results_folder/pvnoise.txt
done

for i in "${range_tract[@]}"
do
    python $code_validation -b $bootstrap_iter -n $fixed_noise -s $i -r $results_folder_npv/noise_tract$i.txt -d $user_tract
    python $code_validation -b $bootstrap_iter -n $fixed_noise -s $i -r $results_folder_pv/noise_tract$i.txt -d $user_tract -p
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
