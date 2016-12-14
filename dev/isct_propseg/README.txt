sct_segmentation_propagation

NeuroPoly, Polytechnique Montreal <www.neuropoly.info> 
Author: Benjamin De Leener
Version 0.2 - 2014-03-13
Compiled for OSX 10.6, 10.7, 10.8

This program segments the spinal cord in MR images (T1- or T2-weighted).

Usage: (-verbose option can be useful…)
./sct_segmentation_propagation -i example_data/t1.nii.gz -o example_data/results_t1/ -t t1
./sct_segmentation_propagation -i example_data/t2.nii.gz -o example_data/results_t2/ -t t2

Help for more options (as verbose, initialization, cross-sectional areas, etc):
./sct_segmentation_propagation -help


Some permission issues can appear. Please allow full permission on sct_segmentation_propagation (chmod).