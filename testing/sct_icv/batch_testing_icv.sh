#!/bin/bash
#
# testing sct_icv.py

sct_icv.py -i data/errsm_20_t1.nii.gz -c t1 -o tmp.output_sienax -d sienax

sct_icv.py -i data/errsm_20_t1.nii.gz -c t1 -o tmp.output_rbm -d rbm

