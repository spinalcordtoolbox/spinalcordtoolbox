#!/bin/bash
# set environment variables for the study.
# Julien Cohen-Adad 2019-04-02

# Set every other path relative to the location of this script
export PATH_PARENT="/Users/nipin_local"

# path to input data (do not add "/" at the end). This path should be absolute (i.e. do not use ".")
export PATH_DATA="${PATH_PARENT}/unf_test"

# Path where to save results (do not add "/" at the end).
export PATH_OUTPUT="${PATH_PARENT}/results_hogancest"
export PATH_QC="${PATH_PARENT}/results_qc_hogancest"
export PATH_LOG="${PATH_PARENT}/results_log_hogancest"

# Misc
export JOBS=4  # Number of jobs for parallel processing
