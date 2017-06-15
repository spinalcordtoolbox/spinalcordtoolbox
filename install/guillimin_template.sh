#!/usr/bin/env bash
#PBS -l procs=101  # This will reserve 101 processors for your computation
                   #  including 1 for orchestration -> need a minimum of a 100 subjects

#PBS -l walltime=03:00:00   # After 3 hours the computation will be shut down

#PBS -N MY_NAME # Name given to you job in the queuing system

#PBS -A rrp-355-aa # Neuropoly project number on guillimin


# Source the SCT where it is installed
# make sure the SCT was installed with the --mpi option
# If the sct was already installed witout the option, just rerun it with  ./install_sct --mpi -dmb
source $HOME/path/to/my/spinalcordtoolbox/bin/sct_launcher

# cd to a scratch space where outputs will be saved
cd  $SCRATCH/my_work_dir

DATA_PATH=/path_to/the_subjects/data/

# to run propseg on subjects t2s with 100 cpu --> need a minimum of a 100 subjects
sct_pipeline  -f sct_propseg -d $DATA_PATH  -p  \"-i t2s/t2s.nii.gz  -c t2\" -cpu-nb 100 > propseg.out 2> propseg.err