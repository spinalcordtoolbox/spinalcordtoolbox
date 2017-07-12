#!/usr/bin/env bash
#PBS -l nodes=6:ppn=8  # This will reserve 48 (6*8) processors for your computation
                       #  including 1 for orchestration -> need a minimum of a 47 subjects
#PBS -l walltime=00:15:00   # After 15 minutes the computation will be shut down

#PBS -N MY_NAME # Name given to you job in the queuing system

#PBS -A rrp-355-aa # Neuropoly project number on guillimin


# Source the SCT where it is installed
# make sure the SCT was installed with the --mpi option
# If the sct was already installed witout the option, just rerun it with  ./install_sct --mpi -dmb
source /path/to/my_install/of/spinalcordtoolbox/bin/sct_launcher

# cd to a scratch space where outputs will be saved. This step is compulsory on scinet since
# your home dir is read only on the compute node, make sure the directory exist before launching
cd  $SCRATCH/my_work_dir

# where the data is stored
DATA_PATH=/path_to/the_subjects/data/

# to run propseg on subjects t2s with 47 cpu --> need a minimum of a 47 subjects
# outputs will be saved in $SCRATCH/my_work_dir/propseg.out and propseg.err
sct_pipeline  -f sct_propseg -d $DATA_PATH  -p  \"-i t2s/t2s.nii.gz  -c t2\" -cpu-nb 47 > propseg.out 2> propseg.err