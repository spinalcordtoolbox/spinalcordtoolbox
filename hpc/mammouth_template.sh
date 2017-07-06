#!/bin/bash
#PBS -l procs=101  # This will reserve 101 processors for your computation
                   #  including 1 for orchestration -> need a minimum of a 100 subjects
#PBS -l walltime=03:00:00   # After 3 hours the computation will be shut down
#PBS -N MY_NAME # Name given to you job in the queuing system
#PBS -W rrp-355-aa # Neuropoly project number on mammouth



# Source the SCT where it is installed
# make sure the SCT was installed with the --mpi option
# If the sct was already installed witout the option, just rerun it with  ./install_sct --mpi -dmb
source /path/to/my_install/of/spinalcordtoolbox/bin/sct_launcher

# cd to your workdir
WORK_DIR=/path/to/output/dir
cat $WORK_DIR

# where the data is stored
DATA_PATH=/path_to/the_subjects/data/

# to run propseg on subjects t2s with 100 cpu --> need a minimum of a 100 subjects
# outputs will be saved in $WORK_DIR/propseg.out and propseg.err
sct_pipeline  -f sct_propseg -d $DATA_PATH  -p  \"-i t2s/t2s.nii.gz  -c t2\" -cpu-nb 100 > propseg.out 2> propseg.err
