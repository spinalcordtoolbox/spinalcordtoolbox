#!/usr/bin/env bash
#SBATCH --ntasks=101  # This will reserve 100 processors for your computation
                   #  + 1 for orchestration -> need a minimum of a 100 subjects
#SBATCH --cpus-per-task=1  # sct with mpi runs one task per worker
#SBATCH --mem-per-cpu=2048M  # max memory for a single task 
#SBATCH --time=03:00:00  # After 3 hours the computation will be shut down
#SBATCH --job-name=sct_pipeline  # Name given to you job in the queuing system
#SBATCH --account=def-jcohen # Neuropoly project number on cedar



# Source the SCT where it is installed
# make sure the SCT was installed with the --mpi option
# If the sct was already installed witout the option, just rerun it with  ./install_sct --mpi -dmb
source /path/to/my_install/of/spinalcordtoolbox/bin/sct_launcher

# cd to a scratch space where outputs will be saved
cd /scratch/$USER/workdir

DATA_PATH=/path_to/the_subjects/data/

# to run propseg on subjects t2s with 100 cpu --> need a minimum of a 100 subjects
sct_pipeline  -f sct_propseg -d $DATA_PATH  -p  \"-i t2s/t2s.nii.gz  -c t2\" -cpu-nb 100 > ${HOME}/propseg.out 2> ${HOME}/propseg.err

