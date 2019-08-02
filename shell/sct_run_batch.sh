#!/bin/bash
#
# Wrapper to processing scripts, which loops across subjects. Data should be
# organized according to the BIDS structure:
# https://github.com/sct-pipeline/spine_generic#file-structure
#
# Usage:
#   ./run_process.sh <parameters> <script>
#
# Example:
#   ./run_process.sh parameters.sh process_data.sh
#
# Author: Julien Cohen-Adad


# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Build color coding (cosmetic stuff)
Color_Off='\033[0m'  # Text Reset
Green='\033[0;92m'  # Yellow
Red='\033[0;91m'  # Red
On_Black='\033[40m'  # Black

# Functions
# =============================================================================
create_folder() {
  local folder="$1"
  mkdir -p $folder  # "-p" creates parent folders if needed
  if [ ! -d "$folder" ]; then
    printf "\n${Red}${On_Black}ERROR: Cannot create folder: $folder. Exit.${Color_Off}\n\n"
    exit 1
  fi
}


# Script starts here
# =============================================================================

# Check number of input params
if [ "$#" -ne 2 ]; then
    echo "Wrong number of input parameters. Correct usage is: ./run_process.sh <parameters> <script>"
    exit 1
fi

# Initialization
PATH_SCRIPT="$( cd "$(dirname "$0")" ; pwd -P )"
time_start=$(date +%x_%r)

# Load config file
source $1
fileparam="`pwd`/$1"

# build syntax for process execution
task="`pwd`/$2"

# Create folders
create_folder $PATH_LOG
create_folder $PATH_RESULTS

# Build list of folders to process
# if variable ONLY_PROCESS_THESE_SUBJECTS does not exist, fetch all folders in directory
echo "Processing:"
if [ -z ${ONLY_PROCESS_THESE_SUBJECTS} ]; then
  # Look into PATH_DATA and fetch all folders
  list_path_subject=`find ${PATH_DATA} -mindepth 1 -maxdepth 1 -type d -name "$SUBJECT_PREFIX*"`
  echo "${list_path_subject[@]}"
else
  # Prepend PATH_DATA to each subject
  echo "${ONLY_PROCESS_THESE_SUBJECTS[*]}"
  list_path_subject=( "${ONLY_PROCESS_THESE_SUBJECTS[@]/#/${PATH_DATA}/}" )
fi

# Run processing with or without "GNU parallel", depending if it is installed or not
if [ -x "$(command -v parallel)" ]; then
  echo 'GNU parallel is installed! Processing subjects in parallel using multiple cores.' >&2
  for path_subject in ${list_path_subject[@]}; do
    subject=`basename $path_subject`
    echo "${PATH_SCRIPT}/_run_with_log.sh $task $subject $fileparam"
  done \
  | parallel -j ${JOBS} --halt-on-error soon,fail=1 bash -c "{}"
else
  echo 'GNU parallel is not installed. Processing subjects sequentially.' >&2
  for path_subject in ${list_path_subject[@]}; do
    subject=`basename $path_subject`
    ${PATH_SCRIPT}/_run_with_log.sh $task $subject $fileparam
  done
fi

# Display stuff
echo "FINISHED :-)"
echo "Started: $time_start"
echo "Ended  : $(date +%x_%r)"
