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

check_file_exist() {
  local path="$1"
  if [ ! -f "$path" ]; then
    printf "\n${Red}${On_Black}ERROR: File does not exist: $path ${Color_Off}\n\n"
    exit 1
  fi
}

create_folder() {
  local folder="$1"
  mkdir -p $folder  # "-p" creates parent folders if needed
  if [ ! -d "$folder" ]; then
    printf "\n${Red}${On_Black}ERROR: Cannot create folder: $folder ${Color_Off}\n\n"
    exit 1
  fi
}

# Fetch OS type (used to open QC folder)
command_open() {
  if uname -a | grep -i  darwin > /dev/null 2>&1; then
    # OSX
    OPEN_CMD="open"
  elif uname -a | grep -i  linux > /dev/null 2>&1; then
    # Linux
    OPEN_CMD="xdg-open"
  fi
}

# Usage of this script
usage() {
  echo -e "This function runs a batch script across multiple subjects that are
  present in a source folder (as defined in the file parameters.sh). If
  GNU parallel is installed, it will used by distributing subjects across the
  available CPU cores. For more information about the file parameters.sh,
  see the example file: $SCT_DIR/shell/parameters_example.sh."
  echo -e "Correct usage ./sct_run_batch.sh <parameters.sh> <process_data.sh>"
  echo -e "-s  processing subjects sequentially, without parallel processing."
  echo -e "-h  Help"
  exit 99
}



# Script starts here
# =============================================================================

# check number of input parameters
if [ "$#" -ne 2 -a "$#" -ne 3 ]; then
    echo "Wrong number of input parameters."
    usage
 fi

# getopts processes flag input parmaeters
use_parallel=true
while getopts "sh" opt "$3"; do
  case $opt in
    s)  use_parallel=false;;
    h)  echo "help instruction:"; usage;;
    *)  echo "incorrect flag entered:"; usage;;
  esac
done


# Initialization
PATH_SCRIPT="$( cd "$(dirname "$0")" ; pwd -P )"
echo $PATH_SCRIPT
time_start=$(date +%x_%r)


# Check existence of input files
check_file_exist $1
check_file_exist $2

source $1

# Get absolute paths
fileparam="$(cd "$(dirname "$1")"; pwd)/$(basename "$1")"
task="$(cd "$(dirname "$2")"; pwd)/$(basename "$2")"

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


# Run processing with or without "GNU parallel", depending if it is flagged and installed or not
if [ -x "$(command -v parallel)" ] && $use_parallel; then
  echo "GNU parallel is installed! Processing subjects in parallel"
  for path_subject in ${list_path_subject[@]}; do
    subject=`basename $path_subject`
    echo "${PATH_SCRIPT}/_run_with_log.sh $task $subject $fileparam"
  done \
  | parallel -j ${JOBS} --halt-on-error soon,fail=1 bash -c "{}"
else
  echo "GNU parallel is not installed or flag -s has been used!
   Processing subjects sequentially"
  for path_subject in ${list_path_subject[@]}; do
    subject=`basename $path_subject`
    ${PATH_SCRIPT}/_run_with_log.sh $task $subject $fileparam
  done
fi

# Display stuff
echo "FINISHED :-)"
echo "Started: $time_start"
echo "Ended  : $(date +%x_%r)"

# Display syntax to open QC report on web browser
echo; echo "To open Quality Control (QC) report on a web-browser, run the following:"
command_open
echo "$OPEN_CMD $PATH_QC/index.html"
