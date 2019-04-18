#!/bin/bash
#
# Wrapper to processing scripts, which loops across subjects.
#
# Usage:
#   ./run_process.sh <script>
#
# Example:
#   ./run_process.sh process_data.sh
#
# Note:
#   Make sure to edit the file parameters.sh with the proper list of subjects
#   and variable.
#
# NB: add the flag "-x" after "!/bin/bash" for full verbose of commands.
#
# Julien Cohen-Adad 2019-01-19


# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Build color coding (cosmetic stuff)
Color_Off='\033[0m'  # Text Reset
Green='\033[0;92m'  # Yellow
Red='\033[0;91m'  # Red
On_Black='\033[40m'  # Black

# Fetch OS type (used to open QC folder)
if uname -a | grep -i  darwin > /dev/null 2>&1; then
  # OSX
  export OPEN_CMD="open"
elif uname -a | grep -i  linux > /dev/null 2>&1; then
  # Linux
  export OPEN_CMD="xdg-open"
fi

create_folder() {
  local folder="$1"
  mkdir -p $folder  # "-p" creates parent folders if needed
  if [ ! -d "$folder" ]; then
    printf "\n${Red}${On_Black}ERROR: Cannot create folder: $folder. Exit.${Color_Off}\n\n"
    exit 1
  fi
}

# Initialization
time_start=$(date +%x_%r)

# Load config file
if [ -e "parameters.sh" ]; then
  source parameters.sh
else
  printf "\n${Red}${On_Black}ERROR: The file parameters.sh was not found. You need to create one for this pipeline to work. Please see README.md.${Color_Off}\n\n"
  exit 1
fi

# build syntax for process execution
task=`pwd`/$1

# Create folders
create_folder $PATH_LOG
create_folder $PATH_RESULTS

# Run processing with or without "GNU parallel", depending if it is installed or not
if [ -x "$(command -v parallel)" ]; then
  echo 'GNU parallel is installed! Processing subjects in parallel using multiple cores...' >&2
  find ${PATH_DATA} -mindepth 1 -maxdepth 1 -type d | while read path_subject; do
    subject=`basename $path_subject`
    echo "rsync -avzh ${PATH_DATA}/${subject}/ ${PATH_RESULTS}/${subject}/; cd ${PATH_RESULTS}/${subject}; $task $subject $PATH_RESULTS $PATH_QC $PATH_LOG > ${PATH_LOG}/${subject}.log"
  done \
  | parallel -j ${JOBS} --halt-on-error soon,fail=1 sh -c "{}"
else
  echo 'GNU parallel is not installed. Processing subjects sequentially...' >&2
  find ${PATH_DATA}/ -mindepth 1 -maxdepth 1 -type d | while read path_subject; do
    subject=`basename $path_subject`
    rsync -avzh ${PATH_DATA}/${subject}/ ${PATH_RESULTS}/${subject}/; cd ${PATH_RESULTS}/${subject}; $task $subject $PATH_RESULTS $PATH_QC $PATH_LOG > ${PATH_LOG}/${subject}.log
  done
fi

# Display stuff
echo "DONE!"
echo "Started: $time_start"
echo "Ended  : $(date +%x_%r)"

# Display syntax to open QC report on web browser
echo "To open Quality Control (QC) report on a web-browser, run the following:"
echo "${OPEN_CMD} ${PATH_QC}/index.html"
