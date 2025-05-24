#!/usr/bin/env bash

# Ensure that at least one task was requested
if [ "$#" == "0" ]; then
  echo "You must provide at least one task name (from \`sct_deepseg -h\`) to install models into the container."
  exit 1
fi

# Just passes the arguments through, saving the results to a temporary file so the original isn't deleted if it fails
APPTAINER_BIND=' ' apptainer build --build-arg task_installs="$*" sct_tmp.sif sct_model_install.def

# Removes the old .sif and replaced it with the new one, given the prior command ran correctly
if [ ! -f sct_tmp.sif ]; then
  echo "Failed to install new SCT DeepSeg tasks, terminating"
  exit 0
fi
echo "Replacing old sct.dif with updated one!"
rm sct.sif
mv sct_tmp.sif sct.sif

echo "Done!"
