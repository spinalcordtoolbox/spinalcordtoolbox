#!/usr/bin/env bash

# Ensure that at least one task was requested
if [ "$#" == "0" ]; then
  echo "You must provide at least one task name (from \`sct_deepseg -h\`) to install models into the container."
  exit 1
fi

# Just passes the arguments through
# NB: Even though we are forcefully overwriting the original `sct.sif` file, the original file won't be touched
#     if the build step fails (since it gets extracted to a tmpdir). This is inherent to "Bootstrap: localimage".
APPTAINER_BIND=' ' apptainer build --force --build-arg task_installs="$*" sct.sif sct_model_install.def

# Catch errors and alert user that the `sct.sif` file was not updated
status=$?
if [ $status -ne 0 ];  then
  echo "Apptainer build step exited with non-zero exit code, 'sct.sif' file not updated."
  exit 1
fi

echo "Done!"
