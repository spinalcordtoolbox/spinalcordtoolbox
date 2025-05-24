#!/usr/bin/env bash

# TODO: Add argument to allow user to specify SCT version

if [ "$#" -gt 0 ]; then
  TASK_LIST="$*"
  # The leading APPTAINER_BIND declaration forces Apptainer to not use any existing Conda installs, helping w/ portability
  APPTAINER_BIND=' ' apptainer build --build-arg task_installs="$TASK_LIST" sct.sif sct.def
else
  # The leading APPTAINER_BIND declaration forces Apptainer to not use any existing Conda installs, helping w/ portability
  APPTAINER_BIND=' ' apptainer build sct.sif sct.def
fi
