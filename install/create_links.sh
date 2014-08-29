#!/bin/bash
#
# Script to create links in installer
#
CURRENT_DIR=$PWD

# create soft link to each script in SCT_DIR/script
echo "Create soft link to each script in $SCT_DIR/script"
suffix_py='.py'
cd ${SCT_DIR}/scripts/
for script in *.py
do
  echo ${script}
  scriptname=${script%$suffix_py}
  cmd="ln -s ${script} $SCT_DIR/bin/${scriptname}"
  echo ">> $cmd"
  $cmd
done

cd ${CURRENT_DIR}