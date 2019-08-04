#!/bin/bash
# Wrapper to be able to output stdout and stderr in text file

TASK=$1
SUBJECT=$2
FILEPARAM=$3

source $FILEPARAM

cd ${PATH_DATA}
TASK_BASENAME=`basename ${TASK} .sh`
echo ${TASK_WITHOUT_EXT}
${TASK} $SUBJECT $FILEPARAM 2>&1 | tee ${PATH_LOG}/${TASK_BASENAME}_${SUBJECT}.log ; test ${PIPESTATUS[0]} -eq 0
if [ ! $? -eq 0 ]; then
  mv ${PATH_LOG}/${TASK_BASENAME}_${SUBJECT}.log ${PATH_LOG}/err.${TASK_BASENAME}_${SUBJECT}.log
fi
