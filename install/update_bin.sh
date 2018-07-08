#!/usr/bin/env bash
# Convenience script to creates the boilerplate-script/wrapper of
# the bin/ directory
#
#########################################################################################
# Copyright (c) 2016 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: PO Quirion
# License: see the file LICENSE.TXT
#########################################################################################

# internal stuff or stuff under development that should be accessible in bin/
FILES_TO_REMOVE="msct_nurbs sct_utils sct_dmri_eddy_correct sct_change_image_type sct_invert_image msct_moco msct_parser msct_smooth sct_denoising_onlm"
# Script that uses multiprocessing
PIPELINE_SCRIPT="sct_pipeline"

SCT_SOURCE="$1"
SCT_DIR="$2"

read  -d '' boiler_plate << EOF
#!/bin/bash
\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )/sct_launcher \$(basename \$0).py \$@
EOF

read  -d '' boiler_plate_pipeline << EOF
#!/bin/bash
. \$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )/sct_launcher
export SCT_MPI_MODE=yes
mpiexec -n 1 \${SCT_DIR}/scripts/\$(basename \$0).py \$@
EOF

mkdir -p ${SCT_DIR}/bin
sed "s|{DYNAMIC_INSTALL_TEMPLATE}|${SCT_ADD_ENV_VARS}|" "${SCT_SOURCE}/install/sct_launcher" > "${SCT_DIR}/bin/sct_launcher"
cp "${SCT_SOURCE}/install/sct_env" "${SCT_DIR}/bin/"

grep -l "__main__" "${SCT_SOURCE}/scripts/"*.py | while read -r filename ; do

  filename=$(basename ${filename})
  filename=${filename%.*}

  [[ ${FILES_TO_REMOVE} =~ ${filename} ]] && echo -n " -${filename}" && continue || echo -n " +${filename}"

  if [ -x "${SCT_DIR}/bin/${filename}" ]; then
    rm -f "${SCT_DIR}/bin/${filename}"
  fi

  if [[ ${PIPELINE_SCRIPT} =~ ${filename} && -n ${SCT_MPI_MODE} ]]; then
    # distribute2mpi script
    echo "${boiler_plate_pipeline}" > "${SCT_DIR}/bin/${filename}"
  else
    # Normal script
    echo "${boiler_plate}" > "${SCT_DIR}/bin/${filename}"
  fi
done

chmod 755 "${SCT_DIR}/bin/"*



