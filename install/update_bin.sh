#!/usr/bin/env bash
# Convenience script to creates the boilerplate-script/wrapper of
# the bin/ directory
#
# This script needs to be run in the install/ directory
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

echo "$boiler_plate"

sed "s|{DYNAMIC_INSTALL_TEMPLATE}|${SCT_ADD_ENV_VARS}|" sct_launcher > ../bin/sct_launcher
cp sct_env ../bin/.

grep -l "__main__" ../scripts/*.py | while read -r filename ; do

  filename=$(basename ${filename})
  filename=${filename%.*}

  [[ ${FILES_TO_REMOVE} =~ ${filename} ]] && echo "no ${filename}" && continue || echo "yes ${filename}"
   
  if [[ ${PIPELINE_SCRIPT} =~ ${filename} && -n ${SCT_MPI_MODE} ]]; then
    # distribute2mpi script
    echo "${boiler_plate_pipeline}" > ../bin/${filename}
  else
    # Normal script
    echo "${boiler_plate}" > ../bin/${filename}
  fi
  chmod 755 ../bin/${filename}

done



