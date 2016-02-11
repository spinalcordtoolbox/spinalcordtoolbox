#!/usr/bin/env bash

read  -d '' boiler_plate << EOF
#!/bin/bash
sct_launcher \$(basename \$0).py \$@
EOF

echo "$boiler_plate"

grep -l "Parser(" scripts/*.py | while read -r filename ; do
  filename=$(basename ${filename})
  filename=${filename%.*}

  echo "$boiler_plate" > bin/${filename}
  chmod 755 bin/${filename}

done

