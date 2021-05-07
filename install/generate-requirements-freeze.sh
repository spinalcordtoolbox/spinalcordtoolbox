#!/bin/bash
# generate-requirements-freeze.sh: Automate steps for generating
# requirements-freeze.txt for a release

# Generate initial requirements-freeze.txt file
source "$SCT_DIR"/python/etc/profile.d/conda.sh
conda activate venv_sct
pip freeze |
  # Exclude SCT itself (not needed in requirements.txt)
  grep -v "-e git+https://github.com/spinalcordtoolbox/spinalcordtoolbox.git" |
  # Exclude torch-related lines (these will be re-added from requirements.txt)
  grep -v "torch" > "$SCT_DIR"/requirements-freeze.txt
conda deactivate

# Copy over torch-related lines
echo "# Platform-specific torch requirements (See SCT Issue #2745)" \
  >> "$SCT_DIR"/requirements-freeze.txt
grep "torch" "$SCT_DIR"/requirements.txt >> "$SCT_DIR"/requirements-freeze.txt

