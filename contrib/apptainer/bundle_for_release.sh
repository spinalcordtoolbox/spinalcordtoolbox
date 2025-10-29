#!/usr/bin/env bash
# Bundle's the apptainer scripts for release

SCT_VERSION="$1"

# Change the version of SCT the installation task to be the requested version
sed_str='s|sct_version="X.Y"|sct_version='"$SCT_VERSION"'|g'
sed -e "$sed_str" "sct.def" > "sct_$SCT_VERSION.def"

# Change the "install_sct_container.sh" script to reference this new definition
sed_str='s|sct.def|sct_'"$SCT_VERSION"'.def|g'
sed -e "$sed_str" "install_sct_containered.sh" > "install_sct_containered_$SCT_VERSION.sh"

# Mark it as an executable file, if (for some reason) it doesn't inherit that fact
chmod +x "install_sct_containered_$SCT_VERSION.sh"

# Bundle all of the files into a gzip tarball
tar -czf "sct_apptainer_$SCT_VERSION.tar.gz" \
  "install_sct_containered_$SCT_VERSION.sh" \
  "sct_$SCT_VERSION.def" \
  "install_deepseg_task.sh" \
  "sct_model_install.def"

# Clean up
rm "sct_$SCT_VERSION.def"
rm "install_sct_containered_$SCT_VERSION.sh"
