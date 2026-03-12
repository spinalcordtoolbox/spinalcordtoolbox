#!/usr/bin/env bash
# Bundle's the apptainer scripts for release

SCT_VERSION="$1"

# Make sure that whatever version is requested is safe to use in filenames
SCT_VERSION_SAFE=$(printf '%s' "$SCT_VERSION" | sed 's/[^A-Za-z0-9._-]/_/g')

# Change the version of SCT the installation task to be the requested version
sed_str='s|sct_version="X.Y"|sct_version='"$SCT_VERSION"'|g'
sed -e "$sed_str" "sct.def" > "sct_$SCT_VERSION_SAFE.def"

# Change the "install_sct_container.sh" script to reference this new definition
sed_str='s|sct.def|sct_'"$SCT_VERSION_SAFE"'.def|g'
sed -e "$sed_str" "install_sct_containered.sh" > "install_sct_containered_$SCT_VERSION_SAFE.sh"

# Mark it as an executable file, if (for some reason) it doesn't inherit that fact
chmod +x "install_sct_containered_$SCT_VERSION_SAFE.sh"

# Bundle all of the files into a gzip tarball
tar -czf "sct_apptainer_$SCT_VERSION_SAFE.tar.gz" \
  "install_sct_containered_$SCT_VERSION_SAFE.sh" \
  "sct_$SCT_VERSION_SAFE.def" \
  "install_deepseg_task.sh" \
  "sct_model_install.def"

# Clean up
rm "sct_$SCT_VERSION_SAFE.def"
rm "install_sct_containered_$SCT_VERSION_SAFE.sh"
