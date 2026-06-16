#!/usr/bin/env bash
set -euo pipefail

# Why this script exists:
# - In this devcontainer setup, Apptainer's bundled mksquashfs (4.7.5) can crash
#   intermittently during `apptainer build` with exit status 139.
# - In mksquashfs 4.7.5, MALLOC_ARENA_MAX controlling the number of memory arenas is too small to build large images on large CPUs.
# - The issue will be fixed by version 1.5.2 of Apptainer
#
# Issue opened on Apptainer's GitHub: https://github.com/apptainer/apptainer/issues/3577

BUNDLED_BIN="/usr/libexec/apptainer/bin/mksquashfs"
BACKUP_BIN="/usr/libexec/apptainer/bin/mksquashfs.bundled-4.7.5"
FIX_BIN="/usr/libexec/apptainer/bin/mksquashfs-fix.sh"

sudo tee "${FIX_BIN}" > /dev/null << EOF
#!/usr/bin/env bash
set -euo pipefail

export MALLOC_ARENA_MAX=1000000

${BACKUP_BIN} "\$@"

unset MALLOC_ARENA_MAX
EOF

sudo chmod ugo+x "${FIX_BIN}"

if [[ ! -f "${BACKUP_BIN}" ]]; then
  echo "[apptainer-fix] Backing up bundled mksquashfs to ${BACKUP_BIN}"
  sudo cp "${BUNDLED_BIN}" "${BACKUP_BIN}"
else
  echo "[apptainer-fix] Backup already present at ${BACKUP_BIN}"
fi

echo "[apptainer-fix] Linking ${BUNDLED_BIN} -> ${FIX_BIN}"
sudo ln -sf "${FIX_BIN}" "${BUNDLED_BIN}"

echo "[apptainer-fix] Active mksquashfs version:"
#"${BUNDLED_BIN}" -version | head -n 1
