#!/usr/bin/env bash
set -euo pipefail

# Why this script exists:
# - In this devcontainer setup, Apptainer's bundled mksquashfs (4.7.5) can crash
#   intermittently during `apptainer build` with exit status 139.
# - The distro-provided mksquashfs (squashfs-tools, currently 4.5.1 on Debian 12)
#   is stable for the same image builds.
#
# What this script does:
# 1) Install squashfs-tools.
# 2) Back up Apptainer's bundled helper binary once.
# 3) Point Apptainer's helper path to the system mksquashfs binary.
#
# The operations are idempotent and safe to run multiple times.
#
# Issue opened on Apptainer's GitHub: https://github.com/apptainer/apptainer/issues/3577

BUNDLED_BIN="/usr/libexec/apptainer/bin/mksquashfs"
BACKUP_BIN="/usr/libexec/apptainer/bin/mksquashfs.bundled-4.7.5"
SYSTEM_BIN="/usr/bin/mksquashfs"

echo "[apptainer-fix] Installing squashfs-tools..."
sudo apt-get update
sudo apt-get install -y squashfs-tools

if [[ ! -f "${SYSTEM_BIN}" ]]; then
  echo "[apptainer-fix] ERROR: ${SYSTEM_BIN} not found after install"
  exit 1
fi

if [[ ! -f "${BACKUP_BIN}" ]]; then
  echo "[apptainer-fix] Backing up bundled mksquashfs to ${BACKUP_BIN}"
  sudo cp "${BUNDLED_BIN}" "${BACKUP_BIN}"
else
  echo "[apptainer-fix] Backup already present at ${BACKUP_BIN}"
fi

echo "[apptainer-fix] Linking ${BUNDLED_BIN} -> ${SYSTEM_BIN}"
sudo ln -sf "${SYSTEM_BIN}" "${BUNDLED_BIN}"

echo "[apptainer-fix] Active mksquashfs version:"
"${BUNDLED_BIN}" -version | head -n 1
