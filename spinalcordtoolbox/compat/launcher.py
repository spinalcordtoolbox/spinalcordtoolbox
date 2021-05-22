#!/usr/bin/env python
# Compatibility layer to launch old scripts

import sys
import os

import spinalcordtoolbox as sct


def main():
    """
    Compatibility entry point to run scripts
    """
    # Convert command syntax: "sct_<function> -args" --> "python $ABSOLUTEPATH/sct_<function>.py -args"
    command = os.path.basename(sys.argv[0])
    pkg_dir = os.path.dirname(sct.__file__)
    script = os.path.join(pkg_dir, "scripts", f"{command}.py")
    cmd = [sys.executable, script] + sys.argv[1:]

    # Adapt command to MPI, which is used for parallelization on HPC architectures (https://hpc-wiki.info/hpc/MPI)
    mpi_flags = os.environ.get("SCT_MPI_MODE", None)
    if mpi_flags is not None:
        if mpi_flags == "yes":  # compat
            mpi_flags = "-n 1"
        cmd = ["mpiexec"] + mpi_flags.split() + cmd

    os.execvp(cmd[0], cmd[0:])
