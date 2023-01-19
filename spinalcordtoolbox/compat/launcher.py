# Compatibility layer to launch old scripts

import sys
import os
import subprocess
import multiprocessing

from spinalcordtoolbox import __file__ as package_init_file


def main():
    """
    Compatibility entry point to run scripts
    """

    # Force scripts to not use graphical output
    env = dict()
    env.update(os.environ)

    if "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS" not in os.environ:
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(multiprocessing.cpu_count())

    # Needed to allow `sct_check_dependencies` to import voxelmorph/neurite without
    # failing due to a missing `tensorflow` dependency (since the backend defaults to TF)
    env['VXM_BACKEND'] = 'pytorch'
    env['NEURITE_BACKEND'] = 'pytorch'

    command = os.path.basename(sys.argv[0])
    pkg_dir = os.path.dirname(package_init_file)

    script = os.path.join(pkg_dir, "scripts", "{}.py".format(command))
    assert os.path.exists(script)

    cmd = [sys.executable, script] + sys.argv[1:]

    mpi_flags = os.environ.get("SCT_MPI_MODE", None)
    if mpi_flags is not None:
        if mpi_flags == "yes":  # compat
            mpi_flags = "-n 1"
        cmd = ["mpiexec"] + mpi_flags.split() + cmd

    return subprocess.run(cmd, env=env).returncode
