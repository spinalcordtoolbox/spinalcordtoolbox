"""
Compatibility layer to launch old scripts

Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import sys
import os
import subprocess
import multiprocessing

from spinalcordtoolbox import __file__ as package_init_file
from spinalcordtoolbox.utils.sys import __sct_dir__


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

    # Override LD_LIBRARY_PATH to prevent a dependency on system libs from PyQt5.
    # This may be brittle, and requires us to install certain packages ourselves
    # into the conda environment.
    # Alternatively, we could specify the need for certainly libraries in our installation docs.
    # See: https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4869#discussion_r2066701055
    env['LD_LIBRARY_PATH'] = os.path.join(__sct_dir__, 'python', 'envs', 'venv_sct', 'lib')

    command = os.path.basename(sys.argv[0])
    pkg_dir = os.path.dirname(package_init_file)

    script = os.path.join(pkg_dir, "scripts", "{}.py".format(command))
    if not os.path.exists(script):
        raise FileNotFoundError(script)

    cmd = [sys.executable, script] + sys.argv[1:]

    mpi_flags = os.environ.get("SCT_MPI_MODE", None)
    if mpi_flags is not None:
        if mpi_flags == "yes":  # compat
            mpi_flags = "-n 1"
        cmd = ["mpiexec"] + mpi_flags.split() + cmd

    return subprocess.run(cmd, env=env).returncode
