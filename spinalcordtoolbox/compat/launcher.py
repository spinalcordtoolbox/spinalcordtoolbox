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

    # Warn the user if they have `LD_LIBRARY_PATH` set
    sct_lib_path = os.path.join(__sct_dir__, 'python', 'envs', 'venv_sct', 'lib')
    ld_library_path = env.get("LD_LIBRARY_PATH", "")
    if ld_library_path and ld_library_path != sct_lib_path:
        print(f"WARNING: The environment variable `LD_LIBRARY_PATH` is set to '{ld_library_path}'. "
              f"This may cause conflicts with library files bundled with SCT. Please consider unsetting "
              f"this variable if you run into errors related to 'Undefined symbols' or `.so` files.")

    # Add SCT's `venv_sct/lib` folder to the `LD_LIBRARY_PATH`, allowing us to supply libraries
    # that may be missing from the user's system, such as `libxcb-xinerama.so.0` on Ubuntu (#3925).
    env['LD_LIBRARY_PATH'] = (sct_lib_path if not ld_library_path
                              else os.pathsep.join([ld_library_path, sct_lib_path]))

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
