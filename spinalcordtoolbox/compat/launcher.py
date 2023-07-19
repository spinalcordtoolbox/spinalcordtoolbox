"""
Compatibility layer to launch old scripts

Copyright (c) 2019 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import sys
import os
import subprocess
import multiprocessing
import importlib

from spinalcordtoolbox import __file__ as package_init_file
from spinalcordtoolbox.utils.sys import init_sct
from spinalcordtoolbox.utils.fs import generate_json_sidecar


def main(argv=None):
    """
    Compatibility entry point to run scripts
    """
    if argv is None:
        argv = sys.argv[1:]

    if "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS" not in os.environ:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(multiprocessing.cpu_count())

    # Needed to allow `sct_check_dependencies` to import voxelmorph/neurite without
    # failing due to a missing `tensorflow` dependency (since the backend defaults to TF)
    os.environ['VXM_BACKEND'] = 'pytorch'
    os.environ['NEURITE_BACKEND'] = 'pytorch'

    command = os.path.basename(sys.argv[0])
    pkg_dir = os.path.dirname(package_init_file)

    script = os.path.join(pkg_dir, "scripts", "{}.py".format(command))
    assert os.path.exists(script)

    mpi_flags = os.environ.get("SCT_MPI_MODE", None)
    if mpi_flags is not None:
        if mpi_flags == "yes":  # compat
            mpi_flags = "-n 1"
        cmd = ["mpiexec"] + mpi_flags.split() + [sys.executable, script] + argv
        return subprocess.run(cmd, env=os.environ).returncode
    else:
        init_sct()
        module = importlib.import_module(name=f"spinalcordtoolbox.scripts.{command}")
        return_values = module.main(argv)
        if isinstance(return_values, tuple) and len(return_values) == 2:
            generate_json_sidecar(
                script_name=command,
                output_folder=return_values[0],
                output_files=return_values[1],
            )
