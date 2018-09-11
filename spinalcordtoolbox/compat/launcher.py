#!/usr/bin/env python
# Compatibility layer to launch old scripts

import sys, os, subprocess, multiprocessing

def main():
	"""
	Compatibility entry point to run scripts
	"""

	# Force scripts to not use graphical output
	env = dict()
	env.update(os.environ)

	if "DISPLAY" not in os.environ:
		# No DISPLAY, set suitable default matplotlib backend as pyplot is used
		env["MPLBACKEND"] = "Agg"

	if "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS" not in os.environ:
		env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(multiprocessing.cpu_count())

	command = os.path.basename(sys.argv[0])
	sct_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
	script = os.path.join(sct_dir, "scripts", "{}.py".format(command))
	assert os.path.exists(script)
	cmd = [sys.executable, script] + sys.argv[1:]

	mpi_flags = os.environ.get("SCT_MPI_MODE", None)
	if mpi_flags is not None:
		if mpi_flags == "yes": # compat
			mpi_flags = "-n 1"
		cmd = ["mpiexec"] + mpi_flags.split() + cmd

	os.execvpe(cmd[0], cmd[0:], env)
