#!/usr/bin/env python
# Compatibility layer to launch old scripts

import sys, os, subprocess

def main():
	"""
	Compatibility entry point to run scripts
	"""
	command = os.path.basename(sys.argv[0])
	sct_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
	script = os.path.join(sct_dir, "scripts", "{}.py".format(command))
	assert os.path.exists(script)
	cmd = [sys.executable, script] + sys.argv[1:]
	return subprocess.call(cmd)
