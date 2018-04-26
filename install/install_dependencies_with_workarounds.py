#!/usr/bin/env python
# coding: utf-8
# Check if the platform requires conda tensorflow.
#
# Author: Christian S. Perone

import platform, subprocess

def tf_workarounds():
	for idx_attempt in range(2):
		cmd = ["python", "-c", "import tensorflow"]
		res = subprocess.call(cmd)
		if res == 0:
			break

		cmd = "conda install -c defaults --yes tensorflow==1.3.0".split()
		subprocess.call(cmd)
	else:
		raise RuntimeError("Couldn't apply a tensorflow workaround")

def skimage_workarounds():
	for idx_attempt in range(2):
		cmd = ["python", "-c", "from skimage.feature import greycomatrix, greycoprops"]
		res = subprocess.call(cmd)
		if res == 0:
			break

		cmd = "conda install -c defaults --yes nomkl".split()
		subprocess.call(cmd)
	else:
		raise RuntimeError("Couldn't apply an skimage workaround")

if __name__ == '__main__':
	tf_workarounds()
	skimage_workarounds()
