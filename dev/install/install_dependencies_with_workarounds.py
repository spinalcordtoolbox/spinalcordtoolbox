#!/usr/bin/env python
# coding: utf-8
# Check if the platform requires conda tensorflow.
#
# Author: Christian S. Perone

import platform, subprocess

def tf_workarounds():
	print("Checking if TensorFlow is working, if not attempting to install another version")
	for idx_attempt in range(2):
		cmd = ["python", "-c", "import tensorflow"]
		p = subprocess.Popen(cmd,
		 stdout=subprocess.PIPE,
		 stderr=subprocess.PIPE,
		)
		o, e = p.communicate()
		res = p.returncode
		if res == 0:
			print("TensorFlow seems to be working.")
			break

		if idx_attempt == 0:
			print("Trying workaround")
			cmd = "conda install -c defaults --yes tensorflow==1.3.0".split()
			subprocess.call(cmd)
	else:
		print("Couldn't work around bad TF installation")
		print("Here's some info:")
		print(o.decode())
		print("\x1B[31;1m{}\x1B[0m".format(e.decode()))
		raise RuntimeError("Couldn't apply a tensorflow workaround")

def skimage_workarounds():
	print("Checking if skimage is working, if not attempting work around it")
	for idx_attempt in range(2):
		cmd = ["python", "-c", "from skimage.feature import greycomatrix, greycoprops"]
		p = subprocess.Popen(cmd,
		 stdout=subprocess.PIPE,
		 stderr=subprocess.PIPE,
		)
		o, e = p.communicate()
		res = p.returncode
		if res == 0:
			print("skimage seems to be working.")
			break

		if idx_attempt == 0:
			print("Trying workaround")
			cmd = "conda install -c defaults --yes nomkl".split()
			subprocess.call(cmd)
	else:
		print("Couldn't work around bad skimage installation")
		print("Here's some info:")
		print(o.decode())
		print("\x1B[31;1m{}\x1B[0m".format(e.decode()))
		raise RuntimeError("Couldn't apply an skimage workaround")

if __name__ == '__main__':
	tf_workarounds()
	skimage_workarounds()
