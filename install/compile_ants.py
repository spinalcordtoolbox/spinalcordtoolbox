#!/usr/bin/env python
# 
# This script is used to compile ANTs binaries
# Inputs: ANTs folder, must be a sct fork of original ANTs folder.
# Outputs: binaries that are directly put into sct
# ants_scripts = ['ANTSLandmarksBSplineTransform',
# 				'antsApplyTransforms',
# 				'antsRegistration',
# 				'antsSliceRegularizedRegistration',
# 				'ANTSUseLandmarkImagesToGetAffineTransform',
# 				'ComposeMultiTransform']

import sct_utils as sct
import os

path_ants = '/Users/benjamindeleener/code/'
os_target = 'osx'
ants_scripts = ['ANTSLandmarksBSplineTransform']

if not os.path.isdir(path_ants + 'antsbin/'):
	os.makedirs(path_ants + 'antsbin/')
	os.chdir(path_ants + 'antsbin/')
	status, output = sct.run('cmake ../ANTs', verbose=2)
else:
	os.chdir(path_ants + 'antsbin/')

status, output = sct.run('make -j 8', verbose=2)
status, path_sct = sct.run('echo $SCT_DIR')

for script in ants_scripts:		
	sct.run('cp ' + path_ants + 'antsbin/bin/' + script +' ' + path_sct + '/bin/' + os_target + '/isct_' + script, verbose=2)
