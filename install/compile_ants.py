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
import getopt
import sys

#=======================================================================================================================
# usage
#=======================================================================================================================
def usage():
    print 'USAGE: \n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Compile ANTs binaries for the Spinal Cord Toolbox.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+'\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -a <path2ants>	 directory in which ANTs is. Must contains ANTs/ folder.\n' \
        '  -s <OS name>      name of the OS {osx,linux}.\n' \
        '  -f <script>		 name of scripts to compile. Separated by commas.\n' \
        '\n'\
        'EXAMPLE:\n' \
        '  compile_ants.py -s linux\n'
    sys.exit(2)


listOS = ['osx', 'linux']
os_target = ''

path_ants = ''
scripts_target = ''
ants_scripts = ['ANTSLandmarksBSplineTransform',
 				'antsApplyTransforms',
 				'antsRegistration',
 				'antsSliceRegularizedRegistration',
 				'ANTSUseLandmarkImagesToGetAffineTransform',
 				'ComposeMultiTransform']

# Check input param
try:
    opts, args = getopt.getopt(sys.argv[1:],'ha:s:f:')
except getopt.GetoptError as err:
    print str(err)
    usage()
for opt, arg in opts:
    if opt == '-h':
        usage()
    elif opt in ('-a'):
        path_ants = str(arg)
    elif opt in ('-s'):
        os_target = str(arg)
    elif opt in ('-f'):
        scripts_target = str(arg)

if os_target not in listOS:
    print 'ERROR: OS name should be one of the following: '+'[%s]' % ', '.join(map(str,listOS))
    usage()

if not path_ants:
	print 'ERROR: ANTs folder must be specified.'
	usage()
else:
	path_ants = sct.slash_at_the_end(path_ants) + '/'
	if not os.path.isdir(path_ants + 'ANTs/'):
		print 'ERROR: Path to ANTs must be a directory containing the folder ANTs/.'
		print 'Path specified: ' + path_ants
		usage()

# process list of scripts
scripts = ants_scripts
if scripts_target:
	scripts = []
	scripts_temp = scripts_target.split(',')
	for sc in scripts_temp:
		if sc in ants_scripts:
			scripts.append(sc)
		else:
			print 'WARNING: the ANTs binary ' + sc + ' is not part of SCT. It wasn\'t included into SCT.'

if not os.path.isdir(path_ants + 'antsbin/'):
	os.makedirs(path_ants + 'antsbin/')
	os.chdir(path_ants + 'antsbin/')
	status, output = sct.run('cmake ../ANTs', verbose=2)
else:
	os.chdir(path_ants + 'antsbin/')

status, output = sct.run('make -j 8', verbose=2)
status, path_sct = sct.run('echo $SCT_DIR')

for script in scripts:		
	sct.run('cp ' + path_ants + 'antsbin/bin/' + script +' ' + path_sct + '/bin/' + os_target + '/isct_' + script, verbose=2)
