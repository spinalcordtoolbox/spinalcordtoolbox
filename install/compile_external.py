#!/usr/bin/env python
#
# Compile external packages.
#

# TODO: before adding line for PYTHONPATH, check if already exists

import os
import sys
import commands
from msct_parser import Parser
import sct_utils as sct

status_sct, path_sct = commands.getstatusoutput('echo $SCT_DIR')


def compile_dipy(target_os, issudo=''):
    path_dipy = path_sct + '/dev/dipy-wheels'
    numpy_version = '1.9.2'

    # check if dipy-wheels directory is already present. If so, remove it to update it.
    if os.path.isdir(path_dipy):
        sct.run(issudo + 'rm -rf ' + path_dipy)

    os.chdir(path_sct + '/dev/')
    sct.run('git clone --recursive https://github.com/MacPython/dipy-wheels.git')
    sct.run(issudo + 'pip install delocate numpy==' + numpy_version + ' scipy cython')
    sct.run(issudo + 'pip install nibabel')
    os.chdir('dipy-wheels/dipy')
    sct.run('python setup.py bdist_wheel')
    if target_os == 'darwin':  # delocating is only for osx
        sct.run('delocate-listdeps dist/*.whl # lists library dependencies')
        sct.run('delocate-wheel dist/*.whl # copies library dependencies into wheel')
        sct.run('delocate-addplat --rm-orig -x 10_9 -x 10_10 dist/*.whl')
    sct.run(issudo + 'cp ' + path_dipy + '/dipy/dist/*.whl ' + path_sct + '/external/')


def compile_denoise(target_os, issudo=''):
    path_denoise = path_sct + '/dev/denoise/ornlm'

    # go to folder
    os.chdir(path_denoise)
    sct.run('python setup.py bdist_wheel')
    if target_os == 'darwin':  # delocating is only for osx
        sct.run('delocate-listdeps dist/*.whl # lists library dependencies')
        sct.run('delocate-wheel dist/*.whl # copies library dependencies into wheel')
        sct.run('delocate-addplat --rm-orig -x 10_9 -x 10_10 dist/*.whl')
    sct.run(issudo + 'cp ' + path_denoise + '/dist/*.whl ' + path_sct + '/external/')


def get_parser():
    # Initialize parser
    parser = Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description("")
    parser.add_option(name="-f",
                      type_value="multiple_choice",
                      description="Library to compile.",
                      mandatory=False,
                      example=['dipy', 'denoise'])

    parser.add_option(name="-a",
                      type_value=None,
                      description="If provided, compile with sudo.",
                      mandatory=False)

    return parser

if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    import platform
    target_os = platform.system().lower()

    issudo = ''
    if "-a" in arguments:
        issudo = 'sudo '

    libraries = ['dipy', 'denoise']
    if "-f" in arguments:
        libraries = [arguments['-f']]

    if 'dipy' in libraries:
        compile_dipy(target_os, issudo)

    if 'denoise' in libraries:
        compile_denoise(target_os, issudo)

    print "Done! Open a new Terminal window to load environment variables."
