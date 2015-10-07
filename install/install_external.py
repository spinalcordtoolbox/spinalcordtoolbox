import os
import commands
from msct_parser import Parser
import sys


def install_external_packages(target_os, path_sct='', issudo=''):
    if not path_sct:
        status_sct, path_sct = commands.getstatusoutput('echo $SCT_DIR')

    package_list = []
    # search through folder to grab libraries that correspond to the OS
    for filename in os.listdir(path_sct + '/external/'):
        if target_os in filename:
            package_list.append(path_sct + '/external/' + filename)

    for package in package_list:
        print "Trying to install: " + package
        cmd = issudo + "pip install " + package
        status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'
        else:
            print output

    return status


def get_parser():
    # Initialize parser
    parser = Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description("")

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
    if target_os == 'darwin':
        target_os = 'osx'

    issudo = ''
    if "-a" in arguments:
        issudo = 'sudo '

    status = install_external_packages(target_os)

    if status != 0:
        sys.exit(2)
