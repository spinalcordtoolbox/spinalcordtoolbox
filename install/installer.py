#!/usr/bin/env python
# coding=utf-8
#########################################################################################
#
# Installer for spinal cord toolbox.
# 
# This script will install the spinal cord toolbox under and configure your environment.
# Must be run as a non-administrator (no sudo).
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad, Benjamin De Leener
# Modified: 2015-01-23
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import os
import sys
import getopt
from datetime import date
import subprocess
import signal
import errno


# small function for input with timeout
def interrupted(signum, frame):
    """called when read times out"""
    print 'interrupted!'
signal.signal(signal.SIGALRM, interrupted)


def input_timeout(text):
    try:
        foo = raw_input(text)
        return foo
    except:
        # timeout
        return


class Version(object):
    """
    Version is a class that contains three levels of versioning
    Inspired by FSL installer
    """
    def __init__(self, version_sct):
        self.version_sct = version_sct

        if not isinstance(version_sct,basestring):
            print version_sct
            raise Exception('Version is not a string.')

        # detect beta, if it exist
        version_sct_beta = self.version_sct.split('_')
        try:
            self.beta = version_sct_beta[1]
            version_sct_main = version_sct_beta[0]
            self.isbeta = True
        except IndexError:
            self.beta = ""
            version_sct_main = version_sct
            self.isbeta = False

        version_sct_split = version_sct_main.split('.')

        for v in version_sct_split:
            if not v.isdigit():
                raise ValueError('Bad version string: '+self.version_sct)
        self.major = int(version_sct_split[0])
        try:
            self.minor = int(version_sct_split[1])
        except IndexError:
            self.minor = 0
        try:
            self.patch = int(version_sct_split[2])
        except IndexError:
            self.patch = 0
        try:
            self.hotfix = int(version_sct_split[3])
        except IndexError:
            self.hotfix = 0

    def __repr__(self):
        return "Version(%s,%s,%s,%s,%r)" % (self.major, self.minor, self.patch, self.hotfix, self.beta)

    def __str__(self):
        result = str(self.major)+"."+str(self.minor)
        if self.patch != 0:
            result = result+"."+str(self.patch)
        if self.hotfix != 0:
            result = result+"."+str(self.hotfix)
        if self.beta != "":
            result = result+"_"+self.beta
        return result

    def __ge__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self > other or self == other:
            return True
        return False

    def __le__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self < other or self == other:
            return True
        return False

    def __cmp__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.__lt__(other):
            return -1
        if self.__gt__(other):
            return 1
        return 0

    def __lt__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.major < other.major:
            return True
        if self.major > other.major:
            return False
        if self.minor < other.minor:
            return True
        if self.minor > other.minor:
            return False
        if self.patch < other.patch:
            return True
        if self.patch > other.patch:
            return False
        if self.hotfix < other.hotfix:
            return True
        if self.hotfix > other.hotfix:
            return False
        if self.isbeta and not other.isbeta:
            return True
        if not self.isbeta and other.isbeta:
            return False
        # major, minor and patch all match so this is not less than
        return False
    
    def __gt__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.major > other.major:
            return True
        if self.major < other.major:
            return False
        if self.minor > other.minor:
            return True
        if self.minor < other.minor:
            return False
        if self.patch > other.patch:
            return True
        if self.patch < other.patch:
            return False
        if self.hotfix > other.hotfix:
            return True
        if self.hotfix < other.hotfix:
            return False
        if not self.isbeta and other.isbeta:
            return True
        if self.isbeta and not other.isbeta:
            return False
        # major, minor and patch all match so this is not less than
        return False 
    
    def __eq__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.major == other.major and self.minor == other.minor and self.patch == other.patch and \
                        self.hotfix == other.hotfix and self.beta == other.beta:
            return True
        return False
    
    def __ne__(self, other):
        if not isinstance(other, Version):
            return NotImplemented
        if self.__eq__(other):
            return False
        return True

    def isLessThan_MajorMinor(self, other):
        if self.major < other.major:
            return True
        if self.major > other.major:
            return False
        if self.minor < other.minor:
            return True
        else:
            return False

    def isGreaterOrEqualThan_MajorMinor(self, other):
        if self.major > other.major:
            return True
        if self.major < other.major:
            return False
        if self.minor >= other.minor:
            return True
        else:
            return False

    def isEqualTo_MajorMinor(self, other):
        return self.major == other.major and self.minor == other.minor

    def isLessPatchThan_MajorMinor(self, other):
        if self.isEqualTo_MajorMinor(other):
            if self.patch < other.patch:
                return True
        return False

    def getFolderName(self):
        result = str(self.major)+"."+str(self.minor)
        if self.patch != 0:
            result = result+"."+str(self.patch)
        if self.hotfix != 0:
            result = result+"."+str(self.hotfix)
        result = result+"_"+self.beta
        return result


class bcolors(object):
    normal = '\033[0m'
    red = '\033[91m'
    green = '\033[92m'
    yellow = '\033[93m'
    blue = '\033[94m'
    magenta = '\033[95m'
    cyan = '\033[96m'
    bold = '\033[1m'
    underline = '\033[4m'


class InstallationResult(object):
    SUCCESS = 0
    WARN = 1
    ERROR = 2

    def __init__(self, result, status, message):
        self.result = result
        self.status = status
        self.message = message


class MsgUser(object): 
    __debug = False
    __quiet = False
    
    @classmethod
    def debug_on(cls):
        cls.__debug = True

    @classmethod
    def debug_off(cls):
        cls.__debug = False

    @classmethod
    def quiet_on(cls):
        cls.__quiet = True

    @classmethod
    def quiet_off(cls):
        cls.__quiet = False

    @classmethod
    def isquiet(cls):
        return cls.__quiet
    
    @classmethod
    def isdebug(cls):
        return cls.__debug
    
    @classmethod    
    def debug(cls, message, newline=True):
        if cls.__debug:
            from sys import stderr
            mess = str(message)
            if newline:
                mess += "\n"
            stderr.write(mess)
    
    @classmethod
    def message(cls, msg):
        if cls.__quiet:
            return
        print msg
    
    @classmethod
    def question(cls, msg):
        print msg,
                  
    @classmethod
    def skipped(cls, msg):
        if cls.__quiet:
            return
        print "".join((bcolors.magenta, "[Skipped] ", bcolors.normal, msg))

    @classmethod
    def ok(cls, msg):
        if cls.__quiet:
            return
        print "".join((bcolors.green, "[OK] ", bcolors.normal, msg))
    
    @classmethod
    def failed(cls, msg):
        print "".join((bcolors.red, "[FAILED] ", bcolors.normal, msg))
    
    @classmethod
    def warning(cls, msg):
        if cls.__quiet:
            return
        print "".join((bcolors.yellow, bcolors.bold, "[Warning]", bcolors.normal, " ", msg))


class ProgressBar(object):
    def __init__(self, x=0, y=0, mx=1, numeric=False):
        self.x = x
        self.y = y
        self.width = 50
        self.current = 0
        self.max = mx
        self.numeric = numeric

    def update(self, reading):
        from sys import stdout
        if MsgUser.isquiet():
            return
        percent = reading * 100 / self.max
        cr = '\r'

        if not self.numeric:
            bar = '#' * int(percent)
        else:
            bar = "/".join((str(reading), str(self.max))) + ' - ' + str(percent) + "%\033[K"
        stdout.write(cr)
        stdout.write(bar)
        stdout.flush()
        self.current = percent

        if percent == 100:
            stdout.write(cr)
            if not self.numeric:
                stdout.write(" " * int(percent))
                stdout.write(cr)
                stdout.flush()
            else:
                stdout.write(" " * (len(str(self.max))*2 + 8))
                stdout.write(cr)
                stdout.flush()


class InstallFailed(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class UnsupportedOs(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Os(object):
    """
    Work out which platform we are running on.
    Also determine which python we are using: version and distribution.
    """
    def __init__(self):
        import os
        if os.name != 'posix':
            raise UnsupportedOs('We only support OS X/Linux')
        import platform
        self.os = platform.system().lower()
        self.arch = platform.machine()
        self.applever = ''

        # check out OS
        if self.os == 'darwin':
            self.os = 'osx'
            self.vendor = 'apple'
            self.version = Version(platform.release())
            (self.applever, _, _) = platform.mac_ver()
            if self.arch == 'Power Macintosh':
                raise UnsupportedOs('We do not support PowerPC')
            self.glibc = ''
            self.bits = ''
        elif self.os == 'linux':
            if hasattr(platform, 'linux_distribution'):
                # We have a modern python (>2.4)
                (self.vendor, version, _) = platform.linux_distribution(full_distribution_name=0)
            else:
                (self.vendor, version, _) = platform.dist()
            self.vendor = self.vendor.lower()
            self.version = Version("1.0.0")  # Version(version) not supported yet
            self.glibc = platform.libc_ver()
            if self.arch == 'x86_64':
                self.bits = '64'
            else:
                self.bits = '32'
                # raise UnsupportedOs("We no longer support 32 bit Linux. "
                #                     "If you must use 32 bit Linux then try building from our sources.")
        else:
            raise UnsupportedOs("We do not support this OS.")


class Python(object):
    """
    This class check if the python that the user used is miniconda and python 2.7
    """
    def __init__(self):
        # check out Python
        import sys
        self.python_version = sys.version
        print self.python_version
        if 'Continuum Analytics, Inc.' not in self.python_version and 'conda' not in self.python_version.lower():
            raise Exception("WARNING: Unsupported Python")
        else:
            print '.. conda (OK)'


def open_url(url, start=0, timeout=20):
    import urllib2
    import socket
    socket.setdefaulttimeout(timeout)
    MsgUser.debug("Attempting to download %s." % url)
    
    try:
        req = urllib2.Request(url)
        if start != 0:
            req.headers['Range'] = 'bytes=%s-' % start
        rf = urllib2.urlopen(req)
    except urllib2.HTTPError, err:
        MsgUser.debug("%s %s" % (url, err.msg))
        return InstallationResult(False, InstallationResult.ERROR,
                                  "Cannot find file %s on server (%s). Try again later." % (url, err.msg))
    except urllib2.URLError, err:
        errno = err.reason.args[0]
        message = err.reason.args[1]
        if errno == 8:
            # Bad host name
            MsgUser.debug("%s %s" % (url, 'Unable to find download server in the DNS'))
        else:
            # Other error
            MsgUser.debug("%s %s" % (url, message))
        return InstallationResult(False, InstallationResult.ERROR,
                                  "Cannot find %s (%s). Try again later." % (url, message))
    except socket.timeout:
        return InstallationResult(False, InstallationResult.ERROR,
                                  "Failed to contact web site. Try again later.")
    return InstallationResult(rf, InstallationResult.SUCCESS, '')


def download_file(url, localf, timeout=20):
    """
    Get a file from the url given storing it in the local file specified
    """
    import socket
    import time
    
    result = open_url(url, 0, timeout)

    if result.status == InstallationResult.SUCCESS:
        rf = result.result
    else:
        return result
    
    metadata = rf.info()
    rf_size = int(metadata.getheaders("Content-Length")[0])
    
    dl_size = 0
    block = 16384
    x = 0
    y = 0
    pb = ProgressBar(x, y, rf_size, numeric=True)

    for attempt in range(1,6):
        # Attempt download 5 times before giving up
        pause = timeout
        try:  
            try:
                lf = open(localf, 'ab')    
            except:
                return InstallationResult(False, InstallationResult.ERROR, "Failed to create temporary file.")

            while True:
                buf = rf.read(block)
                if not buf:
                    break
                dl_size += len(buf)
                lf.write(buf)
                pb.update(dl_size)
            lf.close()
        except (IOError, socket.timeout), err:
            MsgUser.debug(err.strerror)
            MsgUser.message("\nDownload failed re-trying (%s)..." % attempt)
            pause = 0
        if dl_size != rf_size:
            time.sleep(pause)
            MsgUser.message("\nDownload failed re-trying (%s)..." % attempt)
            result = open_url(url, dl_size, timeout)
            if result.status == InstallationResult.ERROR:
                MsgUser.debug(result.message)
            else:
                rf = result.result
        else:
            break      
    if dl_size != rf_size:
        return InstallationResult(False, InstallationResult.ERROR, "Failed to download file.")
    return InstallationResult(True, InstallationResult.SUCCESS, '')


# ======================================================================================================================
# create_folder:  create folder (check if exists before creating it)
#   output: 0 -> folder created
#           1 -> folder already exist
#           2 -> permission denied
# ======================================================================================================================
def create_folder(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
            return 0
        except OSError, e:
            if e.errno != errno.EEXIST:
                return 2
    else:
        return 1


def run(cmd, verbose=1):
    if verbose:
        # print cmd
        print(bcolors.blue+cmd+bcolors.normal)
    process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output_final = ''
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            if verbose == 2:
                print output.strip()
            output_final += output.strip()+'\n'
    # need to remove the last \n character in the output -> return output_final[0:-1]
    # if process.returncode:
    #     # from inspect import stack
    #     print output_final[0:-1]
    # else:
    return process.returncode, output_final[0:-1]


def edit_profile_files(path_home, SCT_DIR):
    # Files are listed in inverse order of reading when shell starts
    file_profile = ['.bashrc', '.profile', '.bash_login', '.bash_profile', ]
    file_profile_default = ''
    # TODO: deal with TSCH and CSH
    # edit_profile_files()

    # loop across profile files
    print "Delete previous SCT entries in existing profile files..."
    for i_file in file_profile:
        # delete previous SCT entries
        if not os.path.isfile(path_home+i_file):
            print '.. ' + i_file + ': Not found.'
        else:
            print '.. ' + i_file + ': Found! Deleting previous SCT entries...'
            # update default_file_profile
            file_profile_default = i_file
            if "SPINALCORDTOOLBOX" in open(path_home+i_file).read():
                cmd = "awk '!/SCT_DIR|SPINALCORDTOOLBOX|ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS/' ~/.bashrc > .bashrc_temp && > ~/.bashrc && cat .bashrc_temp >> ~/.bashrc && rm .bashrc_temp"
                status, output = run(cmd)
                if status != 0:
                    print '\nERROR: \n' + output + '\nExit program.\n'
                    sys.exit()

    print "Add entries to .bashrc..."
    with open(path_home+".bashrc", "a") as bashrc:
        bashrc.write("\n# SPINALCORDTOOLBOX (added on " + str(date.today()) + ")")
        bashrc.write("\nSCT_DIR=\"" + SCT_DIR + "\"")
        bashrc.write("\nexport PATH=${PATH}:$SCT_DIR/bin")
        bashrc.write("\nexport PYTHONPATH=${PYTHONPATH}:$SCT_DIR/scripts")
        bashrc.write("\nexport SCT_DIR PATH")
        from multiprocessing import cpu_count
        number_of_cpu = cpu_count()
        bashrc.write("\nexport ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="+str(number_of_cpu))
        bashrc.close()

    # Because python script cannot source bashrc or bash_profile, it is necessary to modify environment in the
    # current instance of bash
    os.environ['SCT_DIR'] = SCT_DIR
    os.environ['PATH'] = os.environ['PATH']+":"+SCT_DIR+"/bin"
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+":"+SCT_DIR+"/scripts"
    else:
        os.environ['PYTHONPATH'] = SCT_DIR+"/scripts"

    # Check if no profile file other than .bashrc exist
    print "Check if no profile file other than .bashrc exist..."
    if file_profile_default == '' or file_profile_default == '.bashrc':
        print '.. WARNING: No default profile file found: .bash_profile will be created...'
        file_profile_default = '.bash_profile'
    else:
        print '.. OK: Default profile file is:' + file_profile_default

    # Check if .bashrc is sourced in default profile file
    print "Check if .bashrc is sourced in default profile..."
    if "source ~/.bashrc" in open(path_home+file_profile_default).read():
        print ".. .bashrc seems to be sourced already"
    # TODO: check for the case if the user did comment source ~/.bashrc in his .bash_profile
    else:
        print ".. .bashrc is NOT sourced. Appending to "+file_profile_default+" ..."
        with open(path_home+file_profile_default, "a") as bashprofile:
            bashprofile.write("\nif [ -f ~/.bashrc ]; then")
            bashprofile.write("\n  source ~/.bashrc")
            bashprofile.write("\nfi")
            bashprofile.close()

    # launch .bashrc. This line doesn't always work. Best way is to open a new terminal.
    print "Source .bashrc:"
    cmd = ". ~/.bashrc"
    status, output = run(cmd)  # run does not seems to work on Travis when sourcing .bashrc
    # status, output = commands.getstatusoutput(cmd)
    if status != 0:
        print '\nERROR! \n' + output + '\nExit program.\n'
        sys.exit()


class Installer:
    def __init__(self):
        """
        Path by default is /usr/local/sct + version of SCT. Exemple: /usr/local/sct2.2

        The installation is not possible with admin rights because the location of .bashrc and .bash_profile are
        not the same when being admin (sudoer) and non-admin. Therefore, the installation needs to be done without
        using "sudo", but at some point in the installation process, admin permissions may be needed, for exemple when
        installing SCT in /usr/local/ folder.

        If SCT is already installed, we do not want to remove it. Therefore, the installation is stopped and the user
        is asked to fix the issue by, for exemple, removing or renaming the old version of SCT.

        When the user provides the installation folder (using -p option), a folder called "sct" is created and SCT is
        installed in it. If the folder already exists, the installation is stopped and the user is asked to empty the
        folder.
        """
        self.issudo = ""

        # check if user is sudoer
        if os.geteuid() == 0:
            print "Sorry, you are root. Please type: ./installer.py without sudo. Your password will be required " \
                  "later. Exit program\n"
            sys.exit(2)

        # fetch version of the toolbox
        print '\nFetch version of the Spinal Cord Toolbox...'
        with open("spinalcordtoolbox/version.txt", "r") as myfile:
            version_sct_str = myfile.read().replace('\n', '')
            version_sct = Version(version_sct_str)
        print "  Version: " + str(version_sct)

        self.path_install = "/usr/local/sct" + version_sct_str

        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hp:')
        except getopt.GetoptError:
            usage()
            sys.exit(2)

        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt == '-p':
                self.path_install = arg
                if self.path_install[-1:] == '/':
                    self.path_install += 'sct' + version_sct_str
                else:
                    self.path_install += '/sct' + version_sct_str

        print ""
        print "============================="
        print "SPINAL CORD TOOLBOX INSTALLER"
        print "Modified: 2016-01-22"
        print "============================="

        # Check if OS is compatible with SCT
        # The list of compatible OS is available here: TODO: add list of compatible OS
        try:
            Os()
        except UnsupportedOs, err:
            MsgUser.debug(str(err))
            raise InstallFailed(str(err))

        self.SCT_DIR = self.path_install

        # Retrieving home folder because in python, paths with ~ do not seem to work.
        self.home = os.path.expanduser('~') + '/'

        # Check Python
        print ('\nCheck which Python distribution is running...')
        try:
            Python()
        except Exception, err:
            print err
            print "The Python distribution that you are using is not supported by SCT:\n" \
                  "http://sourceforge.net/p/spinalcordtoolbox/wiki/install_python/\n" \
                  "You can still use your own Python distribution, but you will have to install " \
                  "dependencies by yourself.\n" \
                  "Do you still want to continue?"
            install_new = ""
            signal.alarm(120)
            while install_new not in ["yes", "no"]:
                install_new = input_timeout("[yes|no]: ")
            signal.alarm(0)
            if install_new == "no":
                sys.exit(2)

        # Check if pip is install
        print ('\nCheck if pip is installed...')
        status, output = run('pip')
        if not status == 0:
            print ('.. WARNING: pip is not installed. Installing it with conda...')
            # first make sure conda is installed
            status, output = run('conda')
            if not status == 0:
                print ('.. ERROR: conda is not installed either. Please install pip and rerun the installer.\n'+output)
                sys.exit(2)
            else:
                status, output = run('conda install pip -y')
                if not status == 0:
                    print ('.. ERROR: pip installation failed. Please install it and rerun the installer.\n'+output)
                    sys.exit(2)
                else:
                    print ('.. Testing pip...')
                    status, output = run('pip')
                    if not status == 0:
                        print ('.. ERROR: pip cannot be installed. Please install it and rerun the installer.\n'+output)
                        sys.exit(2)
                    else:
                        print ('.. OK!')
        else:
            print('.. OK!')

        # Check if SCT folder already exists. If so, check if the folder is empty. If not, stops installation.
        print ""
        print "\nCheck if SCT is already installed..."
        if os.path.isdir(self.SCT_DIR) and os.listdir(self.SCT_DIR) != []:
            print 'ERROR! SCT is already installed. Two options:\n' \
                  '1) Use another installation path. E.g.: "./installer.py -p ~"\n' \
                  '2) Manually remove the current installation (e.g., use "rm -rf").\n'
            sys.exit(2)

        print ".. Installation path: " + self.path_install

        # If SCT folder does not exists, let's create it
        if not os.path.isdir(self.SCT_DIR):
            print "\nCreate installation folder: " + self.SCT_DIR + " ..."
            result_folder_creation = create_folder(self.path_install)
            if result_folder_creation == 2:
                MsgUser.message(".. sudo needed for adding elements.")
                self.issudo = "sudo "

                cmd = self.issudo + "mkdir " + self.SCT_DIR
                status, output = run(cmd)
                if status != 0:
                    print output + '\n'

        """
        This section has been temporarily removed due to known issues.
        See https://github.com/neuropoly/spinalcordtoolbox/issues/687 for details.

        # Checking if a new version of the toolbox is available. If so, change it.
        # Check the version on GitHub Master branch. If a new release is available,
        # ask the user if he want to install it.

        # fetch version of the toolbox online
        MsgUser.message("\nCheck online if you have the latest version of SCT...")
        url_version = "https://raw.githubusercontent.com/neuropoly/spinalcordtoolbox/master/version.txt"
        file_name = "tmp.version_online.txt"
        version_result = download_file(url_version, file_name)
        if version_result.status == InstallationResult.SUCCESS:
            isAble2Connect = True
            with open(file_name, "r") as myfile:
                try:
                    version_sct_online_str = myfile.read().replace('\n','')
                    version_sct_online = Version(version_sct_online_str)
                except ValueError:
                    MsgUser.warning("The extraction of online SCT version seemed to have failed. "
                                    "Please contact SCT administrator with this error: " + version_sct_online_str)
                    version_sct_online = version_sct
            if version_sct.isLessThan_MajorMinor(version_sct_online):
                print "WARNING: A new version of the Spinal Cord Toolbox is available online. " \
                      "Do you want to install it?"
                install_new = ""
                signal.alarm(30)
                while install_new not in ["yes","no"]:
                    install_new = input_timeout("[yes|no]: ")
                signal.alarm(0)
                if install_new == "yes":
                    print "The automatic installation of a new release or version of the toolbox is not " \
                          "supported yet. Please download it on https://sourceforge.net/projects/spinalcordtoolbox/"
        else:
            isAble2Connect = False
            print "WARNING: Failed to connect to SCT GitHub website. Please check your connexion. " \
                  "An internet connection is recommended in order to install all the SCT dependences. %s." \
                  % version_result.message
        """

        # copy SCT files
        print "\nCopy Spinal Cord Toolbox on your computer..."
        cmd = self.issudo + "cp -r spinalcordtoolbox/* " + self.SCT_DIR
        status, output = run(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'

        # Edit profile files (.bashrc, .bash_profile, ...)
        print "\nEdit profile files..."
        edit_profile_files(self.home, self.SCT_DIR)

        # install required software
        print "\nInstall dependences... Depending on your internet connection, this may take several minutes."
        current_dir = os.getcwd()
        os.chdir(self.SCT_DIR+"/install/requirements")
        cmd = "python requirements.py"
        status, output = run(cmd)
        # status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print '\nERROR: Installation failed while installing requirements.\n'+output
            sys.exit(2)
        else:
            print output
        os.chdir(current_dir)

        # Create links to python scripts
        print "\nCreate links to python scripts..."
        cmd = self.SCT_DIR+"/install/create_links.sh"
        if self.issudo is "":
            cmd += " -a"
        status, output = run(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'

        """
        This section has been temporarily removed due to known issues.
        See https://github.com/neuropoly/spinalcordtoolbox/issues/687 for details.

        # Checking if patches are available for the latest release. If so, install them. Patches installation is
        # available from release 1.1 (need to be changed to 1.2)
        print "\nCheck for latest patches online..."
        if version_sct.isGreaterOrEqualThan_MajorMinor(Version("1.2")) and \
                version_sct.isEqualTo_MajorMinor(version_sct_online) and \
                isAble2Connect and version_sct != version_sct_online:
            # check if a new patch is available
            if version_sct_online > version_sct:
                print "\nInstalling patch_"+str(version_sct_online) + "..."

                url_patch = "https://raw.githubusercontent.com/neuropoly/spinalcordtoolbox/master/patches/patch_" + \
                            str(version_sct_online) + ".zip"
                file_name_patch = "patch_"+str(version_sct_online) + ".zip"
                name_folder_patch = str(version_sct_online)
                patch_download_result = download_file(url_patch, file_name_patch)

                if patch_download_result.status == InstallationResult.SUCCESS:
                    # unzip patch
                    cmd = "unzip -d temp_patch " + file_name_patch
                    print ">> " + cmd
                    status, output = run(cmd)
                    if status != 0:
                        print '\nERROR! \n' + output + '\nExit program.\n'

                    os.chdir("temp_patch/"+name_folder_patch)
                    # launch patch installation
                    cmd = "python install_patch.py"
                    if self.issudo == "":
                        cmd += " -a"
                    status, output = run(cmd)
                    # status, output = commands.getstatusoutput(cmd)
                    if status != 0:
                        print '\nERROR! \n' + output + '\nExit program.\n'
                    else:
                        print output
                    os.chdir("../..")

                    MsgUser.message("Removing patch-related files...")
                    cmd = "rm -rf "+file_name_patch+" temp_patch"
                    status, output = run(cmd)
                    # status, output = commands.getstatusoutput(cmd)
                    if status != 0:
                        print '\nERROR while removing patch-related files \n' + output + '\nExit program.\n'
                    else:
                        print output
                else:
                    MsgUser.warning(patch_download_result.message)
            else:
                print "  No patch available."
        else:
            print "  No connexion or no patch available for this version of the toolbox."
        """

        # compile external packages
        print "\nCompile external packages..."
        cmd = self.SCT_DIR + '/install/install_external.py'
        if self.issudo:
            cmd += ' -a'
        status, output = run(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'
            sys.exit(2)
        else:
            print output

        # Check if other dependent software are installed
        print "\nCheck if other dependent software are installed..."
        cmd = "sct_check_dependences"
        status, output = run(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'
        else:
            print output

        # deleting temporary files
        cmd = "rm -rf tmp.*"
        status, output = run(cmd)
        if status != 0:
            print '\nERROR while removing temporary files \n' + output + '\nExit program.\n'
        else:
            print output

        # display stuff
        print """\n========================================================================================
Installation finished!

If you noticed errors during installation, please start a new Terminal and run:
sct_check_dependences -c -l
Then send the generated file "sct_check_dependences.log" to <jcohen@polymtl.ca>

To get started, open a new Terminal, go back to the downloaded folder and run: ./batch_processing.sh

To test your installation of SCT, run: sct_testing

If you have any problem, please post your issue here:
http://sourceforge.net/p/spinalcordtoolbox/discussion/help/

Enjoy!
"""


def usage():
    print """
""" + os.path.basename(__file__) + """
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION
Install the Spinal Cord Toolbox

USAGE:
""" + os.path.basename(__file__) + """ -p <path>

OPTIONS:
-p <path>         Installation path. Default is: /usr/local/sct
                    Do not specify SCT folder, only path. E.g.: "./installer.py -p ~"
-h                Display this help
  """

    # exit program
    sys.exit(2)

# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    try:
        Installer()
    except InstallFailed, e:
        MsgUser.failed(e.value)
        exit(1)
    except UnsupportedOs, e:
        MsgUser.failed(e.value)

        exit(1)
    except KeyboardInterrupt, e:
        MsgUser.failed("Install aborted by the user.")
        exit(1)
