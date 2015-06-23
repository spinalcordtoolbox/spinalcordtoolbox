#!/usr/bin/env python
# coding=utf-8
#########################################################################################
#
# Installer for spinal cord toolbox.
# 
# This script will install the spinal cord toolbox under and configure your environment.
# Must be run as a non-administrator (no sudo).
# Installation location: /usr/local/spinalcordtoolbox/
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
import commands
import getopt
from datetime import date
import platform
import subprocess
import signal


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


### Version is a class that contains three levels of versioning
# Inspired by FSL installer
class Version(object):
    def __init__(self,version_sct):
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
        if self.major == other.major and self.minor == other.minor and self.patch == other.patch and self.hotfix == other.hotfix and self.beta == other.beta:
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


class shell_colours(object):
    default = '\033[0m'
    rfg_kbg = '\033[91m'
    gfg_kbg = '\033[92m'
    yfg_kbg = '\033[93m'
    mfg_kbg = '\033[95m'
    yfg_bbg = '\033[104;93m'
    bfg_kbg = '\033[34m'
    bold = '\033[1m'


class InstallationResult(object):
    SUCCESS = 0
    WARN = 1
    ERROR = 2

    def __init__(self, result, status, message):
        self.result = result
        self.status = status
        self.message = message

    def __nonzero__(self):
        self.status


class MsgUser(object): 
    __debug = False
    __quiet = False
    
    @classmethod
    def debugOn(cls):
        cls.__debug = True

    @classmethod
    def debugOff(cls):
        cls.__debug = False

    @classmethod
    def quietOn(cls):
        cls.__quiet = True

    @classmethod
    def quietOff(cls):
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
        print "".join((shell_colours.mfg_kbg, "[Skipped] ", shell_colours.default, msg))

    @classmethod
    def ok(cls, msg):
        if cls.__quiet:
            return
        print "".join((shell_colours.gfg_kbg, "[OK] ", shell_colours.default, msg))
    
    @classmethod
    def failed(cls, msg):
        print "".join((shell_colours.rfg_kbg, "[FAILED] ", shell_colours.default, msg))
    
    @classmethod
    def warning(cls, msg):
        if cls.__quiet:
            return
        print "".join((shell_colours.bfg_kbg, shell_colours.bold, "[Warning]", shell_colours.default, " ", msg))


class Progress_bar(object):
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
                stdout.write(" " * ( len(str(self.max))*2 + 8))
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
        if os.name != 'posix': raise UnsupportedOs('We only support OS X/Linux')
        import platform
        self.os = platform.system().lower()
        self.arch = platform.machine()
        self.applever = ''

        # check out OS
        if self.os == 'darwin':
            self.os = 'osx'
            self.vendor = 'apple'
            self.version = Version(platform.release())
            (self.applever,_,_) = platform.mac_ver()
            if self.arch == 'Power Macintosh': raise UnsupportedOs('We do not support PowerPC')
            self.glibc = ''
            self.bits = ''
        elif self.os == 'linux':
            if hasattr(platform, 'linux_distribution'):
                # We have a modern python (>2.4)
                (self.vendor, version, _) = platform.linux_distribution(full_distribution_name=0)
            else:
                (self.vendor, version, _) = platform.dist()
            self.vendor = self.vendor.lower()
            self.version = Version("1.0.0") #Version(version) not supported yet
            self.glibc = platform.libc_ver()
            if self.arch == 'x86_64':
                self.bits = '64'
            else:
                self.bits = '32'
                # raise UnsupportedOs("We no longer support 32 bit Linux. If you must use 32 bit Linux then try building from our sources.")
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
        if 'Continuum Analytics, Inc.' not in self.python_version and 'conda' not in self.python_version.lower():
            raise Exception("Unsupported Python")


def open_url(url, start=0, timeout=20):
    import urllib2
    import socket
    socket.setdefaulttimeout(timeout)
    MsgUser.debug("Attempting to download %s." % (url))
    
    try:
        req = urllib2.Request(url)
        if start != 0:
            req.headers['Range'] = 'bytes=%s-' % (start)
        rf = urllib2.urlopen(req)
    except urllib2.HTTPError, e:
        MsgUser.debug("%s %s" % (url, e.msg))
        return InstallationResult(False, InstallationResult.ERROR, "Cannot find file %s on server (%s). Try again later." % (url, e.msg))
    except urllib2.URLError, e:
        errno = e.reason.args[0]
        message = e.reason.args[1]
        if errno == 8:
            # Bad host name
            MsgUser.debug("%s %s" % (url, 'Unable to find FSL download server in the DNS'))
        else:
            # Other error
            MsgUser.debug("%s %s" % (url, message))
        return InstallationResult(False, InstallationResult.ERROR, "Cannot find %s (%s). Try again later." % (url, message))
    except socket.timeout, e:
        MsgUser.debug(e.value)
        return InstallationResult(False, InstallationResult.ERROR, "Failed to contact FSL web site. Try again later.")    
    return InstallationResult(rf, InstallationResult.SUCCESS,'')
    
def download_file(url, localf, timeout=20):
    '''Get a file from the url given storing it in the local file specified'''
    import socket, time
    
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
    pb = Progress_bar( x, y, rf_size, numeric=True)

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
                pb.update( dl_size )
            lf.close()
        except (IOError, socket.timeout), e:
            MsgUser.debug(e.strerror)
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

def runProcess(cmd, verbose=1):
    if verbose:
        print cmd
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
    if process.returncode:
        # from inspect import stack
        print output_final[0:-1]
    else:
        return process.returncode, output_final[0:-1]


class Installer:
    def __init__(self):
        self.path_install = "/usr/local"
        self.issudo = "sudo "

        # check if user is sudoer
        if os.geteuid() == 0:
            print "Sorry, you are root. Please type: ./installer without sudo. Your password will be required later." \
                  "Exit program\n"
            sys.exit(2)

        # Check input parameters
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'hp:')
        except getopt.GetoptError:
            usage()
        for opt, arg in opts:
            if opt == '-h':
                usage()
            elif opt == '-p':
                self.path_install = arg

        print ""
        print "============================="
        print "SPINAL CORD TOOLBOX INSTALLER"
        print "Modified: 2015-04-17"
        print "============================="

        try:
            this_computer = Os()
        except UnsupportedOs, e:
            MsgUser.debug(str(e))
            raise InstallFailed(str(e))

        try:
            this_python = Python()
        except Exception, e:
            print e
            print "WARNING: The distribution of Python that you are using is not supported by the SCToolbox.\n" \
                  "You still can use your own distribution of Python but you will have to install our dependencies by yourself.\n" \
                  "Do you still want to continue?"
            install_new = "no"
            signal.alarm(120)
            while install_new not in ["yes", "no"]:
                install_new = input_timeout("[yes|no]: ")
            signal.alarm(0)
            if install_new == "no":
                sys.exit(2)

        if not os.path.isdir(self.path_install):
            print "ERROR: The path you entered does not exist: ${PATH_INSTALL}. Create it first. Exit program\n"
            sys.exit(2)

        # check if sudo is needed to write in installation folder
        MsgUser.message("Checking if administrator rights are needed for installation...")
        if os.access(self.path_install, os.W_OK):
            MsgUser.message("  No sudo needed for adding elements.")
            self.issudo = ""
        else:
            MsgUser.message("  sudo needed for adding elements.")
            self.issudo = "sudo "

        # check if last character is "/". If so, remove it.
        if self.path_install[-1:] == '/':
            self.path_install = self.path_install[:-1]

        self.SCT_DIR = self.path_install + "/spinalcordtoolbox"

        # Retrieving home folder because in python, paths with ~ do not seem to work.
        self.home = os.path.expanduser('~')

        # check if SCT folder already exists - if so, delete it
        print ""
        print "Check if spinalcordtoolbox is already installed (if so, delete it)..."
        if os.path.isdir(self.SCT_DIR):
            # check if sudo is required for removing SCT
            if os.access(self.path_install+"/spinalcordtoolbox", os.W_OK):
                MsgUser.message("  No sudo needed for removing SCT.")
                self.issudo_remove = ""
            else:
                MsgUser.message("  sudo needed for removing SCT.")
                self.issudo_remove = "sudo "

            cmd = self.issudo_remove+"rm -rf "+self.SCT_DIR
            print ">> " + cmd
            status, output = runProcess(cmd)
            #status, output = commands.getstatusoutput(cmd)
            if status != 0:
                print 'ERROR! \n' + output + '\nExit program.\n'
                sys.exit(2)

        # create SCT folder
        print "Create folder: " + self.SCT_DIR + " ..."
        cmd = self.issudo+"mkdir "+self.SCT_DIR
        print ">> " + cmd
        status, output = runProcess(cmd)
        #status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print output + '\n'

        # Checking if a new version of the toolbox is available. If so, change it.
        # Check the version on GitHub Master branch. If a new release is available, ask the user if he want to install it.
        # fetch version of the toolbox
        print 'Fetch version of the Spinal Cord Toolbox...'
        with open ("spinalcordtoolbox/version.txt", "r") as myfile:
            version_sct_str = myfile.read().replace('\n','')
            version_sct = Version(version_sct_str)
        print "  Version: "+str(version_sct)

        # fetch version of the toolbox online
        MsgUser.message("Checking for connection and SCT version online...")
        url_version = "https://raw.githubusercontent.com/neuropoly/spinalcordtoolbox/master/version.txt"
        file_name = "tmp.version_online.txt"
        version_result = download_file(url_version, file_name)
        if version_result.status == InstallationResult.SUCCESS:
            isAble2Connect = True
            with open (file_name, "r") as myfile:
                try:
                    version_sct_online_str = myfile.read().replace('\n','')
                    version_sct_online = Version(version_sct_online_str)
                except ValueError:
                    MsgUser.warning("The extraction of online SCT version seemed to have failed. Please contact SCT administrator with this error: "+version_sct_online_str)
                    version_sct_online = version_sct

            if version_sct.isLessThan_MajorMinor(version_sct_online):
                print "Warning: A new version of the Spinal Cord Toolbox is available online. Do you want to install it?"
                install_new = ""
                signal.alarm(30)
                while install_new not in ["yes","no"]:
                    install_new = input_timeout("[yes|no]: ")
                signal.alarm(0)
                if install_new == "yes":
                    print "The automatic installation of a new release or version of the toolbox is not supported yet. Please download it on https://sourceforge.net/projects/spinalcordtoolbox/"
        else:
            isAble2Connect = False
            print "WARNING: Failed to connect to SCT github website. Please check your connexion. An internet connection is recommended in order to install all the SCT dependences. %s." % (version_result.message)

        # copy SCT files
        print "\nCopy Spinal Cord Toolbox on your computer..."
        cmd = self.issudo + "cp -r spinalcordtoolbox/* " + self.SCT_DIR
        print ">> " + cmd
        status, output = runProcess(cmd)
        #status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'


        # check if .bashrc was already modified. If so, we delete lines related to SCT
        print "\nEdit .bashrc..."

        # check if .bashrc exist. If not, create it.
        if not os.path.isfile(self.home+"/.bashrc"):
            print "  ~/.bashrc does not exist. Creating it..."
            open(self.home+'/.bashrc', 'w+').close()
        else:
            if "SPINALCORDTOOLBOX" in open(self.home+'/.bashrc').read():
                print "  Deleting previous SCT entries in .bashrc"
                cmd = "awk '!/SCT_DIR|SPINALCORDTOOLBOX|ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS/' ~/.bashrc > .bashrc_temp && > ~/.bashrc && cat .bashrc_temp >> ~/.bashrc && rm .bashrc_temp"
                print ">> " + cmd
                status, output = runProcess(cmd)
                if status != 0:
                    print '\nERROR! \n' + output + '\nExit program.\n'
                # test if .bash_profile exists
                if os.path.isfile(self.home+"/.bash_profile"):
                    # delete previous entries in .bash_profile
                    print "  Deleting previous SCT entries in .bash_profile"
                    cmd = "awk '!/SCT_DIR|SPINALCORDTOOLBOX|ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS/' ~/.bash_profile > .bash_profile_temp && > ~/.bash_profile && cat .bash_profile_temp >> ~/.bash_profile && rm .bash_profile_temp"
                    print ">> " + cmd
                    status, output = runProcess(cmd)
                    #status, output = commands.getstatusoutput(cmd)
                    if status != 0:
                        print '\nERROR! \n' + output + '\nExit program.\n'

        # edit .bashrc. Add bin
        with open(self.home+"/.bashrc", "a") as bashrc:
            bashrc.write("\n# SPINALCORDTOOLBOX (added on " + str(date.today()) + ")")
            bashrc.write("\nSCT_DIR=\"" + self.SCT_DIR + "\"")
            bashrc.write("\nexport PATH=${PATH}:$SCT_DIR/bin")
            # add PYTHONPATH variable to allow import of modules
            bashrc.write("\nexport PYTHONPATH=${PYTHONPATH}:$SCT_DIR/scripts")
            bashrc.write("\nexport SCT_DIR PATH")
            # forbid to run several ITK instances in parallel (see issue #201).
            from multiprocessing import cpu_count
            number_of_cpu = cpu_count()
            bashrc.write("\nexport ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="+str(number_of_cpu))
            bashrc.close()

        # Because python script cannot source bashrc or bash_profile, it is necessary to modify environment in the current instance of bash
        os.environ['SCT_DIR'] = self.SCT_DIR
        os.environ['PATH'] = os.environ['PATH']+":"+self.SCT_DIR+"/bin"
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = os.environ['PYTHONPATH']+":"+self.SCT_DIR+"/scripts"
        else:
            os.environ['PYTHONPATH'] = self.SCT_DIR+"/scripts"

        # check if .bash_profile exists. If so, we check if link to .bashrc is present in it. If not, we add it at the end.
        if os.path.isfile(self.home+"/.bash_profile"):
            if "source ~/.bashrc" in open(self.home+'/.bash_profile').read():
                print "\n.bashrc seems to be called in .bash_profile"
            # TODO: check for the case if the user did comment source ~/.bashrc in his .bash_profile
            else:
                print "edit .bash_profile..."
                with open(self.home+"/.bash_profile", "a") as bashprofile:
                    bashprofile.write("\nif [ -f ~/.bashrc ]; then")
                    bashprofile.write("\n  source ~/.bashrc")
                    bashprofile.write("\nfi")
                    bashprofile.close()

        # launch .bashrc. This line doesn't always work. Best way is to open a new terminal.
        cmd = ". ~/.bashrc"
        print ">> " + cmd
        status, output = runProcess(cmd) # runProcess does not seems to work on Travis when sourcing .bashrc
        #status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'

        # install required software
        print "\nInstalling dependences... Depending on your internet connection, this step may take several minutes."
        os.chdir("requirements")
        cmd = self.issudo + "bash requirements.sh"
        print ">> " + cmd
        status, output = runProcess(cmd)
        #status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'
        else:
            print output
        os.chdir("..")

        # Create links to python scripts
        print "\nCreate links to python scripts..."
        cmd = self.SCT_DIR+"/install/create_links.sh"
        if self.issudo is "":
            cmd = cmd+" -a"
        print ">> " + cmd
        status, output = runProcess(cmd)
        #status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'

        # Checking if patches are available for the latest release. If so, install them. Patches installation is available from release 1.1 (need to be changed to 1.2)
        print "\nChecking for available patches..."
        if version_sct.isGreaterOrEqualThan_MajorMinor(Version("1.2")) and version_sct.isEqualTo_MajorMinor(version_sct_online) and isAble2Connect and version_sct != version_sct_online:
            # check if a new patch is available
            if version_sct_online > version_sct:
                print "\nInstalling patch_"+str(version_sct_online)+"..."

                url_patch = "https://raw.githubusercontent.com/neuropoly/spinalcordtoolbox/master/patches/patch_"+str(version_sct_online)+".zip"
                file_name_patch = "patch_"+str(version_sct_online)+".zip"
                name_folder_patch = str(version_sct_online)
                patch_download_result = download_file(url_patch, file_name_patch)

                if patch_download_result.status == InstallationResult.SUCCESS:
                    # unzip patch
                    cmd = "unzip -d temp_patch "+file_name_patch
                    print ">> " + cmd
                    status, output = commands.getstatusoutput(cmd)
                    if status != 0:
                        print '\nERROR! \n' + output + '\nExit program.\n'

                    os.chdir("temp_patch/"+name_folder_patch)
                    # launch patch installation
                    cmd = "python install_patch.py"
                    if self.issudo == "":
                        cmd = cmd + " -a"
                    print ">> " + cmd
                    status, output = runProcess(cmd)
                    #status, output = commands.getstatusoutput(cmd)
                    if status != 0:
                        print '\nERROR! \n' + output + '\nExit program.\n'
                    else:
                        print output
                    os.chdir("../..")

                    MsgUser.message("Removing patch-related files...")
                    cmd = "rm -rf "+file_name_patch+" temp_patch"
                    print ">> " + cmd
                    status, output = runProcess(cmd)
                    #status, output = commands.getstatusoutput(cmd)
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


        # check if other dependent software are installed
        print "\nCheck if other dependent software are installed..."
        cmd = "sct_check_dependences"
        print ">> " + cmd
        status, output = runProcess(cmd)
        #status, output = commands.getstatusoutput(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'
        else:
            print output

        # deleting temporary files
        cmd = "rm -rf tmp.*"
        print ">> " + cmd
        status, output = runProcess(cmd)
        #status, output = commands.getstatusoutput(cmd)
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

USAGE
""" + os.path.basename(__file__) + """ -p <path>

MANDATORY ARGUMENTS
-p <path>                   installation path. Do not add "/" at the end!
-h                          display this help
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
