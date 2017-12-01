#!/usr/bin/env python
# 
# This script is used to compile ANTs binaries
# Inputs: ANTs folder, must be a sct fork of original ANTs folder.
# Outputs: binaries that are directly put into sct
# ants_scripts = ['antsApplyTransforms',
# 				  'antsRegistration',
# 				  'antsSliceRegularizedRegistration',
# 				  'ComposeMultiTransform']

import sys

import os
import getopt

# status, path_sct = getstatusoutput('echo $SCT_DIR')
sys.path.append('../scripts')
import sct_utils as sct

url_to_ants_repository = 'https://github.com/stnava/ANTs/archive/master.zip'
ants_downloaded_folder = 'ANTs-master'
ants_downloaded_file = ants_downloaded_folder + '.zip'
listOS = ['osx', 'linux']


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
        print "".join( (bcolors.magenta, "[Skipped] ", bcolors.normal, msg ) )

    @classmethod
    def ok(cls, msg):
        if cls.__quiet:
            return
        print "".join( (bcolors.green, "[OK] ", bcolors.normal, msg ) )

    @classmethod
    def failed(cls, msg):
        print "".join( (bcolors.red, "[FAILED] ", bcolors.normal, msg ) )

    @classmethod
    def warning(cls, msg):
        if cls.__quiet:
            return
        print "".join( (bcolors.yellow, bcolors.bold, "[Warning]", bcolors.normal, " ", msg ) )


def open_url(url, start=0, timeout=20):
    """
    Open URL
    :param url:
    :param start:
    :param timeout:
    :return:
    """
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
    :param url:
    :param localf:
    :param timeout:
    :return:
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


class ProgressBar(object):
    """
    Display nice progress bar
    """
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


def usage():
    print 'USAGE: \n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        '  Compile ANTs binaries for the Spinal Cord Toolbox.\n' \
        '\n' \
        'USAGE\n' \
        '  '+os.path.basename(__file__)+'\n' \
        '\n' \
        'MANDATORY ARGUMENTS\n' \
        '  -s <OS name>\t\tname of the OS {'+', '.join(listOS)+'}.\n' \
        '\n' \
        'OPTIONAL ARGUMENT\n' \
        '  -f <script>\t\tname of scripts to compile. Separated by commas.\n\t\t\tIf not provided, all necessary scripts are compiled and copied in SCT.\n' \
        '  -a <path2ants>\tdirectory in which ANTs is. Must contains ANTs/ folder.\n\t\t\tIf not specified, ANTs folder is downloaded from \n\t\t\t'+url_to_ants_repository+'\n' \
        '\n'\
        'EXAMPLE:\n' \
        '  compile_ants.py -s linux\n'
    sys.exit(2)



os_target = ''
path_ants = ''
scripts_target = ''
ants_scripts = ['antsApplyTransforms',
                'antsRegistration',
                'antsSliceRegularizedRegistration',
                'ComposeMultiTransform']

pwd = os.getcwd()

# Check input param
try:
    opts, args = getopt.getopt(sys.argv[1:], 'ha:s:f:')
except getopt.GetoptError as err:
    print str(err)
    usage()
for opt, arg in opts:
    if opt == '-h':
        usage()
    elif opt in '-a':
        path_ants = str(arg)
    elif opt in '-s':
        os_target = str(arg)
    elif opt in '-f':
        scripts_target = str(arg)

if os_target not in listOS:
    print 'ERROR: OS name should be one of the following: ' + '[%s]' % ', '.join(map(str, listOS)) + '\n'
    usage()

if not path_ants:
    print 'ANTs folder not specified. Cloning from SCT repository...'
    path_ants = pwd
    # from installer import download_file, InstallationResult
    file_download_result = download_file(url_to_ants_repository, ants_downloaded_file)
    if file_download_result.status == InstallationResult.SUCCESS:
        # unzip ants repository
        cmd = 'unzip -u -d ./ ' + ants_downloaded_file
        print ">> " + cmd
        status, output = sct.run(cmd)
        if status != 0:
            print '\nERROR! \n' + output + '\nExit program.\n'
            sys.exit(2)
        sct.run('mv ' + ants_downloaded_folder + ' ANTs/')
    else:
        print 'ERROR: ANTs download failed. Please check your internet connexion or contact administrators.\n'
        usage()

if not os.path.isdir(os.path.join(path_ants, 'ANTs')):
    print 'ERROR: Path to ANTs must be a directory containing the folder ANTs/.'
    print 'Path specified: ' + path_ants + '\n'
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

if not scripts:
    print 'ERROR: No scripts to compile. Please check input.\n'
    usage()

if not os.path.isdir(os.path.join(path_ants, 'antsbin')):
    os.makedirs(os.path.join(path_ants, 'antsbin'))
    os.chdir(os.path.join(path_ants, 'antsbin'))
    sct.run('cmake ../ANTs', verbose=2)
else:
    os.chdir(os.path.join(path_ants, 'antsbin'))

sct.run('make -j 8', verbose=2)
# status, path_sct = sct.run('echo $SCT_DIR')

# copy scripts to ants_binaries
os.chdir(pwd)
sct.run('mkdir ants_binaries')
for script in scripts:
    sct.run('cp ' + os.path.join(path_ants, 'antsbin', 'bin', script) + ' ' + os.path.join("ants_binaries", "isct_' + script"), verbose=2)

# some cleaning
sct.run('rm -rf ANTs/')

print "DONE! binaries are under antsbin/bin"
