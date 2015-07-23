#!/usr/bin/env python
#########################################################################################
#
# Module containing several useful functions.
#
# PLEASE!! SORT FUNCTIONS ALPHABETICALLY!
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
# Modified: 2014-07-01
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import errno
import sys
import commands
import subprocess
import re

# TODO: under run(): add a flag "ignore error" for isct_ComposeMultiTransform
# TODO: check if user has bash or t-schell for fsloutput definition

fsloutput = 'export FSLOUTPUTTYPE=NIFTI; ' # for faster processing, all outputs are in NIFTI'


# define class color
class bcolors:
    blue = '\033[94m'
    green = '\033[92m'
    yellow = '\033[93m'
    red = '\033[91m'
    normal = '\033[0m'
    purple = '\033[95m'
    cyan = '\033[96m'
    bold = '\033[1m'
    underline = '\033[4m'



#=======================================================================================================================
# add suffix
#=======================================================================================================================
def add_suffix(file_ext, suffix):
    file_name, ext_name = file_ext.split(os.extsep, 1)  # here we use os.extsep to account for nii.gz extensions
    # add suffix
    return file_name+suffix+'.'+ext_name



#=======================================================================================================================
# run
#=======================================================================================================================
# Run UNIX command
def run_old(cmd, verbose=1):
    if verbose:
        print(bcolors.blue+cmd+bcolors.normal)
    status, output = commands.getstatusoutput(cmd)
    if status != 0:
        printv('\nERROR! \n'+output+'\nExit program.\n', 1, 'error')
    else:
        return status, output

def run(cmd, verbose=1):
    # print sys._getframe().f_back.f_code.co_name
    if verbose:
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
    if process.returncode:
        # from inspect import stack
        printv(output_final[0:-1], 1, 'error')
        # printv('\nERROR in '+stack()[1][1]+'\n', 1, 'error')  # print name of parent function
        # sys.exit()
    else:
        # no need to output process.returncode (because different from 0)
        return process.returncode, output_final[0:-1]



#=======================================================================================================================
# check RAM usage
# work only on Mac OSX
#=======================================================================================================================
def checkRAM(os,verbose=1):
    if (os == 'linux'):
        status, output = run('grep MemTotal /proc/meminfo', 0)
        print output
        ram_split = output.split()
        ram_total = float(ram_split[1])
        status, output = run('free -m', 0)
        print output
        return ram_total/1024

    elif (os == 'osx'):
        status, output = run('hostinfo | grep memory', 0)
        print output
        ram_split = output.split(' ')
        ram_total = float(ram_split[3])

        # Get process info
        ps = subprocess.Popen(['ps', '-caxm', '-orss,comm'], stdout=subprocess.PIPE).communicate()[0]
        vm = subprocess.Popen(['vm_stat'], stdout=subprocess.PIPE).communicate()[0]

        # Iterate processes
        processLines = ps.split('\n')
        sep = re.compile('[\s]+')
        rssTotal = 0 # kB
        for row in range(1, len(processLines)):
            rowText = processLines[row].strip()
            rowElements = sep.split(rowText)
            try:
                rss = float(rowElements[0]) * 1024
            except:
                rss = 0 # ignore...
            rssTotal += rss

        # Process vm_stat
        vmLines = vm.split('\n')
        sep = re.compile(':[\s]+')
        vmStats = {}
        for row in range(1,len(vmLines)-2):
            rowText = vmLines[row].strip()
            rowElements = sep.split(rowText)
            vmStats[(rowElements[0])] = int(rowElements[1].strip('\.')) * 4096
        
        if verbose:
            print 'Wired Memory:\t\t%d MB' % ( vmStats["Pages wired down"]/1024/1024 )
            print 'Active Memory:\t\t%d MB' % ( vmStats["Pages active"]/1024/1024 )
            print 'Inactive Memory:\t%d MB' % ( vmStats["Pages inactive"]/1024/1024 )
            print 'Free Memory:\t\t%d MB' % ( vmStats["Pages free"]/1024/1024 )
            #print 'Real Mem Total (ps):\t%.3f MB' % ( rssTotal/1024/1024 )

        return ram_total


#=======================================================================================================================
# extract_fname
#=======================================================================================================================
# Extract path, file and extension
def extract_fname(fname):

    # extract path
    path_fname = os.path.dirname(fname)+'/'
    # check if only single file was entered (without path)
    if path_fname == '/':
        path_fname = ''
    # extract file and extension
    file_fname = fname
    file_fname = file_fname.replace(path_fname,'')
    file_fname, ext_fname = os.path.splitext(file_fname)
    # check if .nii.gz file
    if ext_fname == '.gz':
        file_fname = file_fname[0:len(file_fname)-4]
        ext_fname = ".nii.gz"

    return path_fname, file_fname, ext_fname

#=======================================================================================================================
# get_absolute_path
#=======================================================================================================================
# Return the absolute path of a file or a directory
def get_absolute_path(fname):
    if os.path.isfile(fname) or os.path.isdir(fname):
        return os.path.realpath(fname)
    else:
        printv('\nERROR: ' + fname + ' does not exist. Exit program.\n', 1, 'error')

#=======================================================================================================================
# check_file_exist:  Check existence of a file or path
#=======================================================================================================================
def check_file_exist(fname, verbose=1):
    if os.path.isfile(fname):
        if verbose:
            printv('  OK: '+fname, verbose, 'normal')
        pass
    else:
        printv('\nERROR: The file ' + fname + ' does not exist. Exit program.\n', 1, 'error')


#=======================================================================================================================
# check_folder_exist:  Check existence of a folder.
#   Does not create it. If you want to create a folder, use create_folder
#=======================================================================================================================
def check_folder_exist(fname, verbose=1):
    if os.path.isdir(fname):
        if verbose:
            printv('  OK: '+fname, verbose, 'normal')
        pass
    else:
        printv('\nERROR: The directory ' + str(fname) + ' does not exist. Exit program.\n', 1, 'error')

#=======================================================================================================================
# return_folder_exist:  Check existence of a folder.
#   Does not create it. If you want to create a folder, use create_folder
#=======================================================================================================================
def return_folder_exist(fname, verbose=1):
    if os.path.isdir(fname):
        if verbose:
            printv('  OK: '+fname, verbose, 'normal')
            return True
        pass
    else:
        printv('\nERROR: The directory ' + str(fname) + ' does not exist.\n', 1, 'warning')
        return False

#=======================================================================================================================
# check_write_permission:  Check existence of a folder.
#   Does not create it. If you want to create a folder, use create_folder
#=======================================================================================================================
def check_write_permission(fname, verbose=1):
    if os.path.isdir(fname):
        if os.path.isdir(fname):
            return os.access(fname, os.W_OK)
        else:
            printv('\nERROR: The directory ' + fname + ' does not exist. Exit program.\n', 1, 'error')
    else:
        path_fname, file_fname, ext_fname = extract_fname(os.path.abspath(fname))
        return os.access(path_fname, os.W_OK)



#=======================================================================================================================
# create_folder:  create folder (check if exists before creating it)
#   output: 0 -> folder created
#           1 -> folder already exist
#           2 -> permission denied
#=======================================================================================================================
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

#=======================================================================================================================
# check_if_3d
#=======================================================================================================================
def check_if_3d(fname):
    nx, ny, nz, nt, px, py, pz, pt = get_dimension(fname)
    if not nt == 1:
        printv('\nERROR: '+fname+' is not a 3D volume. Exit program.\n', 1, 'error')


#=======================================================================================================================
# check_if_rpi:  check if data are in RPI orientation
#=======================================================================================================================
def check_if_rpi(fname):
    from sct_orientation import get_orientation
    if not get_orientation(fname) == 'RPI':
        printv('\nERROR: '+fname+' is not in RPI orientation. Use sct_orientation to reorient your data. Exit program.\n', 1, 'error')


#=======================================================================================================================
# find_file_within_folder
#=======================================================================================================================
def find_file_within_folder(fname, directory):
    """Find file (or part of file, e.g. 'my_file*.txt') within folder tree recursively - fname and directory must be
    strings"""
    import fnmatch

    all_path = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file, fname):
                all_path.append(os.path.join(root, file))
    return all_path


#=======================================================================================================================
# get_dimension
#=======================================================================================================================
# Get dimensions of a nifti file using FSL
def get_dimension(fname):
    # apply fslsize on data
    cmd = 'fslsize '+fname
    status, output = commands.getstatusoutput(cmd)
    # split output according to \n field
    output_split = output.split()

    if output_split[0] == 'ERROR:':
        printv('\n'+output, 1, 'error')
    else:
        # extract dimensions as integer
        try:
            nx = int(output_split[1])
            ny = int(output_split[3])
            nz = int(output_split[5])
            nt = int(output_split[7])
            px = float(output_split[9])
            py = float(output_split[11])
            pz = float(output_split[13])
            pt = float(output_split[15])
            return nx, ny, nz, nt, px, py, pz, pt
        except Exception, e:
            print "Output of the command: \n", output_split
            raise Exception


#=======================================================================================================================
# generate_output_file
#=======================================================================================================================
def generate_output_file(fname_in, fname_out, verbose=1):
    # import stuff
    import shutil  # for moving files
    path_in, file_in, ext_in = extract_fname(fname_in)
    path_out, file_out, ext_out = extract_fname(fname_out)
    # if input image does not exist, give error
    if not os.path.isfile(fname_in):
        printv('  ERROR: File '+fname_in+' does not exist. Exit program.', 1, 'error')
        sys.exit(2)
    # if input and output fnames are the same, do nothing and exit function
    if fname_in == fname_out:
        printv('  WARNING: fname_in and fname_out are the same. Do nothing.', verbose, 'warning')
        print '  File created: '+path_out+file_out+ext_out
        return path_out+file_out+ext_out
    # if fname_out already exists in nii or nii.gz format
    if os.path.isfile(path_out+file_out+ext_out):
        printv('  WARNING: File '+path_out+file_out+ext_out+' already exists. Deleting it...', 1, 'warning')
        os.remove(path_out+file_out+ext_out)
    # Move file to output folder (keep the same extension as input)
    shutil.move(fname_in, path_out+file_out+ext_in)
    # convert to nii (only if necessary)
    if ext_out == '.nii' and ext_in != '.nii':
        os.system('fslchfiletype NIFTI '+path_out+file_out)
    # convert to nii.gz (only if necessary)
    if ext_out == '.nii.gz' and ext_in != '.nii.gz':
        os.system('fslchfiletype NIFTI_GZ '+path_out+file_out)
    # display message
    if verbose:
        print '  File created: '+path_out+file_out+ext_out
    return path_out+file_out+ext_out


#=======================================================================================================================
# check_if_installed
#=======================================================================================================================
# check if dependant software is installed
def check_if_installed(cmd, name_software):
    status, output = commands.getstatusoutput(cmd)
    if status != 0:
        print('\nERROR: '+name_software+' is not installed.\nExit program.\n')
        sys.exit(2)


#=======================================================================================================================
# printv: enables to print or not, depending on verbose status
#   type: handles color: normal (default), warning (orange), error (red)
#=======================================================================================================================
def printv(string, verbose=1, type='normal'):
    # select color based on type of message
    if type == 'normal':
        color = bcolors.normal
    if type == 'info':
        color = bcolors.green
    elif type == 'warning':
        color = bcolors.yellow
    elif type == 'error':
        color = bcolors.red
    elif type == 'code':
        color = bcolors.blue
    elif type == 'bold':
        color = bcolors.bold

    # print message
    if verbose:
        print(color+string+bcolors.normal)

    # if error, exit program
    if type == 'error':
        #raise NameError('Error!')
        sys.exit(2)


#=======================================================================================================================
# sign
#=======================================================================================================================
# Get the sign of a number. Returns 1 if x>=0 and -1 if x<0
def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


#=======================================================================================================================
# slash_at_the_end: make sure there is (or not) a slash at the end of path name
#=======================================================================================================================
def slash_at_the_end(path, slash=0):
    if slash == 0:
        if path[-1:] == '/':
            path = path[:-1]
    if slash == 1:
        if not path[-1:] == '/':
            path = path+'/'
    return path




#=======================================================================================================================
# delete_nifti: delete nifti file(s)
#=======================================================================================================================
def delete_nifti(fname_in):
    # extract input file extension
    path_in, file_in, ext_in = extract_fname(fname_in)
    # delete nifti if exist
    if os.path.isfile(path_in+file_in+'.nii'):
        os.system('rm '+path_in+file_in+'.nii')
    # delete nifti if exist
    if os.path.isfile(path_in+file_in+'.nii.gz'):
        os.system('rm '+path_in+file_in+'.nii.gz')


#=======================================================================================================================
# get_interpolation: get correct interpolation field depending on program used. Supported programs: ants, flirt, WarpImageMultiTransform
#=======================================================================================================================
def get_interpolation(program, interp):
    # TODO: check if field and program exists
    interp_program = ''
    # FLIRT
    if program == 'flirt':
        if interp == 'nn':
            interp_program = ' -interp nearestneighbour'
        elif interp == 'linear':
            interp_program = ' -interp trilinear'
        elif interp == 'spline':
            interp_program = ' -interp spline'
    # ANTs
    elif program == 'ants' or program == 'ants_affine' or program == 'isct_antsApplyTransforms' or program == 'isct_antsSliceRegularizedRegistration':
        if interp == 'nn':
            interp_program = ' -n NearestNeighbor'
        elif interp == 'linear':
            interp_program = ' -n Linear'
        elif interp == 'spline':
            interp_program = ' -n BSpline[3]'
    # check if not assigned
    if interp_program == '':
        printv('WARNING ('+os.path.basename(__file__)+'): interp_program not assigned. Using linear for ants_affine.', 1, 'warning')
        interp_program = ' -n Linear'
    # return
    return interp_program

class UnsupportedOs(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Os(object):
    '''Work out which platform we are running on'''

    def __init__(self):
        import os
        if os.name != 'posix': raise UnsupportedOs('We only support OS X/Linux')
        import platform
        self.os = platform.system().lower()
        self.arch = platform.machine()
        self.applever = ''
        
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
            self.version = Version(version)
            self.glibc = platform.libc_ver()
            if self.arch == 'x86_64':
                self.bits = '64'
            else:
                self.bits = '32'
                # raise UnsupportedOs("We no longer support 32 bit Linux. If you must use 32 bit Linux then try building from our sources.")
        else:
            raise UnsupportedOs("We do not support this OS.")

class Version(object):
    def __init__(self,version_sct):
        self.version_sct = version_sct

        if not isinstance(version_sct,basestring):
            print version_sct
            raise Exception('Version is not a string.')

        # detect beta, if it exist
        version_sct_beta = version_sct.split('_')
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
                raise ValueError('Bad version string.')
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
        print "".join( (shell_colours.mfg_kbg, "[Skipped] ", shell_colours.default, msg ) )

    @classmethod
    def ok(cls, msg):
        if cls.__quiet:
            return
        print "".join( (shell_colours.gfg_kbg, "[OK] ", shell_colours.default, msg ) )
    
    @classmethod
    def failed(cls, msg):
        print "".join( (shell_colours.rfg_kbg, "[FAILED] ", shell_colours.default, msg ) )
    
    @classmethod
    def warning(cls, msg):
        if cls.__quiet:
            return
        print "".join( (shell_colours.bfg_kbg, shell_colours.bold, "[Warning]", shell_colours.default, " ", msg ) )
