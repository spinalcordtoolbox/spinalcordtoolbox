# -*- coding: utf-8 -*-
import subprocess
import datetime
import termios
import signal
import shutil
import time
import glob
import sys
import re
import os
from string import Template
from jinja2 import Environment, FileSystemLoader

__author__initial = "Mathieu Desrosiers"
__contributor__ = "Thierno Barry"
__copyright__ = "Copyright (C) 2014, TOAD et SCT_TOOL"
__credits__ = ["Mathieu Desrosiers"]


def symlink(source, targetDir, targetName=None):
    """link a file into the target directory. the link name is the same as the file

    Args:
        source: name of the source file
        targetDir: destination directory
        targetName: link name

    Returns:
        the relative link name created

    """
    if not os.path.exists(source):  # if source doesnt exist
        return False

    if not os.path.isabs(source):  # Get absolute path source
        source = os.path.abspath(source)

    if not os.path.isabs(targetDir):  # Get absolute path targetDir
        targetDir = os.path.abspath(targetDir)

    if targetName is None:
        targetName = os.path.basename(source)

    target = os.path.join(targetDir, targetName)  # Create full path to target

    if os.path.exists(target):  # Delete target if exist
        os.remove(target)

    targetSplit = target.split(os.path.sep)  # Split with os.path.sep
    sourceSplit = source.split(os.path.sep)  # Split with os.path.sep

    # Get common Path
    commonPath = os.path.sep + os.path.join(*os.path.commonprefix((sourceSplit, targetSplit))) + os.path.sep

    source = source.replace(commonPath, '')  # Get ride of the commonPath
    target = target.replace(os.getcwd() + os.path.sep, '')  # Get ride of the commonPath

    deep = target.count(os.path.sep) + 1  # Number of sub_folders

    substring = '..' + os.path.sep  # Substring ../

    source = substring * deep + source  # Relative link

    os.symlink(source, target)
    return source


def copy(source, destination, name):
    """copy a file into a new destination with a new name

    Args:
        source:  name of the source file
        target:  destination directory
        name:    the output filename


    """
    if source:
        shutil.copy(source, os.path.join(destination, name))


def gunzip(source):
    """Uncompress a file

    Args:
        source:  a filename to uncompress

    Returns:
        the filename resulting from the compression

    """
    cmd = "gunzip {}".format(source)
    launchCommand(cmd)
    return source.replace(".gz", "")


def gzip(source):
    """Compress a file

    Args:
        source:  a filename to compress

    Returns:
        the filename resulting from the compression

    """
    cmd = "gzip {}".format(source)
    launchCommand(cmd)
    return "{}.gz".format(source)


def launchCommand(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=None, nice=0):
    """Execute a program in a new process

    Args:
        command: a string representing a unix command to execute
        stdout: this attribute is a file object that provides output from the child process
        stderr: this attribute is a file object that provides error from the child process
        timeout: Number of seconds before a process is consider inactive, usefull against deadlock
        nice: run cmd  with  an  adjusted  niceness, which affects process scheduling


    Returns
        return a 3 elements tuples representing the command line launch, the standards output and the standard error message

    Raises
        OSError:      the function trying to execute a non-existent file.
        ValueError :  the command line is called with invalid arguments

    """

    start = datetime.datetime.now()
    process = subprocess.Popen(cmd, preexec_fn=lambda: os.nice(nice), stdout=stdout, stderr=stderr, shell=True)

    if timeout is None:
        process.wait()
    else:
        while process.poll() is None:
            time.sleep(0.2)
            now = datetime.datetime.now()
            if (now - start).seconds > timeout:
                os.kill(process.pid, signal.SIGKILL)
                os.waitpid(-1, os.WNOHANG)
                return None, "Error, a timeout for this process occurred"
    output = list(process.communicate())
    output.insert(0, cmd)
    return tuple(output)


def createScript(source, text):
    """Very not useful and way over simplistic method for creating a file

    Args:
        source: The absolute name of the script to create
        text: Text to write into the script

    Returns:
        True if the file have been created

    """
    try:
        with open(source, 'w') as f:
            f.write(text)
    except IOError:
        return False
    return True


def __arrayOf(source, type='String'):
    """Convert a comma separated  string to a list of built-in elements

    Args:
       source:  a comma separated list of  string to convert
       type:        the type of element expected into the output list
                    valid value are: String, Integer, Float and Boolean
    Returns
        a list of expected type elements specified by type

    """
    list = source.replace(';', ',').split(',')
    if type in "Boolean":
        array = []
        for i in list:
            array.append(i.lower().strip() == 'true')
        return array
    elif type in "Float":
        list = map(float, list)
    elif type in "Integer":
        list = map(int, list)
    return list


def arrayOfBoolean(source):
    """Convert a comma separated string to a list of Boolean elements

    Args:
       source: a comma separated list of  string to convert

    Returns
        a list of Boolean elements

    """
    return __arrayOf(source, 'Boolean')


def arrayOfInteger(source):
    """Convert a comma separated string to a list of Integer elements

    Args:
       source: a comma separated list of  string to convert

    Returns
        a list of Integer elements

    """
    return __arrayOf(source, 'Integer')


def arrayOfFloat(source):
    """Convert a comma separated string to a list of Float elements

    Args:
       source: a comma separated list of  string to convert

    Returns
        a list of Float elements

    """
    return __arrayOf(source, 'Float')


def arrayOfString(source):
    """Convert a comma separated string to a list of String elements

    Args:
       source: a comma separated list of  string to convert

    Returns
        a list of String elements

    """
    return __arrayOf(source, 'String')


def getImages(config, dir, prefix, postfix=None, extension="nii.gz", subdir=None):
    """A simple utility function that return an mri image given certain criteria

    Args:
        config:  a ConfigParser object
        dir:     the directory where looking for image(s)
        prefix:  an expression that the filename should start with
        postfix: an expression that the filename should end with (excluding the extension)
        extension:     name of the extension of the filename. defaults: nii.gz
        subdir: a subfolder where looking for image(s)

    Returns:
        A list of filenames if found, False otherwise

    """

    if subdir is not None:
        if os.path.exists(os.path.join(dir, subdir)):
            dir = os.path.join(dir, subdir)

    if extension.find('.') == 0:
        extension = extension.replace(".", "", 1)

    if config.has_option('extension', extension):
        extension = config.get('extension', extension)

    if extension.find('.') == 0:
        extension = extension.replace(".", "", 1)

    if postfix is None:
        images = glob.glob("{}/{}*.{}".format(dir, config.get('prefix', prefix), extension))
    else:
        pfixs = ""
        if isinstance(postfix, str):
            if config.has_option('postfix', postfix):
                pfixs = config.get('postfix', postfix)
            else:
                pfixs = "_{}".format(postfix)
        else:
            for element in postfix:
                if config.has_option('postfix', element):
                    pfixs = pfixs + config.get('postfix', element)
                else:
                    pfixs = pfixs + "_{}".format(element)
        criterias = "{}/{}*{}.{}".format(dir, config.get('prefix', prefix), pfixs, extension)
        images = glob.glob(criterias)

    if len(images) > 0:  # Found at least one image
        return images

    return False


def getImage(config, dir, prefix, postfix=None, extension="nii.gz", subdir=None):
    """A simple utility function that return an mri image given certain criteria

    Args:
        config:  a ConfigParser object
        dir:     the directory where looking for the image
        prefix:  an expression that the filename should start with
        postfix: an expression that the filename should end with (excluding the extension)
        extension:     name of the extension of the filename. defaults: nii.gz
        subdir: a subfolder where looking for image(s)

    Returns:
        the absolute filename if found, False otherwise

    """

    images = getImages(config, dir, prefix, postfix, extension, subdir)
    if images:
        return images.pop()

    return False


def buildName(config, target, source, postfix=None, extension=None, absolute=True):
    """A simple utility function that return a file name that contain the postfix and the current working directory

    The path of the filename contain the current directory
    The extension name will be the same as source unless specify by argument

    Args:
        config: A configParser that contain config.cfg information
        target: The path of the resulting target filename
        source: The input file name, a config prefix or simply a string
        postfix: An item or a list of items specified in config at the postfix section
        extension: the Extension of the new target
        absolute: a boolean if the full path must be absolute

    Returns:
        a file name that contain the postfix and the current working directory
    """

    parts = []
    # determine target name
    if config.has_option('prefix', source):
        targetName = config.get('prefix', source)
    else:
        parts = os.path.basename(source).split(os.extsep)
        targetName = parts.pop(0)
        # tractquerier exception
        if len(parts) > 0:
            if any(parts[0] in s for s in ['left', 'right']):
                targetName += ".{}".format(parts[0])

    # add postfix to target name
    if (postfix is not None) and postfix != "":
        if type(postfix) is list:
            for item in postfix:
                if config.has_option('postfix', item):
                    targetName += config.get('postfix', item)
                else:
                    targetName += "_{}".format(item)
        else:
            if config.has_option('postfix', postfix):
                targetName += config.get('postfix', postfix)
            else:
                targetName += "_{}".format(postfix)

    if extension is None:
        extension = ""
        for part in parts:
            extension += ".{}".format(part)
    else:
        if extension.find('.') == 0:
            extension = extension.replace(".", "", 1)

        if config.has_option('extension', extension):
            extension = config.get('extension', extension)

        if extension.find('.') != 0:
            extension = ".{}".format(extension)

    if extension.strip() != ".":
        targetName += extension

    if absolute:
        targetName = os.path.join(target, targetName)

    return "'{}'".format(targetName)


def getFileWithParents(source, levels=1):
    """

    """
    common = source
    for i in range(levels + 1):
        common = os.path.dirname(common)
    return os.path.relpath(source, common)


def which(source):
    """locate a program file in the user's path

    Args:
        source: the name of the program

    Returns:
         The path for the executable that would be run if that command is actually been invoked.
    """

    def isExecutable(sourcePath):
        return os.path.isfile(sourcePath) and os.access(sourcePath, os.X_OK)

    sourcePath, sourceName = os.path.split(source)
    if sourcePath:
        if isExecutable(source):
            return source
    else:
        for dir in os.environ["PATH"].split(os.pathsep):
            dir = dir.strip('"')
            executable = os.path.join(dir, source)
            if isExecutable(executable):
                return executable

    return None


def parseTemplate(dict, template):
    """provide simpler string substitutions as described in PEP 292

    Args:
       dict: dictionary-like object with keys that match the placeholders in the template
       template: object passed to the constructors template argument.

    Returns:
        the string substitute

    """
    with open(template, 'r') as f:
        return Template(f.read()).safe_substitute(dict)


def displayYesNoMessage(msg, question="Continue? (y or n)", default=None):
    """Utility function that display a convenient message, ask a question, and record the answer

    this function will loop until character y or n is press
    Args:
       msg: A message to display before prompt
       question: A yes no question. the answer must be y or n
       default: Return a value by defaults if user do not enter any input value = ("yes","no")
    Returns:
       A boolean True if the user press y, False otherwise
    """
    print msg
    # make sure the buffer is not clear
    while True:
        choice = raw_input(question)
        if choice.strip() == "" and default is not None:
            if default == 'yes':
                return True
            else:
                return False
        if choice.lower().strip() == 'y':
            return True
        elif choice.lower().strip() == 'n':
            return False


def displayContinueQuitRemoveMessage(msg):
    """Utility function that display a convenient message, ask a question, and record the answer

    this function will loop until character y or n is press

    Args:
       msg: A message to display before prompt

    Returns:
       A String y, n or r
    """
    print msg
    while True:
        choice = raw_input("Continue? (y, n or r)")
        if choice.lower() == 'y':
            return "y"
        elif choice.lower() == 'n':
            return "n"
        elif choice.lower() == 'r':
            return "r"


def slugify(s):
    """stolen code from Dolph Mathews """
    s = s.lower()
    for c in [' ', '-', '.', '/']:
        s = s.replace(c, '_')
    s = re.sub('\W', '', s)
    s = s.replace('_', ' ')
    s = re.sub('\s+', ' ', s)
    s = s.strip()
    s = s.replace(' ', '_')
    return s


def rawInput(message):
    """ Utility that clear the buffer before , reads a line from input

    Args:
        a string that represent a message to write into the standard output

    Returns:
         a line read from input
    """
    sys.stdout.flush()
    termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    return raw_input(message)


def merge_dicts(*dict_args):
    '''
        Given any number of dicts, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts.
        '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


# own functions
# TODO:separate toad utils and sct_report utils

def get_txt_content(txt):
    """
     read txt file and return content as string.
     PS : if the file is empty or doesn't exist , an empty string will be returned
    :param txt: .txt file name
    :return: .txt content as string
    """
    str_content = None
    try:
        file = open(txt, 'r')
        str_content = file.read()
        file.close()
    except IOError:
        print "Missing file. The description will be empty "
    if str_content:
        str_content = str_content.replace("{", "").replace("}","")
    return str_content


def createHtmlFile(templateDir, templateName, fileLink, tags):
    """
    TODO:perform more errors check and send feedback
    :param templateFileLink:
    :param fileLink:
    :param tags:
    :return:
    """
    jinja2Env = Environment(loader=FileSystemLoader(templateDir), trim_blocks=True)
    tpl = jinja2Env.get_template(templateName)
    htmlCode = tpl.render(tags)
    createScript(fileLink, htmlCode)
