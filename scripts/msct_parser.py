#!/usr/bin/env python
#########################################################################################
#
# Parser
# Add option with name, type, short description, mandatory or not, example using add_option method.
# usage: add_option(name, type_value=None, description=None, mandatory=False, example=None, help=None, default_value=None)
# If the user make a misspelling, the parser will search in the option list what are nearest option and suggests it to the user
#
# Type of options are:
# - "file", "folder" (check existence)
# - "folder_creation" (check existence and if does not exist, create it if writing permission)
# - "file_output" (check writing permission)
# - "str", "int", "float", "long", "complex" (check if input is the correct type)
# - "multiple_choice"
# - "Coordinate" [x, y, z, value]
# - lists of types: example: [[','],'int'] or [[':'],'coordinate']
# - None, return True when detected (example of boolean)
#
# The parser returns a dictionary with all mandatory arguments as well as optional arguments with default values.
#
# Usage:
# from msct_parser import *
# parser = Parser(__file__)
# parser.usage.set_description('Here is your script description')
# parser.add_option("-input","file", "image"*, True*, "t2.nii.gz"*)
# * optional arguments : description, mandatory (boolean), example
# parser.add_option("-test","int")
# parser.add_option("-dim", ['x', 'y', 'z', 't'], 'dimension: x|y|z|t')
# parser.add_option("-test2") # this is a option without argument
#
# Here we define a multiple choice option named "-a"
# To define the list of available choices, we define it in the example section
# parser.add_option(name="-a",
#                   type_value="multiple_choice",
#                   description="Algorithm for curve fitting.",
#                   mandatory=False,
#                   example=["hanning", "nurbs"],
#                   default_value="hanning")
#
#
# Here we define a deprecated option.
# Deprecated option can be defined in 3 ways :
#       - deprecated : the option exists but is no longer supported
#       - deprecated_rm : This option signals the user that the option has been removed in the current version.
#       - deprecated_by : This option serves to indicate that a new option is used to implement the functionality.
# parser.add_option(name="-bzmax",
#                   type_value=None,
#                   description="maximize the cropping of the image (provide -dim if you want to specify the dimensions)",
#                   deprecated_by="-bmax",
#                   mandatory=False)

#
# Usage are available as follow:
# string_usage = parser.usage.generate()
#
# Arguments are available directly:
# arguments = parser.parse(sys.argv[1:])
# for mandatory arguments :
# if "-input" in arguments:
#     fname_input = arguments["-input"]
# if "-dim" in arguments:
#     dim = arguments["-dim"]
# else:
#     sct.printv(string_usage)
# exit(1)
# for non mandatory arguments :
# if "-output" in arguments:
#     fname_output = arguments["-input"]
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Augustin Roux
# Created: 2014-10-27
# Last modified: 2014-11-07
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sct_utils as sct
from msct_types import Coordinate  # DO NOT REMOVE THIS LINE!!!!!!! IT IS MANDATORY!

########################################################################################################################
# OPTION
########################################################################################################################


class Option:
    # list of option type that can be casted
    OPTION_TYPES = ["str", "int", "float", "long", "complex", "Coordinate"]
    # list of options that are path type
    # input file/folder
    OPTION_PATH_INPUT = ["file", "folder", "image_nifti"]
    # output file/folder
    OPTION_PATH_OUTPUT = ["file_output", "folder_output"]

    # Constructor
    def __init__(self, name, type_value, description, mandatory, example, default_value, help, parser, order=0,
                 deprecated_by=None, deprecated_rm=False, deprecated=False, list_no_image=None):
        self.name = name
        self.type_value = type_value
        self.description = description
        self.mandatory = mandatory
        self.example = example
        self.default_value = default_value
        self.help = help
        self.parser = parser
        self.order = order
        self.deprecated_by = deprecated_by
        self.deprecated_rm = deprecated_rm
        self.deprecated = deprecated
        self.list_no_image = list_no_image

        # TODO: check if the option is correctly set

    def __safe_cast__(self, val, to_type):
        return to_type(val)

    # Do we need to stop the execution if the input is not correct?
    def check_integrity(self, param, type=None):
        """
        check integrity of each option type
        if type is provided, use type instead of self.type_value --> allow recursive integrity checking
        """

        type_option = self.type_value
        if type is not None:
            type_option = type

        if type_option in self.OPTION_TYPES:
            return self.checkStandardType(param, type)

        elif type_option == "image_nifti":
            return self.checkIfNifti(param)

        elif type_option == "file":
            return self.checkFile(param)

        elif type_option == "file_output":  # check if permission are required
            if not sct.check_write_permission(param):
                self.parser.usage.error("Error of writing permissions on file: " + param)
            return param

        elif type_option == "folder":
            return self.checkFolder(param)

        elif type_option == "folder_creation":
            return self.checkFolderCreation(param)

        elif type_option == "multiple_choice":
            """
            the choices are listed in example variable
            """
            if param not in self.example:
                self.parser.usage.error(self.name + " only takes " + self.parser.usage.print_list_with_brackets(self.example) + " as potential arguments.")
            return param

        elif isinstance(type_option, list):
            """
            This option is defined as a list delimited by a delimiter (that cannot be a space)
            For now, only one-layer list are available
            Examples:
            [[','],'int']
            [[':'],'coordinate']
            """
            delimiter = type_option[0][0]
            sub_type = type_option[1]
            param_splitted = param.split(delimiter)
            if len(param_splitted) != 0:
                # check if files are separated by space (if "*" was used)
                if not param_splitted[0].find(' ') == -1:
                    # if so, split and return list
                    param_splitted = param_splitted[0].split(' ')
                return list([self.check_integrity(val, sub_type) for val in param_splitted])
            else:
                self.parser.usage.error("ERROR: Option " + self.name + " must be correctly written. See usage.")

        else:
            # self.parser.usage.error("ERROR: Type of option \"" + str(self.type_value) +"\" is not supported by the parser.")
            sct.printv("WARNING : Option " + str(self.type_value) + " does not exist and will have no effect on the execution of the script", "warining")
            sct.printv("Type -h to see supported options", "warning")

    def checkStandardType(self, param, type=None):
        # check if a int is really a int (same for str, float, long and complex)
        type_option = self.type_value
        if type is not None:
            type_option = type
        try:
            return self.__safe_cast__(param, eval(type_option))
        except ValueError:
            self.parser.usage.error("ERROR: Option " + self.name + " must be " + type_option)

    def checkFile(self, param):
        # check if the file exist
        sct.printv("Check file existence...", 0)
        if self.parser.check_file_exist:
            sct.check_file_exist(param, 0)
        return param

    def checkIfNifti(self, param):
        import os
        sct.printv("Check file existence...", 0)
        nii = False
        niigz = False
        no_image = False
        param_tmp = str()
        if param.lower().endswith('.nii'):
            if self.parser.check_file_exist:
                nii = os.path.isfile(param)
                niigz = os.path.isfile(param + '.gz')
            else:
                nii, niigz = True, False
            param_tmp = param[:-4]
        elif param.lower().endswith('.nii.gz'):
            if self.parser.check_file_exist:
                niigz = os.path.isfile(param)
                nii = os.path.isfile(param[:-3])
            else:
                nii, niigz = False, True
            param_tmp = param[:-7]
        elif param.lower() in self.list_no_image:
            no_image = True
        else:
            sct.printv("ERROR: File is not a NIFTI image file. Exiting", type='error')

        if nii:
            return param_tmp + '.nii'
        elif niigz:
            return param_tmp + '.nii.gz'
        elif no_image:
            return param
        else:
            sct.printv("ERROR: File " + param + " does not exist. Exiting", type='error')

    def checkFolder(self, param):
        # check if the folder exist. If not, create it.
        if self.parser.check_file_exist:
            sct.check_folder_exist(param, 0)
        return param

    def checkFolderCreation(self, param):
        # check if the folder exist. If not, create it.
        if self.parser.check_file_exist:
            result_creation = sct.create_folder(param)
        else:
            result_creation = 0  # no need for checking
        if result_creation == 2:
            sct.printv("ERROR: Permission denied for folder creation...", type="error")
        elif result_creation == 1:
            sct.printv("Folder " + param + " has been created.", 0, type='warning')
        return param


########################################################################################################################
# PARSER
########################################################################################################################

class Parser:
    # Constructor
    def __init__(self, file_name):
        self.file_name = file_name
        self.options = dict()
        self.spelling = SpellingChecker()
        self.errors = ''
        self.usage = Usage(self, file_name)
        self.check_file_exist = True

    def add_option(self, name, type_value=None, description=None, mandatory=False, example=None, help=None, default_value=None, deprecated_by=None, deprecated_rm=False, deprecated=False, list_no_image=None):
        order = len(self.options) + 1
        self.options[name] = Option(name, type_value, description, mandatory, example, default_value, help, self, order, deprecated_by, deprecated_rm, deprecated, list_no_image)

    def parse(self, arguments, check_file_exist=True):
        # if you only want to parse a string and not checking for file existence, change flag check_file_exist
        self.check_file_exist = check_file_exist

        # if no arguments, sct.printv(usage and quit)
        if len(arguments) == 0 and len([opt for opt in self.options if self.options[opt].mandatory]) != 0:
            self.usage.error()

        # check if help is asked by the user
        if "-h" in arguments:
            sct.printv(self.usage.generate())
            exit(1)

        if "-sf" in arguments:
            doc_sourceforge = DocSourceForge(self, self.file_name)
            doc_sourceforge.generate()
            exit(1)

        # initialize results
        dictionary = dict()

        # initialize the spelling checker
        self.spelling.setWordsAsList([name for name in self.options])

        # checking if some file names or folder names contains spaces.
        # We suppose here that the user provides correct structure of arguments (i.e., one "-something", one "argument value", one "-somethingelse", one "another argument value", etc.)
        # We also suppose that multiple spaces can be present
        # we also check if double-quotes are present. If so, we need to concatenate the fields.
        arguments_temp = []
        index_next = 0
        for index in range(0, len(arguments)):
            if index == index_next:
                if arguments[index][0] == '-':
                    arguments_temp.append(arguments[index])
                    index_next = index + 1
                else:
                    temp_str = arguments[index]
                    index_temp = index
                    if index_temp < len(arguments) - 1:
                        if arguments[index][0] == '"':
                            while arguments[index_temp + 1][-1] != '"':  # loop until we find a double quote. Then concatenate.
                                temp_str += ' ' + arguments[index_temp + 1]
                                index_temp += 1
                                if index_temp >= len(arguments) - 1:
                                    break
                            temp_str += ' ' + arguments[index_temp + 1]
                            temp_str = temp_str[1:-1]
                        else:
                            while arguments[index_temp + 1][0] != '-':  # check if a space is present. If so, concatenation of strings.
                                temp_str += ' ' + arguments[index_temp + 1]
                                index_temp += 1
                                if index_temp >= len(arguments) - 1:
                                    break
                    index_next = index_temp + 1
                    if '"' not in temp_str:
                        arguments_temp.append(temp_str)
        arguments = arguments_temp

        skip = False
        for index, arg in enumerate(arguments):
            if skip:  # if argument need to be skipped, we pass
                skip = False
                continue

            if arg in self.options:
                if self.options[arg].deprecated_rm:
                    sct.printv("ERROR : " + arg + " is a deprecated argument and is no longer supported by the current version.", 1, 'error')
                # for each argument, check if is in the option list.
                # if so, check the integrity of the argument
                if self.options[arg].deprecated:
                    sct.printv("WARNING : " + arg + " is a deprecated argument and will no longer be updated in future versions.", 1, 'warning')
                if self.options[arg].deprecated_by is not None:
                    try:
                        sct.printv("WARNING : " + arg + " is a deprecated argument and will no longer be updated in future versions. Changing argument to " + self.options[arg].deprecated_by + ".", 1, 'warning')
                        arg = self.options[arg].deprecated_by
                    except KeyError as e:
                        sct.printv("ERROR : Current argument non existent : " + e.message, 1, 'error')
                if self.options[arg].type_value:
                    if len(arguments) > index + 1:  # Check if option is not the last item
                        param = arguments[index + 1]
                    else:
                        self.usage.error("ERROR: Option " + self.options[arg].name + " needs an argument...")

                    # check if option has an argument that is not another option
                    if param in self.options:
                        self.usage.error("ERROR: Option " + self.options[arg].name + " needs an argument...")

                    # check if this flag has already been used before, then create a list and append this string to the previous string
                    if arg in dictionary:
                        # check if dictionary[arg] is already a list
                        if isinstance(dictionary[arg], list):
                            dictionary[arg].append().check_integrity(param)
                        else:
                            dictionary[arg] = [dictionary[arg], self.options[arg].check_integrity(param)]
                    else:
                        dictionary[arg] = self.options[arg].check_integrity(param)
                    skip = True
                else:
                    dictionary[arg] = True
            else:
                # if not in the list of known options, there is a syntax error in the list of arguments
                # check if the input argument is close to a known option
                spelling_candidates = self.spelling.correct(arg)
                if len(spelling_candidates) != 0:
                    self.usage.error("ERROR: argument " + arg + " does not exist. Did you mean: " + ', '.join(spelling_candidates) + '?')
                else:
                    self.usage.error("ERROR: argument " + arg + " does not exist. See documentation.")

        # check if all mandatory arguments are provided by the user
        for option in [opt for opt in self.options if self.options[opt].mandatory and self.options[opt].deprecated_by is None]:
            if option not in dictionary:
                self.usage.error('ERROR: ' + option + ' is a mandatory argument.\n')

        # check if optional arguments with default values are all in the dictionary. If not, add them.
        for option in [opt for opt in self.options if not self.options[opt].mandatory]:
            if option not in dictionary and self.options[option].default_value:
                dictionary[option] = self.options[option].default_value

        # return a dictionary with each option name as a key and the input as the value
        return dictionary

    def add_path_to_file(self, dictionary, path_to_add, input_file=True, output_file=False, do_not_add_path=[]):
        """
        This function add a path in front of each value in a dictionary (provided by the parser) for option that are files or folders.
        This function can affect option files that represent input and/or output with "input_file" and output_file" parameters.
        Output is the same dictionary as provided but modified with added path.
        :param dictionary:
        :param path_to_add:
        :param input_file:
        :param output_file:
        :param do_not_add_path: list of keys for which path should NOT be added.
        :return:
        """
        for key, option in dictionary.items():
            # Check if option is present in this parser
            if key in self.options:
                # if key is listed in the do_not_add_path variable, do nothing
                if not key in do_not_add_path:
                    # If input file is a list, we need to check what type of list it is.
                    if isinstance(self.options[key].type_value, list):
                        for i, value in enumerate(option):
                            # check if value is a string
                            if isinstance(value, str):
                                # If value is a file, path must be updated
                                if os.path.isfile(os.path.join(path_to_add, value)):
                                    option[i] = os.path.join(path_to_add, value)
                    # if not a list:
                    else:
                        # If it contains files, it must be updated.
                        if self.options[key].type_value is None:
                            dictionary[key] = ''
                        elif (input_file and self.options[key].type_value in Option.OPTION_PATH_INPUT) or (output_file and self.options[key].type_value in Option.OPTION_PATH_OUTPUT):
                            # if the option contains an "no image file", do nothing
                            if self.options[key].list_no_image is not None:
                                if str(option) in self.options[key].list_no_image:
                                    dictionary[key] = option
                            else:
                                dictionary[key] = os.path.join(path_to_add, option)
            else:
                sct.printv("ERROR: the option you provided is not contained in this parser. Please check the dictionary", verbose=1, type='error')

        return dictionary

    def dictionary_to_string(self, dictionary):
        """
        This function transform a dictionary (key="-i", value="t2.nii.gz") into a string "-i t2.nii.gz".
        """
        result = ""
        for key, option in dictionary.items():
            if isinstance(option, list):
                result = result + ' ' + key + ' ' + self.options[key].type_value[0][0].join([str(op) for op in option])
            else:
                result = result + ' ' + key + ' ' + str(option)

        return result


########################################################################################################################
# USAGE
########################################################################################################################
class Usage:
    # Constructor
    def __init__(self, parser, file):
        self.file = file
        self.header = ''
        self.version = ''
        self.usage = ''
        self.example = ''
        self.description = ''
        self.arguments = parser.options
        #self.error = parser.errors
        self.arguments_string = ''
        self.section = dict()

#     def set_header(self):
#         from time import gmtime
#         from os.path import basename, getmtime
#         creation = gmtime(getmtime(self.file))
#         self.header = """
# """+basename(self.file)+"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>
# Version: """ + str(self.get_sct_version())

    def set_description(self, description):
        self.description = '\nDESCRIPTION\n' + self.align(description, length=100, pad=0)

    def addSection(self, section):
        self.section[len(self.arguments) + 1] = section

    def set_usage(self):
        from os.path import basename
        self.usage = '\n\nUSAGE\n' + basename(self.file).strip('.py')
        sorted_arguments = sorted(self.arguments.items(), key=lambda x: x[1].order)
        mandatory = [opt[0] for opt in sorted_arguments if self.arguments[opt[0]].mandatory]
        for opt in mandatory:
            self.usage += ' ' + opt + ' ' + self.refactor_type_value(opt)
            # if self.arguments[opt].type_value == 'multiple_choice':
            #     self.usage += ' ' + opt + ' ' + str(self.arguments[opt].example)
            # elif isinstance(self.arguments[opt].type_value, list):
            #     self.usage += ' ' + opt + ' <list of: ' + str(self.arguments[opt].type_value[1]) + '>'
            # else:
            #     self.usage += ' ' + opt + ' <' + str(self.arguments[opt].type_value) + '>'
        self.usage += '\n'

    def set_arguments(self):
        sorted_arguments = sorted(self.arguments.items(), key=lambda x: x[1].order)
        mandatory = [opt[0] for opt in sorted_arguments if self.arguments[opt[0]].mandatory and not self.arguments[opt[0]].deprecated_by]
        optional = [opt[0] for opt in sorted_arguments if not self.arguments[opt[0]].mandatory and not self.arguments[opt[0]].deprecated_by]
        if mandatory:
            self.arguments_string = '\nMANDATORY ARGUMENTS\n'
            for opt in mandatory:
                # check if section description has to been displayed
                if self.arguments[opt].order in self.section:
                    self.arguments_string += self.section[self.arguments[opt].order] + '\n'
                # display argument
                type_value = self.refactor_type_value(opt)
                line = [" " + opt + " " + type_value, self.align(self.arguments[opt].description)]
                self.arguments_string += self.tab(line) + '\n'
        if optional:
            self.arguments_string += '\nOPTIONAL ARGUMENTS\n'
            for opt in optional:
                # check if section description has to been displayed
                if self.arguments[opt].order in self.section:
                    self.arguments_string += self.section[self.arguments[opt].order] + '\n'
                # display argument
                type_value = self.refactor_type_value(opt)
                description = self.arguments[opt].description
                if self.arguments[opt].default_value:
                    description += " Default value = " + str(self.arguments[opt].default_value)
                if self.arguments[opt].deprecated:
                    description += " Deprecated argument!"
                line = [" " + opt + " " + type_value, self.align(description)]
                self.arguments_string += self.tab(line) + '\n'

        if len(self.arguments) + 1 in self.section:
            self.arguments_string += self.section[len(self.arguments) + 1] + '\n'

    def refactor_type_value(self, opt):
        if self.arguments[opt].type_value is None:
            type_value = ''
        elif self.arguments[opt].type_value == 'multiple_choice':
            type_value = self.print_list_with_brackets(self.arguments[opt].example)
        elif type(self.arguments[opt].type_value) is list:
            type_value = '<list of: ' + str(self.arguments[opt].type_value[1]) + '>'
        else:
            type_value = '<' + self.arguments[opt].type_value + '>'
        return type_value

    def set_example(self):
        from os.path import basename
        self.example = '\nEXAMPLE\n' + \
            basename(self.file).strip('.py')
        sorted_arguments = sorted(self.arguments.items(), key=lambda x: x[1].order)
        for opt in [opt[0] for opt in sorted_arguments if self.arguments[opt[0]].example and not self.arguments[opt[0]].deprecated_by]:
            if type(self.arguments[opt].example) is list:
                self.example += ' ' + opt + ' ' + str(self.arguments[opt].example[0])
            else:
                self.example += ' ' + opt + ' ' + str(self.arguments[opt].example)

    def generate(self, error=None):
        # self.set_header()
        self.set_arguments()
        self.set_usage()
        self.set_example()
        # removed example: https://github.com/neuropoly/spinalcordtoolbox/issues/957
        # usage = self.header + self.description + self.usage + self.arguments_string + self.example + '\n'
        usage = self.header + self.description + self.usage + self.arguments_string

        if error:
            sct.printv(error + '\nAborted...', type='warning')
            sct.printv(usage, type='normal')
            raise SyntaxError(error)
            exit(1)
        else:
            return usage

    def error(self, error=None):
        if error:
            self.generate(error)
        else:
            sct.printv(self.generate())
            from sys import exit
            exit(0)

    def print_list_with_brackets(self, l):
        type_value = '{'
        for char in l:
            type_value += str(char) + ','
        type_value = type_value[:-1]
        type_value += '}'
        return type_value

    def tab(self, strings):
        """
        This function is used for arguments usage's field to vertically align words
        :param strings: list of string to align vertically
        :return: string with aligned strings
        """
        tab = ''
        for string in strings:
            if len(string) < 30:
                spaces = ' ' * (30 - len(string))
                string += spaces
            tab += string

        return tab

    def align(self, string, length=70, pad=30):
        """
        This function split a string into a list of 100 char max strings
        :param string: string to split
        :param length: maximum length of a string, default=70
        :param pad: blank space in front of the string, default=30
        :return: string with \n separator
        """
        s = ''
        strings = []

        # check if "\n" are present in the string. If so, decompose the string.
        string_split_line = string.split('\n')
        if len(string_split_line) > 1:
            for i in range(0, len(string_split_line)):
                if i != 0:
                    string_split_line[i] = '  ' + string_split_line[i]

        # check if a string length is over "length"
        for k, stri in enumerate(string_split_line):
            i = 0
            for c in stri:
                i += 1
                if c == ' ':
                    last_space = i
                if i%length == 0:
                    strings.append(stri[0:last_space])
                    stri = stri[last_space:]
                    if k != 0:
                        stri = '    ' + stri
                    i = i - last_space
            strings.append(stri)

        # Concatenate strings
        for i, yes in enumerate(strings):
            if i != 0:
                s += ' ' * pad
            s += yes
            if i != len(strings) - 1:
                s += '\n'
        return s

########################################################################################################################
# GENERATION OF SOURCEFORGE GENERATED DOC
########################################################################################################################


class DocSourceForge:
    # Constructor
    def __init__(self, parser, file):
        self.file = file
        self.parser = parser
        self.header = ''
        self.version = ''
        self.usage = ''
        self.example = ''
        self.description = ''
        self.arguments = parser.options
        #self.error = parser.errors
        self.arguments_string = ''
        self.section = dict()

#     def set_header(self):
#         from time import gmtime
#         from os.path import basename, getmtime
#         creation = gmtime(getmtime(self.file))
#         self.header = """
# """+basename(self.file)+"""
# ------"""

    def set_description(self, description):
        self.description = '-----------\n#####DESCRIPTION#####\n' + self.align(description, length=100, pad=0)

    def addSection(self, section):
        self.section[len(self.arguments) + 1] = section

    def set_usage(self):
        from os.path import basename
        self.usage = '\n\n#####USAGE#####\n`' + basename(self.file)
        sorted_arguments = sorted(self.arguments.items(), key=lambda x: x[1].order)
        mandatory = [opt[0] for opt in sorted_arguments if self.arguments[opt[0]].mandatory]
        for opt in mandatory:
            if self.arguments[opt].type_value == 'multiple_choice':
                self.usage += ' ' + opt + ' ' + str(self.arguments[opt].example)
            else:
                self.usage += ' ' + opt + ' <' + str(self.arguments[opt].type_value) + '>'
        self.usage += '`\n'

    def set_arguments(self):
        sorted_arguments = sorted(self.arguments.items(), key=lambda x: x[1].order)
        mandatory = [opt[0] for opt in sorted_arguments if self.arguments[opt[0]].mandatory]
        optional = [opt[0] for opt in sorted_arguments if not self.arguments[opt[0]].mandatory]
        if mandatory:
            self.arguments_string = '\n\nMANDATORY ARGUMENTS  |` `\n--------------------|---\n'
            for opt in mandatory:
                self.arguments_string += '`'
                # check if section description has to been displayed
                if self.arguments[opt].order in self.section:
                    self.arguments_string += self.section[self.arguments[opt].order] + '\n'
                # display argument
                type_value = self.refactor_type_value(opt)
                line = ["  " + opt + " " + type_value + '`', '|' + self.arguments[opt].description]
                self.arguments_string += self.tab(line) + '\n'
        if optional:
            self.arguments_string += '\n\nOPTIONAL ARGUMENTS  |` `\n--------------------|---\n'
            for opt in optional:
                if not self.arguments[opt].deprecated_rm and not self.arguments[opt].deprecated and self.arguments[opt].deprecated_by is None:
                    self.arguments_string += '`'
                    # check if section description has to been displayed
                    if self.arguments[opt].order in self.section:
                        self.arguments_string += self.section[self.arguments[opt].order] + '\n'
                    # display argument
                    type_value = self.refactor_type_value(opt)
                    line = ["  " + opt + " " + type_value + '`', '|' + self.arguments[opt].description]
                    self.arguments_string += self.tab(line) + '\n'

    def refactor_type_value(self, opt):
        if self.arguments[opt].type_value is None:
            type_value = ''
        elif self.arguments[opt].type_value == 'multiple_choice':
            type_value = self.print_list_with_brackets(self.arguments[opt].example)
        elif type(self.arguments[opt].type_value) is list:
            type_value = '<list>'
        else:
            type_value = '<' + self.arguments[opt].type_value + '>'
        return type_value

    def set_example(self):
        from os.path import basename
        self.example = '\n\n#####EXAMPLE#####\n' + \
            '`' + basename(self.file)
        sorted_arguments = sorted(self.arguments.items(), key=lambda x: x[1].order)
        mandatory = [opt[0] for opt in sorted_arguments if self.arguments[opt[0]].mandatory]
        for opt in [opt[0] for opt in sorted_arguments if self.arguments[opt[0]].example]:
            if type(self.arguments[opt].example) is list:
                self.example += ' ' + opt + ' ' + str(self.arguments[opt].example[0])
            else:
                self.example += ' ' + opt + ' ' + str(self.arguments[opt].example)
        self.example += '`'

    def generate(self, error=None):
        # self.set_header()
        self.set_description(self.parser.usage.description[2 + len('description'):])
        self.set_arguments()
        self.set_usage()
        self.set_example()
        # removed example: https://github.com/neuropoly/spinalcordtoolbox/issues/957
        # doc = self.header + self.description + self.usage + self.arguments_string + self.example
        doc = self.header + self.description + self.usage + self.arguments_string
        from os.path import basename
        file_doc_sf = open('doc_sf_' + basename(self.file)[:-3] + '.txt', 'w')
        file_doc_sf.write(doc)
        file_doc_sf.close()

        sct.printv(doc)

    def error(self, error=None):
        if error:
            self.generate(error)
        else:
            sct.printv(self.generate())
            from sys import exit
            exit(0)

    def print_list_with_brackets(self, l):
        type_value = '{'
        for char in l:
            type_value += str(char) + ','
        type_value = type_value[:-1]
        type_value += '}'
        return type_value

    def tab(self, strings):
        """
        This function is used for arguments usage's field to vertically align words
        :param strings: list of string to align vertically
        :return: string with aligned strings
        """
        tab = ''
        for string in strings:
            if len(string) < 30:
                spaces = ' ' * (30 - len(string))
                string += spaces
            tab += string

        return tab

    def align(self, string, length=70, pad=30):
        """
        This function split a string into a list of 100 char max strings
        :param string: string to split
        :param length: maximum length of a string, default=70
        :param pad: blank space in front of the string, default=30
        :return: string with \n separator
        """
        s = ''
        strings = []

        # check if "\n" are present in the string. If so, decompose the string.
        string_split_line = string.split('\n')
        if len(string_split_line) > 1:
            for i in range(0, len(string_split_line)):
                if i != 0:
                    string_split_line[i] = '  ' + string_split_line[i]

        # check if a string length is over "length"
        for k, stri in enumerate(string_split_line):
            i = 0
            for c in stri:
                i += 1
                if c == ' ':
                    last_space = i
                if i%length == 0:
                    strings.append(stri[0:last_space])
                    stri = stri[last_space:]
                    if k != 0:
                        stri = '    ' + stri
                    i = i - last_space
            strings.append(stri)

        # Concatenate strings
        for i, yes in enumerate(strings):
            if i != 0:
                s += ' ' * pad
            s += yes
            if i != len(strings) - 1:
                s += '\n'
        return s


########################################################################################################################
# SPELLING CHECKER
########################################################################################################################

class SpellingChecker:
    # spelling checker from http://norvig.com/spell-correct.html
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz-_0123456789'

    # wirds_dict must be a list of string
    def setWordsAsList(self, words_dict):
        self.NWORDS = self.train(words_dict)

    # text must be a string with all the word, separated with space of \n
    def setWordsAsText(self, text):
        self.NWORDS = self.train(self.words(text))

    # fname must be the path of the file containing the dictionary
    def setWordsAsFile(self, fname):
        self.NWORDS = self.train(self.words(file(fname).read()))

    def words(self, text):
        from re import findall
        return findall('[a-z]+', text.lower())

    def train(self, features):
        from collections import defaultdict
        model = defaultdict(lambda: 1)
        for f in features:
            model[f] += 1
        return model

    def edits1(self, word):
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces   = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
        inserts    = [a + c + b     for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.NWORDS)

    def known(self, words): return set(w for w in words if w in self.NWORDS)

    def correct(self, word):
        return self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word)
        # return max(candidates, key=self.NWORDS.get) # return the most potential candidate
