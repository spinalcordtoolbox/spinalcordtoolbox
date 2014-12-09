#!/usr/bin/env python
#########################################################################################
#
# Parser
# Add option with name, type, short description, mandatory or not, example using add_option method.
# If the user make a misspelling, the parser will search in the option list what are nearest option and suggests it to the user
# Type of options are:
# - file, folder (check existence)
# - folder_creation (check existence and if does not exist, create it)
# - str, int, float, long, complex (check if input is the correct type)
# - multiple_choice
# - None, return True when detected
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
#     print string_usage
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
import time
import sys
import commands
import sct_utils as sct
import re, collections
import datetime
from itertools import *

########################################################################################################################
####### OPTION
########################################################################################################################

class Option:
    # list of option type that can be casted
    OPTION_TYPES = ["str","int","float","long","complex"]

    ## Constructor
    def __init__(self, name, type_value, description, mandatory, example, default_value, help, parser):
        self.name = name
        self.type_value = type_value
        self.description = description
        self.mandatory = mandatory
        self.example = example
        self.default_value = default_value
        self.help = help
        self.parser = parser

    def __safe_cast__(self, val, to_type):
        return to_type(val)

    # Do we need to stop the execution if the input is not correct?
    def check_integrity(self, arguments, index):
        # arguments[index] as the option (example: '-input')
        # & argmuments[index+1] must be the corresponding arg (ex: 't2.nii.gz')
        
        if len(arguments) > index+1: # Check if option is not the last item
            param = arguments[index+1]
        else:
            self.parser.usage.error("ERROR: Option " + self.name + " needs an argument...")

        if param in self.parser.options: # check if option has an argument that is not another option
            self.parser.usage.error("ERROR: Option " + self.name + " needs an argument...")

        
        ###############################################################################
        # check integrity of each option type
        if self.type_value in self.OPTION_TYPES:
            # check if a int is really a int (same for str, float, long and complex)
            try:
                return self.__safe_cast__(param,eval(self.type_value))
            except ValueError:
                self.parser.usage.error("ERROR: Option "+self.name+" must be "+self.type_value)

        elif self.type_value == "file":
            # check if the file exist
            sct.printv("Check file existence...")
            sct.check_file_exist(param,1)
            return param

        elif self.type_value == "folder":
            # check if the folder exist. If not, create it.
            sct.printv("Check folder existence...")
            sct.check_folder_exist(param,1)
            return param

        elif self.type_value == "folder_creation":
            # check if the folder exist. If not, create it.
            sct.printv("Check folder existence...")
            result_creation = sct.create_folder(param)
            if result_creation == 2:
                sct.printv("ERROR: Permission denied for folder creation...",type="error")
            elif result_creation == 1:
                sct.printv("Folder "+param+" has been created.",type='warning')
            return param

        elif type(self.type_value) is list:
            if param not in self.type_value:
                self.parser.usage.error(self.name + " only takes " + print_list_with_brackets(self.type_value) + " as potential arguments.")
        
        else:
            self.parser.usage.error("ERROR: Type of option \"" + self.type_value +"\" is not supported by the parser.")

########################################################################################################################
####### PARSER
########################################################################################################################

class Parser:
    ## Constructor
    def __init__(self, __file__):
        self.options = dict()
        self.spelling = SpellingChecker()
        self.errors = ''
        self.usage = Usage(self, __file__)

    def add_option(self, name, type_value=None, description=None, mandatory=False, example=None, help=None, default_value=None):
        self.options[name] = Option(name, type_value, description, mandatory, example, default_value, help, self)

    def parse(self, arguments):
        # if no arguments, print usage and quit
        if len(arguments) == 0:
            self.usage.error()

        # initialize results
        dictionary = dict()

        # initialize the spelling checker
        self.spelling.setWordsAsList([name for name in self.options])

        skip = False
        for index,arg in enumerate(arguments):
            if skip: # if argument need to be skipped, we pass
                skip = False
                continue

            if arg in self.options:
                # for each argument, check if is in the option list.
                # if so, check the integrity of the argument
                if self.options[arg].type_value:
                    dictionary[arg] = self.options[arg].check_integrity(arguments, index)
                    skip = True
                else:
                    dictionary[arg] = True
            else:
                # if not in the list of known options, there is a syntax error in the list of arguments
                # check if the input argument is close to a known option
                spelling_candidates = self.spelling.correct(arg)
                if len(spelling_candidates) != 0:
                    self.usage.error(" Did you mean: "+', '.join(spelling_candidates) + '?')
                else:
                    self.usage.error("ERROR: wrong input arguments. See documentation.")

        # check if all mandatory arguments are provided by the user
        if dictionary:
            for option in [opt for opt in self.options if self.options[opt].mandatory]:
                if option not in dictionary:
                    self.usage.error('ERROR: ' + option + ' is a mandatory argument.\n')

        # return a dictionary with each option name as a key and the input as the value
        return dictionary

########################################################################################################################
####### USAGE
########################################################################################################################

class Usage:

    # Constructor
    def __init__(self, parser, file):
        self.file = (file)
        self.header = ''
        self.version = ''
        self.usage = ''
        self.example = ''
        self.description = ''
        self.arguments = parser.options
        #self.error = parser.errors
        self.arguments_string = ''

    def set_header(self):
        creation = time.gmtime(os.path.getmtime(__file__))
        self.header = """
"""+os.path.basename(self.file)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>
last modified on """ + str(creation[0]) + '-' + str(creation[1]) + '-' +str(creation[2])

    def set_description(self, description):
        self.description = '\n\n    DESCRIPTION\n' + align(description)

    def set_usage(self):
        self.usage = '\n\n    USAGE\n' + os.path.basename(self.file)
                     #+ str([arg for arg in self.arguments])
        for opt in self.arguments:
            self.usage += '     ' + opt + ' ' + str(self.arguments[opt].type_value)

    def set_arguments(self):
        mandatory = [opt for opt in self.arguments if self.arguments[opt].mandatory]
        optional = [opt for opt in self.arguments if not self.arguments[opt].mandatory]
        #optional = self.arguments
        #optional = mandatory
        if mandatory:
            self.arguments_string = '\n\n    MANDATORY ARGUMENTS\n'
            for opt in mandatory:
                type_value = self.refactor_type_value(opt)
                line = [opt, type_value, self.arguments[opt].description]
                self.arguments_string += tab(line) + '\n'
        if optional:
            self.arguments_string += '\n\n    OPTIONAL ARGUMENTS\n'
            for opt in optional:
                type_value = self.refactor_type_value(opt)
                line = [opt, type_value, self.arguments[opt].description]
                self.arguments_string += tab(line) + '\n'

    def refactor_type_value(self, opt):
        if type(self.arguments[opt].type_value) is not list:
            type_value = '<' + self.arguments[opt].type_value + '>'
        else:
            type_value = print_list_with_brackets(self.arguments[opt].type_value)
        return type_value

    def set_example(self):
        self.example = '\n\n    EXAMPLE\n' + \
            os.path.basename(self.file)
        for opt in [opt for opt in self.arguments if self.arguments[opt].example]:
            self.example += ' ' + opt + ' ' + str(self.arguments[opt].example)

    def generate(self, error=None):
        self.set_header()
        self.set_arguments()
        self.set_usage()
        self.set_example()
        usage = self.header + self.description + self.usage + self.arguments_string + self.example

        if error:
            sct.printv(error+'\nAborted...',type='warning')
            sct.printv(usage,type='normal')
            exit(1)
        else:
            return usage

    def error(self, error=None):
        if error:
            self.generate(error)
        else:
            print self.generate()
            sys.exit(0)

########################################################################################################################
####### SPELLING CHECKER
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

    def words(self, text): return re.findall('[a-z]+', text.lower())

    def train(self, features):
        model = collections.defaultdict(lambda: 1)
        for f in features:
            model[f] += 1
        return model

    def edits1(self, word):
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
        replaces   = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
        inserts    = [a + c + b     for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def known_edits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.NWORDS)

    def known(self, words): return set(w for w in words if w in self.NWORDS)

    def correct(self, word):
        return self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) #
        #return max(candidates, key=self.NWORDS.get) # return the most potential candidate

########################################################################################################################

########################################################################################################################


def print_list_with_brackets(list):
    type_value = '{'
    for char in list:
        type_value += str(char) + ','
    type_value = type_value[:-1]
    type_value += '}'
    return type_value

# This function is used for arguments usage's field to verticaly align words
def tab(strings):
    tab = ''
    for string in strings:
        if len(string) < 20:
            spaces = ' '*(20 - len(string))
            string += spaces
        tab += string

    return tab


# This function split a string into a list of 100 char max strings
def align(string, pad=100):
    i = 0
    s = ''
    strings = []
    for c in string:
        i += 1
        if c == ' ':
            last_space = i
        if i%pad == 0:
            strings.append(string[0:last_space])
            string = string[last_space:-1]
            i = i - last_space
    strings.append(string)
    for yes in strings:
        s += yes + '\n'
    return s
