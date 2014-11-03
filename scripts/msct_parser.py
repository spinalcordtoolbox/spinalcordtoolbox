#!/usr/bin/env python
#########################################################################################
#
# Parser
# Add option with name, type, default value and help using add_option method.
# If the user make a misspelling, the parser will search in the option list what are nearest option and suggests it to the user
# Type of options are:
# - file (check file existence)
# - str, int, float, long, complex (check if input is the correct type)
# - None, return True when detected
#
# Usage:
# from msct_parser import *
# parser = Parser()
# parser.add_option("-input","file")
# parser.add_option("-test","int")
# parser.add_option("-test2") # this is a option without
# arguments = parser.parse(sys.argv[1:])
# 
# Arguments are available directly:
# fname_input = arguments["-input"]
# test_int_value = arguments["-test"]
#
# TO DO:
# - generate the usage based on the option list
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# Created: 2014-10-27
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sys
import commands
import sct_utils as sct
import re, collections
from itertools import *



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

class Option:
    # list of option type that can be casted
    OPTION_TYPES = ["str","int","float","long","complex"]

    ## Constructor
    def __init__(self, name, type_value, mandatory, default_value, help):
        self.name = name
        self.type_value = type_value
        self.mandatory = mandatory
        self.default_value = default_value
        self.help = help

    def safe_cast(self, val, to_type):
        return to_type(val)

    # Do we need to stop the execution if the input is not correct?
    def check_integrity(self, param):
        if self.type_value in self.OPTION_TYPES:
            try:
                return self.safe_cast(param,eval(self.type_value))
            except ValueError:
                sct.printv("Error: Option "+self.name+" must be "+self.type_value,1,"error")
        elif self.type_value == "file":
            sct.printv("Check file existence...",1)
            sct.check_file_exist(param,1)
            return param
        else:
            sct.printv("Error: type of option \""+self.type_value+"\" is not supported by the parser.",1,"error")

class Parser:
    ## Constructor
    def __init__(self):
        self.options = dict()
        self.spelling = SpellingChecker()

    def add_option(self, name, type_value=None, mandatory=False, help=None, default_value=None):
        self.options[name] = Option(name, type_value, mandatory, default_value, help)

    def parse(self, arguments):
        # initialize results
        dictionary = dict()

        # initialize the spelling checker
        self.spelling.setWordsAsList([name for name in self.options])

        skip = False
        for index,arg in enumerate(arguments):
            # if argument need to be skipped, we pass
            if skip:
                skip = False
                continue
            # for each argument, check if is in the option list.
            # if so, check the integrity of the argument
            if arg in self.options:
                if self.options[arg].type_value:
                    argument = self.options[arg].check_integrity(arguments[index+1])
                    dictionary[arg] = argument
                    skip = True
                else:
                    dictionary[arg] = True
            # if not in the list of known options, there is a syntax error in the list of arguments
            # check if the input argument is close to a known option
            else:
                serror = "Error: wrong input arguments. See documentation."
                spelling_candidates = self.spelling.correct(arg)
                if (len(spelling_candidates)!=0): serror=serror+" Did you mean: "+', '.join(spelling_candidates)
                sct.printv(serror,1,"error")

        # check if all mandatory arguments are provided by the user
        for option in [opt for opt in self.options if opt.mandatory]:
            if option not in dictionary:
                sct.printv("Error: blablabla",1,"error")

        # return a dictionary with each option name as a key and the input as the value
        return dictionary

class Usage:
    # Constructor
    def __init__(self):
        self.description
        self.usage
        self.argument

        self.exemple
        self.