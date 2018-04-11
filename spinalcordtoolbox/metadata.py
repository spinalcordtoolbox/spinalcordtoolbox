#!/usr/bin/env python
# -*- coding: utf-8
# Deal with SCT dataset metadata

import sys, io, os, re

class InfoLabel(object):
    def __init__(self):
        """
        """
        raise NotImplementedError()

    def load(self, file):
        """
        Load self from file
        :param file: input filename or file-like object
        """
        raise NotImplementedError()

    def save(self, file):
        """
        Save self to file
        :param file: output filename or file-like object
        """
        raise NotImplementedError()

