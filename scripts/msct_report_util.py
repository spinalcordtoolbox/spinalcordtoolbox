# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#!/usr/bin/env python
#########################################################################################
#
#  utils functions used for qc report  generation
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Thierno Barry
# Modified: 2016-11-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from jinja2 import Environment, FileSystemLoader


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
