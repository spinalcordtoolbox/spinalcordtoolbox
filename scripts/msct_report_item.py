#!/usr/bin/env python
#########################################################################################
#
# Report Item  class implementation
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Thierno Barry
# Modified: 2016-11-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import os
import msct_report_util


class ReportItem:
    def __init__(self, report_dir, syntax, description, subject_name,cross):
        self.report_dir = report_dir
        self.color_bar = None
        self.syntax = syntax
        self.subject_name = subject_name
        self.cross = cross
        self.contrast_name, self.tool_name = syntax.split(' ')
        self.images_dir = os.path.join("img", self.contrast_name, self.tool_name)
        self.description = msct_report_util.get_txt_content(os.path.join(self.report_dir, self.images_dir,
                                                                         description))
        self.images_link = []
        self.txt_contents = []
        self.html_file_name = '{}-{}.html'.format(self.contrast_name, self.tool_name)

    def has_gif(self):
        # TODO:use blacklist array for prevent gif switch on the view
        # to prevent gif on propseg do : self.tool_name.find("propseg") > - 1 === true return False
        return True

    def set_color_bar(self, bar):
        """

        :param color:
        :return:
        """
        self.color_bar = bar

    def add_image_link(self, link):
        """
        add image link  to a list of images link
        :param link:
        :return:
        """
        self.images_link.append(link)

    def add_txt_content(self, txt):
        """
        add txt file to the arrays of txt
        :param txt:
        :return:
        """
        self.txt_contents.append(txt)

    def generate_html_from_template(self, template_dir, template_name):
        """
        create new hmtl file  in report dir  based on contraste_tool template
        :param template_dir:
        :param template_name:
        :return:
        """
        file_link = os.path.join(self.report_dir, self.html_file_name)
        tags = {
            'description': self.description,
            'syntax': self.syntax,
            'images': self.images_link,
            'texts': self.txt_contents,
            "hasGif": self.has_gif(),
            'subject_name':self.subject_name,
            'color_bar': self.color_bar,
            'cross': self.cross
        }
        msct_report_util.createHtmlFile(template_dir, template_name, file_link, tags)


class Image:
    """
    image class to represent img content in the html file
    """

    def __init__(self, name, src):
        self.name = name
        self.src = src


class Txt:
    """
    txt class to represent file txt name and his description in the html file
    """

    def __init__(self, name, path):
        self.name = name
        self.content = msct_report_util.get_txt_content(path)
