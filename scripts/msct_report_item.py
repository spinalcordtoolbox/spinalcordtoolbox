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
    def __init__(self, report_dir, syntax, description):
        self.report_dir = report_dir
        self.syntax = syntax
        self.contrast_name, self.tool_name = syntax.split(' ')
        self.images_dir = os.path.join("img", self.contrast_name, self.tool_name)
        self.description = msct_report_util.get_description_content(os.path.join(self.report_dir, self.images_dir,
                                                                                 description))
        self.images_link = []
        self.html_file_name = '{}-{}.html'.format(self.contrast_name, self.tool_name)

    def has_gif(self):
        #TODO:use array instead
        if self.tool_name.find("propseg") > - 1 or self.tool_name.find("label_vertebrae") > - 1:
            return True
        return False

    def add_image_link(self, link):
        """
        add image link  to a list of images link
        :param link:
        :return:
        """
        self.images_link.append(link)

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
            "hasGif": self.has_gif()
        }
        msct_report_util.createHtmlFile(template_dir, template_name, file_link, tags)


class Image:
    """
    image class to represent img content in the html file
    """

    def __init__(self, name, src):
        self.name = name
        self.src = src
