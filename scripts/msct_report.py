#!/usr/bin/env python
#########################################################################################
#
# qc report  function implementation
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Thierno Barry
# Modified: 2016-11-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import os
import shutil
import glob
import msct_report_util
import msct_report_item as report_item


class Report:
    def __init__(self, exists, report_dir):
        # constants
        self.templates_dir_name = "qc_templates"
        self.assets_dir_name = "assets"
        self.contrast_tool_file_name = "contrast_tool.html"
        self.index_file_name = "index.html"

        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.report_folder = report_dir
        self.templates_dir_link = os.path.join(self.dir, '..', self.templates_dir_name)

        #  copy all the assets file inside the new folder 
        if not exists:
            self._create_new()

    @staticmethod
    def __create_menu_link(contrast, tool, id=None):
        item = {
            'name': tool,
            'link': '{}-{}'.format(contrast, tool)
        }
        return item

    def __get_last_created__(self):
        return Report.sorted_ls_by_ctime(self.report_folder)[-1].split('.')[0]

    def __get_menu_links__(self):
        """
        this function parse the current report folder and return the correspondind links by parsing html file names
        :return:
        """
        html_files = Report.sorted_ls_by_ctime(self.report_folder)
        links = {}
        for item in html_files:
            tmp = item.split('.')[0].split('-')
            if len(tmp) > 1:
                if not tmp[0] in links:
                    links[tmp[0]] = [self.__create_menu_link(tmp[0], tmp[1])]
                else:
                    links[tmp[0]].append(self.__create_menu_link(tmp[0], tmp[1]))
        return links

    @staticmethod
    def sorted_ls_by_ctime(path):
        """
        sort a directory by files creation time
        :param path:
        :return:
        """
        mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
        files = list(glob.glob1(path, "*.html"))
        if "index.html" in files : files.remove("index.html")
        return list(sorted(files, key=mtime))

    def _create_new(self):
        """
        create a new report folder in the given directory
        :return:e
        """
        # copy assets  into sct_report dir
        shutil.copytree(os.path.join(self.templates_dir_link, self.assets_dir_name),
                        os.path.join(self.report_folder, self.assets_dir_name))

    def append_item(self, item):
        """
        :param item:
        :return:
        """
        # get images link from qc images
        qc_images_item_link = os.path.join(self.report_folder, 'img', item.contrast_name, item.tool_name)
        if os.path.exists(qc_images_item_link):
            # TODO:Marche pas bien =>take all png or jpeg
            images_link = glob.glob1(qc_images_item_link, "*.png")
            if images_link:
                for img in images_link:
                    item.add_image_link(report_item.Image(img, os.path.join(item.images_dir, img)))
            else:
                print "no qc images in the current directory"
        else:
            raise Exception("qc images not founded")

        # generate html file for the item
        item.generate_html_from_template(self.templates_dir_link, self.contrast_tool_file_name)

    def refresh_index_file(self):
        """
         update index file to add generated item
        :return:
        """
        file_link = os.path.join(self.report_folder, self.index_file_name)
        tags = {
            'links': self.__get_menu_links__(),
            'idToPreload': self.__get_last_created__()
        }
        msct_report_util.createHtmlFile(self.templates_dir_link, self.index_file_name, file_link, tags)
