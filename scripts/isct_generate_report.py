#!/usr/bin/env python
#########################################################################################
#
#  qc report  generation function implementation
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Thierno Barry
# Modified: 2016-11-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import os
import webbrowser
import msct_report as report
import msct_report_item as report_item


def generate_report(description, syntax, report_dir, show_report):
    """
    :param description:
    :param syntax:
    :param report_dir:
    :return:
    """
    # create new  or get instance of  the report object
    sct_report = report.Report(report_exists(report_dir), report_dir)

    # create sct_report_item
    sct_report_item = report_item.ReportItem(report_dir, syntax, description)

    # append  item to a new or existing report
    sct_report.append_item(sct_report_item)

    # update index file and menu links
    sct_report.refresh_index_file()

    print " \n The qc report has been generated in {} \n".format(os.path.realpath(report_dir))

    if show_report:
        # display report in the default web browser
        url = 'file://{}'.format(os.path.realpath(os.path.join(report_dir, "index.html")))
        webbrowser.open(url)

def report_exists(report_dir):
    """
    check if a report folder already exists and assets is copied inside the folder
    :param report_dir:
    :return:true or false
    """
    return os.path.isdir(os.path.join(report_dir, "assets"))
