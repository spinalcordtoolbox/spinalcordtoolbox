import os
import msct_report_util
import msct_report_config
import  webbrowser
import msct_report as report
import msct_report_item as reportItem


def generate_report(description, syntax, reportDir):
    """
    :param description:
    :param syntaxt:
    :param qcImages:
    :param reportDir:
    :return:
    """
    print "start generation du rapport html \n !"

    print os.path.dirname(os.path.realpath(__file__));
    # create new  or get instance of  the report object
    sct_report = report.Report(reportExits(reportDir), reportDir)

    #create sct_report_item
    sct_report_item = reportItem.ReportItem(reportDir,syntax, msct_report_util.getTxtContent(description))

    # append  item to a new or existing report
    sct_report.appendItem(sct_report_item)

    # update index file and menu links
    sct_report.refreshIndexFile()

    print "end generation du rapport ! \n"

    #display report in the default web browser

    url = 'file://{}'.format(os.path.realpath(os.path.join(reportDir, "index.html")))
    webbrowser.open(url)

    return


def reportExits(reportDir):
    """
    check if a report folder already exists and assets is copied inside the folder
    :param reportDir:
    :return:
    """
    return os.path.isdir(os.path.join(reportDir, msct_report_config.assetsDirName))

