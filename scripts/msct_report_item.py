import os
import shutil
import msct_report_config
import msct_report_util


class ReportItem:
    def __init__(self,reportDir, syntax, description):
        self.reportDir = reportDir
        self.syntax = syntax
        self.contrastName, self.toolName = syntax.split(' ')
        self.description = description
        self.imagesDir = os.path.join("img", self.contrastName,self.toolName)
        self.imagesLink = []
        self.htmlFileName = '{}-{}.html'.format(self.contrastName, self.toolName)
        return

    def addImageLink(self, link):
        """
        add image link  to a list of images link
        :param link:
        :return:
        """
        self.imagesLink.append(link)

    def generateHtmlFromTemplate(self, templateDir, templateName):
        """
        create new hmtl file  in report dir  based on contraste_tool template
        :param reportDir:
        :param templateDir:
        :param templateName:
        :return:
        """
        fileLink = os.path.join(self.reportDir, self.htmlFileName)
        tags = {
            'description': self.description,
            'syntax': self.syntax,
            'images': self.imagesLink
        }
        msct_report_util.createHtmlFile(templateDir, templateName, fileLink, tags)

        return
