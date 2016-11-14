import os
import shutil
import glob
from collections import OrderedDict
import msct_report_config
import msct_report_util
import msct_report_image


class Report:
    def __init__(self, exists, reportDir):
        self.dir = os.path.dirname(os.path.realpath(__file__));
        self.reportFolder = reportDir

        # TODO:the template link could  change in production
        self.templatesDirLink = os.path.join(self.dir,'..', msct_report_config.templatesDirName)

        #  copy all the assets file inside the new folder 
        if not exists:
            self.__createNew()

    def __createMenuLink(self, contraste, tool, id=None):
        item = {
            'name': tool,
            'link': '{}-{}'.format(contraste, tool)
        }
        return item

    def __getMenuLinks(self):
        """
        this function parse the current report folder and return the correspondind links by parsing html file names

        :return:
        """
        htmls = glob.glob1(self.reportFolder, "*.html")
        links =OrderedDict()
        if htmls:
            for item in htmls:
                rmvHtml = item.split('.')
                tmp = rmvHtml[0].split('-')
                if tmp.__len__() > 1:
                    if not tmp[0] in links:
                        links[tmp[0]] = [self.__createMenuLink(tmp[0], tmp[1])]
                    else:
                        links[tmp[0]].append(self.__createMenuLink(tmp[0], tmp[1]))
        return links

    def __createNew(self):
        """
        create a new report folder in the given directory
        :return:e
        """
        # copy assets  into sct_report dir
        shutil.copytree(os.path.join(self.templatesDirLink, msct_report_config.assetsDirName),
                        os.path.join(self.reportFolder, msct_report_config.assetsDirName))

        # copy the .config.json (TODO:its config really necessary)
        msct_report_util.copy(os.path.join(self.templatesDirLink, msct_report_config.reportConfigFileName), self.reportFolder,
                              msct_report_config.reportConfigFileName)

    def appendItem(self, item):
        """
        :param item:
        :return:
        """
        # get images link from qc images
        qcImagesItemLink = os.path.join(self.reportFolder,'img', item.contrastName, item.toolName)
        print "qcImagesItem",qcImagesItemLink
        if os.path.exists(qcImagesItemLink):
            #TODO:Marche pas bien =>take all png or jpeg
            imagesLink = glob.glob1(qcImagesItemLink, msct_report_config.imagesExt)
            if imagesLink:
                for img in imagesLink:
                    item.addImageLink(msct_report_image.Image(img, os.path.join(item.imagesDir, img)))
            else:
                print "no qc images in the current directory"
        else:
            raise Exception("qc images not founded")

        # generate html file for the item
        item.generateHtmlFromTemplate(self.templatesDirLink, msct_report_config.constrasteToolTemplate)

        return

    def refreshIndexFile(self):

        """

        :return:
        """
        fileLink = os.path.join(self.reportFolder, msct_report_config.indexTemplate)
        tags = {
            'links': self.__getMenuLinks()
        }
        msct_report_util.createHtmlFile(self.templatesDirLink, msct_report_config.indexTemplate, fileLink, tags)
        return
