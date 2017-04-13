# -*- coding: utf-8 -*-
import glob
import json
import logging
import os
from codecs import open

# import jinja2

logger = logging.getLogger(__file__)


def create_html_file(template_dir, template_name, destination_file, context):
    """

    Parameters
    ----------
    template_dir : str
    template_name : str
    destination_file : str
    context : dict
    """
    logger.debug('create_html_file: %s from %s', destination_file, os.path.join(template_dir, template_name))
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir), trim_blocks=True)
    tpl = env.get_template(template_name)
    html_code = tpl.render(context)
    with open(destination_file, 'w') as fd:
        fd.write(html_code)


class ReportGenerator(object):
    def __init__(self, params, context):
        """

        Parameters
        ----------
        params : spinalcordtoolbox.report.qc:Param
        """
        self.params = params
        self.context = context

    def generate_report_details(self):
        """

        Parameters
        ----------
        item : ReportItem
        """

        qc_images_item_link = os.path.join(self.report_folder, 'img', item.contrast_name, item.tool_name)
        if os.path.exists(qc_images_item_link):
            elements = glob.glob1(qc_images_item_link, "*")
            if elements:
                for el in elements:
                    if el.find(".png") > -1:
                        if el.find('colorbar') > -1:
                            item.set_color_bar(Image(el, os.path.join(item.images_dir, el)))
                        else:
                            item.add_image_link(Image(el, os.path.join(item.images_dir, el)))
                    elif el.find(".txt") > -1 and el.find("description") == -1:
                        item.add_txt_content(Txt(el, os.path.join(item.report_dir, item.images_dir, el)))
            else:
                logger.warn('No QC images not found %s', qc_images_item_link)
        else:
            raise IOError("No QC data not found")

        item.generate_html_from_template(self.templates_dir_link, self.contrast_tool_file_name)

    def refresh_index_file(self):
        """Update QC config file to add generated item
        """
        config_file = os.path.join(self.params.destination_folder, self.params.config_file_name)
        try:
            with open(config_file, encoding='utf-8') as fd:
                config = json.loads(fd.read())
        except IOError as err:
            logger.warn('Creating a new qc config file %s', config_file)
            config = []

        config.append(self.context)
        with open(config_file, 'w+', encoding='utf-8') as fd:
            fd.write(json.dumps(config))

    def generate_report(self):
        self.refresh_index_file()
        self.generate_report_details()


class ReportItem(object):
    def __init__(self, params, context):
        pass

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
            'color_bar': self.color_bar,
            'cross': self.cross
        }
        # create_html_file(template_dir, template_name, file_link, tags)


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
        with open(path) as fd:
            self.content = fd.read().replace('{', '').replace('}', '')
