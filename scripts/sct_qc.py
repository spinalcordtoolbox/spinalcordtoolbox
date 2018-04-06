#!/usr/bin/env python
#
# Quality control viewer
#
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file LICENSE.TXT

"""Simple local http server to serve QC reports
"""

import io, sys, os
import shutil

if sys.hexversion > 0x03000000:
    from http.server import HTTPServer, SimpleHTTPRequestHandler
else:
    from BaseHTTPServer import HTTPServer
    from SimpleHTTPServer import SimpleHTTPRequestHandler
from string import Template

from msct_parser import Parser
import sct_utils as sct

import spinalcordtoolbox.reports as reports


def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('Quality Control Viewer')
    parser.add_option(name='-folder',
                      type_value='folder',
                      mandatory=True,
                      description='The root folder of the generated QC data')
    parser.add_option(name='-port',
                      type_value='int',
                      mandatory=False,
                      default_value=8888,
                      description='Port for the QC server')

    return parser


def _copy_assets(dest_path):
    assets_path = os.path.join(reports.__path__[0], 'assets')

    sct.copy(os.path.join(assets_path, 'index.html'), dest_path)

    for path in ['css', 'js', 'imgs', 'fonts']:
        src_path = os.path.join(assets_path, '_assets', path)
        dest_full_path = os.path.join(dest_path, '_assets', path)
        if not os.path.exists(dest_full_path):
            os.makedirs(dest_full_path)
        for file in os.listdir(src_path):
            sct.copy(os.path.join(src_path, file),
                         dest_full_path)


def _copy_data_in_html(html_file, json_file):
    with io.open(json_file, 'r', encoding='utf-8') as json_fh:
        tmp = Template(io.open(html_file, 'r', encoding='utf-8').read())
        output = tmp.substitute(sct_json_data=json_fh.read())
        io.open(html_file, 'w', encoding='utf-8').write(output)


if __name__ == "__main__":
    sct.start_stream_logger()
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    qc_path = arguments['-folder']
    qc_path = os.path.realpath(qc_path)
    qc_port = int(arguments['-port'])

    json_file = os.path.join(qc_path, 'qc_results.json')
    html_file = os.path.join(qc_path, 'index.html')

    if not os.path.isfile(json_file):
        sct.printv('Can not start the quality control viewer.'
               ' This is not a proper QC folder', type='error')
        sys.exit(-1)

    _copy_assets(qc_path)
    _copy_data_in_html(html_file, json_file)

    os.chdir(qc_path)
    httpd = HTTPServer(('', qc_port), SimpleHTTPRequestHandler)
    sct.printv('QC viewer started on:')
    sct.printv('http://127.0.0.1:{}'.format(qc_port), type='info')
    sct.printv('Copy and paste the address into your web browser')
    sct.printv('Press "Ctrl" + "C" to stop sct_qc')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sct.printv('QC viewer stopped')
