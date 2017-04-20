#!/usr/bin/env python
#
# Quality control viewer
#
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file LICENSE.TXT

import os
import shutil
import sys
from BaseHTTPServer import HTTPServer
from SimpleHTTPServer import SimpleHTTPRequestHandler

from msct_parser import Parser
from sct_utils import printv


def get_parser():
    parser = Parser(__file__)
    parser.usage.set_description('Quality Control Viewer')
    parser.add_option(name='-folder',
                      type_value='folder',
                      mandatory=True,
                      description='The root folder of the generated QC data')

    return parser


def _copy_assets(dest_path):
    home_dir = os.path.dirname(os.path.dirname(__file__))
    assets_path = os.path.join(home_dir, 'assets')

    shutil.copy2(os.path.join(assets_path, 'index.html'), dest_path)

    for path in ['css', 'js', 'imgs', 'fonts']:
        src_path = os.path.join(assets_path, '_assets', path)
        dest_full_path = os.path.join(dest_path, '_assets', path)
        if not os.path.exists(dest_full_path):
            os.makedirs(dest_full_path)
        for file in os.listdir(src_path):
            shutil.copy2(os.path.join(src_path, file),
                         dest_full_path)


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    qc_path = arguments['-folder']

    json_file = os.path.join(qc_path, 'qc_results.json')

    if not os.path.isfile(json_file):
        printv('Can not start the quality control viewer.'
               ' This is not a proper QC folder', type='error')
        sys.exit(-1)

    os.chdir(qc_path)
    _copy_assets(qc_path)

    httpd = HTTPServer(('', 8888), SimpleHTTPRequestHandler)
    printv('QC viewer started on:')
    printv('http://127.0.0.1:8888', type='info')
    printv('Copy and paste the address into your web browser')
    printv('Press "Ctrl" + "C" to stop sct_qc')
    httpd.serve_forever()
