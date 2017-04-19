
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file LICENSE.TXT
import os
import shutil

import click as click
import sys

from BaseHTTPServer import HTTPServer
from SimpleHTTPServer import SimpleHTTPRequestHandler


@click.group()
def spinalcordtoolbox():
    pass


@spinalcordtoolbox.command()
@click.argument('qc_path', type=click.Path(exists=True))
def qc(qc_path):
    config_file = os.path.join(qc_path, 'index.json')

    if not os.path.isfile(config_file):
        click.secho('Can not start the quality control viewer.'
                    ' This is not a proper QC folder', fg='red')
        sys.exit(-1)

    os.chdir(qc_path)
    #_copy_assets(qc_path)

    httpd = HTTPServer(('', 8888), SimpleHTTPRequestHandler)
    click.echo('QC viewer started on http://127.0.0.1:8888')
    httpd.serve_forever()


def _copy_assets(dest_path):
    import spinalcordtoolbox

    home_dir = os.path.dirname(spinalcordtoolbox.__file__)
    assets = os.path.join(home_dir, 'assets')

    shutil.copy2(os.path.join(assets, 'index.html'), dest_path)

    for path in ['js', 'css', 'libs', 'imgs']:
        os.makedirs(os.path.join(dest_path, 'assets', path))
        full_path = os.path.join(assets, path)
        _, _, files = os.walk(full_path)
        for file in files:
            shutil.copy2(os.path.join(full_path, file),
                         os.path.join(dest_path, 'assets', path))
