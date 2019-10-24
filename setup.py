# -*- coding: utf-8 -*-

import sys, os
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'version.txt')) as f:
    version = f.read().strip()

setup(
    name='spinalcordtoolbox',
    version=version,
    description='Library of analysis tools for MRI of the spinal cord',
    long_description=long_description,
    url='http://www.neuro.polymtl.ca/home',
    author='NeuroPoly Lab, Polytechnique Montreal',
    author_email='neuropoly@googlegroups.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='Magnetic Resonance Imaging MRI spinal cord analysis template',
    packages=[
     "spinalcordtoolbox",
    ],
    #package_data={},
    data_files=[
     # <hack>
     ("sct_scripts", [ os.path.join("scripts", x) for x in os.listdir("scripts") if x.endswith(".py") ]),
     # </hack>
    ],
    include_package_data=True,
    extras_require={
     'docs': [
      'sphinx',
      'sphinxcontrib-programoutput',
      'sphinx_rtd_theme',
     ],
     'mpi': [
      'mpich==3.2',
      'mpi4py==3.0.0',
     ],
     # 'test': [
     #  "pytest-runner",
     #  "pytest",
     # ],
    },
    entry_points=dict(
     console_scripts=[
     # <hack>
     ] + ['{}=spinalcordtoolbox.compat.launcher:main'.format(os.path.splitext(x)[0]) for x in os.listdir("scripts") if x.endswith(".py")] + [
     # </hack>
      # TODO add proper command-line entry points from refactored code
      #'sct_deepseg_gm=spinalcordtoolbox.deepseg_gm.__main__:main',
     ],
    ),
)
