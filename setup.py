# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from codecs import open
from os import path
#import os

#os.environ['MPLCONFIGDIR'] = "."

version = "3.0.0"
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

with open(path.join(here, 'requirements_dev.txt')) as f:
    requirements_dev = f.read().splitlines()

setup(
    name='spinalcordtoolbox',
    version=version,
    description='Library of analysis tools for the MRI of the spinal cord',
    long_description=long_description,
    url= 'http://www.neuro.polymtl.ca/home',
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
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='Image MRI spinal cord',
    packages=['spinalcordtoolbox', 'scripts', 'dev', 'testing', 'bin'],
    install_requires=requirements,

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
       'bin': ['bin/isct_*'],
    },
    include_package_data=True,

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[
    #    ('my_data', ['data/data_file'])
    # ],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
       'console_scripts': [
           'sct_testing=scripts.sct_testing:main',
           'sct_apply_transfo=scripts.sct_apply_transfo:main'
       ],
    },
)
