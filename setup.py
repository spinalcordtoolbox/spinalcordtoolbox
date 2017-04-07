# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, './install/requirements/requirementsPip.txt')) as f:
    requirements = f.read().splitlines()

with open(path.join(here, 'version.txt')) as f:
    version = f.read().split()

setup(
    name='spinalcordtoolbox',
    description='Library of analysis tools for the MRI of the spinal cord',
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
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='Image MRI spinal cord',
    packages=find_packages(exclude=['dev', 'install']),
    install_requires=requirements,
    include_package_data=True,

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={'console_scripts': console_scripts, },
)
