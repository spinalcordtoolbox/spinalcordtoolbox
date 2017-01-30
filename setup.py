# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from codecs import open
from os import path

commands = [
    'isct_check_detection',
    'isct_convert_binary_to_trilinear',
    'isct_minc2volume_viewer',
    'isct_test_ants',
    'isct_test_function',
    'isct_warpmovie_generator',
    'msct_image',
    'msct_multiatlas_seg',
    'sct_apply_transfo',
    'sct_average_data_within_mask',
    'sct_check_atlas_integrity',
    'sct_check_dependencies',
    'sct_compute_ernst_angle',
    'sct_compute_hausdorff_distance',
    'sct_compute_mscc',
    'sct_compute_mtr',
    'sct_compute_snr',
    'sct_concat_transfo',
    'sct_convert',
    'sct_create_mask',
    'sct_crop_image',
    'sct_dice_coefficient',
    'sct_dmri_compute_dti',
    'sct_dmri_concat_bvals',
    'sct_dmri_concat_bvecs',
    'sct_dmri_create_noisemask',
    'sct_dmri_get_bvalue',
    'sct_dmri_moco',
    'sct_dmri_separate_b0_and_dwi',
    'sct_dmri_transpose_bvecs',
    'sct_download_data',
    'sct_extract_metric',
    'sct_flatten_sagittal',
    'sct_fmri_compute_tsnr',
    'sct_fmri_moco',
    'sct_image',
    'sct_label_utils',
    'sct_label_vertebrae',
    'sct_maths',
    'sct_process_segmentation',
    'sct_propseg',
    'sct_register_graymatter',
    'sct_register_multimodal',
    'sct_register_to_template',
    'sct_resample',
    'sct_segment_graymatter',
    'sct_smooth_spinalcord',
    'sct_straighten_spinalcord',
    'sct_testing',
    'sct_viewer',
    'sct_warp_template',
    'test_regist',
]
console_scripts = ['%s=scripts.%s:main' % (x, x) for x in commands]

version = "3.0.0"
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, './install/requirements/requirementsPip.txt')) as f:
    requirements = f.read().splitlines()

EXCLUDE_PACKAGES = []

setup(
    name='spinalcordtoolbox',
    version=version,
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
    packages=find_packages(exclude=EXCLUDE_PACKAGES),
    install_requires=requirements,

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={
    #    'bin': ['bin/isct_*'],
    # },
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
    # entry_points={'console_scripts': console_scripts, },
)
