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
        '{}=spinalcordtoolbox.compat.launcher:main'.format(x) for x in \
        [
            'isct_check_detection',
            'isct_convert_binary_to_trilinear',
            'isct_minc2volume-viewer',
            'isct_test_ants',
            'isct_warpmovie_generator',
            'sct_analyze_lesion',
            'sct_analyze_texture',
            'sct_apply_transfo',
            'sct_check_dependencies',
            'sct_compute_ernst_angle',
            'sct_compute_hausdorff_distance',
            'sct_compute_mscc',
            'sct_compute_mtr',
            'sct_compute_mtsat',
            'sct_compute_snr',
            'sct_concat_transfo',
            'sct_convert',
            'sct_create_mask',
            'sct_crop_image',
            'sct_deepseg',
            'sct_deepseg_gm',
            'sct_deepseg_lesion',
            'sct_deepseg_sc',
            'sct_denoising_onlm',
            'sct_detect_pmj',
            'sct_dice_coefficient',
            'sct_dmri_compute_bvalue',
            'sct_dmri_compute_dti',
            'sct_dmri_concat_b0_and_dwi',
            'sct_dmri_concat_bvals',
            'sct_dmri_concat_bvecs',
            'sct_dmri_display_bvecs',
            'sct_dmri_moco',
            'sct_dmri_separate_b0_and_dwi',
            'sct_dmri_transpose_bvecs',
            'sct_download_data',
            'sct_extract_metric',
            'sct_flatten_sagittal',
            'sct_fmri_compute_tsnr',
            'sct_fmri_moco',
            'sct_get_centerline',
            'sct_image',
            'sct_label_utils',
            'sct_label_vertebrae',
            'sct_maths',
            'sct_merge_images',
            'sct_process_segmentation',
            'sct_run_batch',
            'sct_propseg',
            'sct_qc',
            'sct_register_multimodal',
            'sct_register_to_template',
            'sct_resample',
            'sct_smooth_spinalcord',
            'sct_straighten_spinalcord',
            'sct_testing',
            'sct_version',
            'sct_warp_template',
        ]
        ],
    ),
)
