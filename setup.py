# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

path_version = path.join(here, 'spinalcordtoolbox', 'version.txt')
with open(path_version) as f:
    version = f.read().strip()


# This function does not handle all corner cases and
#  assumes a simple requirements.txt is fed to it
def get_dependencies(requirements_path=None):
    if not path.exists(requirements_path):
        return []

    with open(requirements_path) as f:
        requirements = f.read().splitlines()

    return requirements


DEFAULT_REQUIREMENTS = [
    'colored',
    'dipy',
    # h5py is pinned to minor than 3 due to issues with Keras/TF
    # https://github.com/tensorflow/tensorflow/issues/44467
    'h5py~=2.10.0',
    'Keras==2.3.1',
    'ivadomed',
    'matplotlib',
    'nibabel',
    'numpy',
    # 1.7.0>onnxruntime>=1.5.1 required `brew install libomp` on macOS.
    # So, pin to >=1.7.0 to avoid having to ask users to install libomp.
    'onnxruntime>=1.7.0',
    'pandas',
    'psutil',
    'pyqt5==5.11.3',
    'pytest',
    'pytest-cov',
    'raven',
    'requests',
    'requirements-parser',
    'scipy',
    'scikit-image',
    'scikit-learn',
    'tensorflow~=1.15.0',
    # PyTorch's Linux/Windows distribution is very large due to its GPU support,
    # but we only need that for training models. For users, use the CPU-only version
    # (only available directly from the PyTorch project).
    # The macOS version has never had GPU support, so doesn't need the workaround.
    'torch==1.5.0+cpu; sys_platform != "darwin"',
    'torch==1.5.0; sys_platform == "darwin"',
    'torchvision==0.6.0+cpu; sys_platform != "darwin"',
    'torchvision==0.6.0; sys_platform == "darwin"',
    'xlwt',
    'tqdm',
    'transforms3d',
    'urllib3[secure]',
    'pytest_console_scripts',
    'wquantiles',
]

frozen_dependencies = get_dependencies(path.join(here, 'requirements-freeze.txt'))

dependencies = frozen_dependencies or DEFAULT_REQUIREMENTS

with open(path.join(here, "install_requirements.txt"), 'wt') as f:
    f.write('\n'.join(dependencies))


setup(
    name='spinalcordtoolbox',
    version=version,
    description='Library of analysis tools for MRI of the spinal cord',
    long_description=long_description,
    url='https://www.neuro.polymtl.ca/',
    author='NeuroPoly Lab, Polytechnique Montreal',
    author_email='neuropoly@googlegroups.com',
    license='LGPLv3',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='Magnetic Resonance Imaging MRI spinal cord analysis template',
    packages=find_packages(exclude=['.git', 'data', 'dev', 'dev.*',
                                    'install', 'testing']),
    include_package_data=True,
    python_requires="==3.7.*",
    install_requires=dependencies,
    extras_require={
        'docs': [
            'sphinxcontrib-programoutput',
            'sphinx_rtd_theme',
            'sphinx-copybutton',
            'furo==2021.11.23',
            'recommonmark',
            'sphinx==4.1.2'
        ],
    },
    entry_points=dict(
        console_scripts=[
            '{}=spinalcordtoolbox.compat.launcher:main'.format(x) for x in
            [
                'isct_convert_binary_to_trilinear',
                'isct_minc2volume-viewer',
                'isct_test_ants',
                'sct_analyze_lesion',
                'sct_analyze_texture',
                'sct_apply_transfo',
                'sct_check_dependencies',
                'sct_concat_transfo',
                'sct_compute_ernst_angle',
                'sct_compute_hausdorff_distance',
                'sct_compute_mscc',
                'sct_compute_mtr',
                'sct_compute_mtsat',
                'sct_compute_snr',
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
