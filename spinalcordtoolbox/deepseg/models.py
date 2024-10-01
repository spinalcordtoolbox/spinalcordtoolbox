"""
Deals with models for deepseg module. Available models are listed under MODELS.
Important: model names (onnx or pt files) should have the same name as the enclosing folder.

Copyright (c) 2020 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""


import os
import json
import logging
import textwrap
import shutil
import glob
from pathlib import Path

from spinalcordtoolbox import download
from spinalcordtoolbox.utils.sys import stylize, __deepseg_dir__


logger = logging.getLogger(__name__)

# List of models. The convention for model names is: (species)_(university)_(contrast)_region
# Regions could be: sc, gm, lesion, tumor
# NB: The 'url' field should either be:
#     1) A <mirror URL list> containing different mirror URLs for the model
#     2) A dict of <mirror URL lists>, where each list corresponds to a different seed (for model ensembling), and
#        each dictionary key corresponds to the seed's name (seed names are used to create subfolders per-seed)
MODELS = {
    "t2star_sc": {
        "url": [
            "https://github.com/ivadomed/t2star_sc/releases/download/r20231004/r20231004_t2star_sc.zip",
            "https://osf.io/8nk5w/download",
        ],
        "description": "Cord segmentation model on T2*-weighted contrast.",
        "contrasts": ["t2star"],
        "default": True,
    },
    "mice_uqueensland_sc": {
        "url": [
            "https://github.com/ivadomed/mice_uqueensland_sc/releases/download/r20200622/r20200622_mice_uqueensland_sc.zip",
            "https://osf.io/nu3ma/download?version=6",
        ],
        "description": "Cord segmentation model on mouse MRI. Data from University of Queensland.",
        "contrasts": ["t1"],
        "default": False,
    },
    "mice_uqueensland_gm": {
        "url": [
            "https://github.com/ivadomed/mice_uqueensland_gm/releases/download/r20200622/r20200622_mice_uqueensland_gm.zip",
            "https://osf.io/mfxwg/download?version=6",
        ],
        "description": "Gray matter segmentation model on mouse MRI. Data from University of Queensland.",
        "contrasts": ["t1"],
        "default": False,
    },
    "t2_tumor": {
        "url": [
            "https://github.com/ivadomed/t2_tumor/archive/r20201215.zip"
        ],
        "description": "Cord tumor segmentation model, trained on T2-weighted contrast.",
        "contrasts": ["t2"],
        "default": False,
    },
    "findcord_tumor": {
        "url": [
            "https://github.com/ivadomed/findcord_tumor/archive/r20201215.zip"
        ],
        "description": "Cord localisation model, trained on T2-weighted images with tumor.",
        "contrasts": ["t2"],
        "default": False,
    },
    "model_seg_sctumor-edema-cavity_t2-t1_unet3d-multichannel": {
        "url": [
            "https://github.com/ivadomed/model_seg_sctumor-edema-cavity_t2-t1_unet3d-multichannel/archive/r20201215.zip"
        ],
        "description": "Multiclass cord tumor segmentation model.",
        "contrasts": ["t2", "t1"],
        "default": False,
    },
    "model_seg_gm_wm_exvivo_t2_unet2d-multichannel-softseg": {
        "url": [
            "https://github.com/ivadomed/model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg/archive/r20210401_v2.zip"
        ],
        "description": "Grey/white matter seg on exvivo human T2w.",
        "contrasts": ["t2"],
        "default": False,
    },
    "model_seg_sc_MS_mp2rage": {
        "url": [
            "https://github.com/ivadomed/model_seg_ms_mp2rage/releases/download/r20211223/model_seg_ms_sc_mp2rage.zip"
        ],
        "description": "Segmentation of spinal cord on MP2RAGE data from MS participants.",
        "contrasts": ["mp2rage"],
        "default": False,
    },
    "model_7t_multiclass_gm_sc_unet2d": {
        "url": [
            "https://github.com/ivadomed/model_seg_gm-wm_t2star_7t_unet3d-multiclass/archive/refs/tags/r20211012.zip"
        ],
        "description": "SC/GM multiclass segmentation on T2*-w contrast at 7T. The model was created by N.J. Laines Medina, "
                       "V. Callot and A. Le Troter at CRMBM-CEMEREM Aix-Marseille University, France",
        "contrasts": ["t2star"],
        "default": False,
    },
    "model_seg_epfl_t2w_lumbar_sc": {
        "url": [
            "https://github.com/ivadomed/lumbar_seg_EPFL/releases/download/r20231004/model_seg_epfl_t2w_lumbar_sc.zip"
        ],
        "description": "Lumbar SC segmentation on T2w contrast with 3D UNet",
        "contrasts": ["t2"],
        "default": False,
    },
    # NB: Handling image binarization threshold for ivadomed vs. non-ivadomed models:
    #   - ivadomed models (above):
    #       - Threshold value is stored in the ivadomed-specific `.json` sidecar file
    #       - Binarization is applied within the ivadomed package
    #   - non-ivadomed models (below)
    #       - Models do not have a `.json` sidecar file, since they were not developed with ivadomed
    #       - So, threshold value is stored here, within the model dict
    #       - Binarization is applied within SCT code
    "model_seg_spinalcord_softseg_monai": {
        "url": [
            "https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/releases/download/v2.5/model_contrast-agnostic_20240930-1002.zip"
        ],
        "description": "Spinal cord segmentation agnostic to MRI contrasts using MONAI-based nnUNet model",
        "contrasts": ["any"],
        "thr": 0.5,  # Softseg model -> threshold at 0.5
        "default": False,
    },
    "model_seg_SCI_multiclass_lesion_sc_nnunet": {
        "url": [
            "https://github.com/ivadomed/model_seg_sci/releases/download/r20240729/model_SCIsegV2_r20240729.zip"
        ],
        "description": "Intramedullary SCI lesion and cord segmentation in T2w MRI",
        "contrasts": ["t2"],
        "thr": None,  # Images are already binarized when splitting into sc-seg + lesion-seg
        "default": False,
    },
    "model_seg_spinal_rootlets_nnunet": {
        "url": [
            "https://github.com/ivadomed/model-spinal-rootlets/releases/download/r20240730/model-spinal-rootlets_M5_r20240129-2.zip"
        ],
        "description": "Segmentation of spinal nerve rootlets for T2w images using nnUNet",
        "contrasts": ["t2"],
        "thr": None,  # Multiclass rootlets model (1.0, 2.0, 3.0...) -> no thresholding
        "default": False,
    },
    "model_seg_gm_wm_mouse_nnunet": {
         "url": [
             "https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/releases/download/v0.4/model.zip"
         ],
         "description": "White and grey matter segmentation on T1-weighted exvivo mouse spinal cord using nnUNet",
         "contrasts": ["t1"],
         "thr": None,  # Images are already binarized when splitting into gm-seg and wm-seg
         "default": False,
     },
    "model_seg_lesion_sc_canproco_nnunet": {
         "url": [
             "https://github.com/ivadomed/canproco/releases/download/r20240125/model_ms_seg_sc-lesion_regionBased_2d.zip"
         ],
         "description": "MS lesion/SC seg for STIR/PSIR contrasts",
         "contrasts": ["stir", "psir"],
         "thr": None,  # Images are already binarized when splitting into spinal cord and lesion
         "default": False,
    },
    "model_seg_sc_epi_nnunet": {
         "url": [
             "https://github.com/sct-pipeline/fmri-segmentation/releases/download/v0.2/model-fmri-segmentation-v0.2_nnUNetTrainer.zip"
         ],
         "description": "Spinal cord segmentation for EPI data (single 3D volume)",
         "contrasts": ["bold"],
         "thr": None,  # Images are already binarized
         "default": False,
     },
    "model_seg_lesion_MS_mp2rage": {
         "url": [
             "https://github.com/ivadomed/model_seg_ms_mp2rage/releases/download/r20240610/nnUNetTrainer_seg_ms_lesion_mp2rage__nnUNetPlans__3d_fullres.zip"
         ],
         "description": "Segmentation of spinal cord MS lesions on MP2RAGE UNIT1 contrast",
         "contrasts": ["UNIT1"],
         "thr": None,  # Images are already binarized
         "default": False,
     },
}


# List of task. The convention for task names is: action_(animal)_region_(contrast)
# Regions could be: sc, gm, lesion, tumor
TASKS = {
    'sc_t2star':
        {'description': 'Cord segmentation on T2*-weighted contrast',
         'long_description': 'This segmentation model for T2*w spinal cords uses the UNet architecture, and was '
                             'created with the `ivadomed` package. A subset of a private dataset (sct_testing_large) '
                             'was used, and consists of 236 subjects across 9 different sessions. A total of 388 pairs '
                             'of T2* images were used (anatomical image + manual cord segmentation). The image '
                             'properties include various orientations (superior, inferior) and crops (C1-C3, C4-C7, '
                             'etc.). The dataset was comprised of both non-pathological (healthy) and pathological (MS '
                             'lesion) adult patients.',
         'url': 'https://github.com/ivadomed/t2star_sc',
         'models': ['t2star_sc'],
         'citation': None
         },
    'sc_mouse_t1':
        {'description': 'Cord segmentation on mouse MRI',
         'long_description': 'This segmentation model for T1w mouse spinal cord segmentation uses the UNet '
                             'architecture, and was created with the `ivadomed` package. Training data was provided by '
                             'Dr. A. Althobathi of the University of Queensland, with the files consisting of T1w '
                             'scans (along with manual spinal cord + gray matter segmentations) from 10 mice in total. '
                             'The dataset was comprised of both non-pathological (healthy) and pathological (diseased) '
                             'mice.',
         'url': 'https://github.com/ivadomed/mice_uqueensland_sc/',
         'models': ['mice_uqueensland_sc'],
         'citation': None,
         },
    'gm_mouse_t1':
        {'description': 'Gray matter segmentation on mouse MRI',
         'long_description': 'This segmentation model for T1w mouse spinal gray matter segmentation uses the UNet '
                             'architecture, and was created with the `ivadomed` package. Training data was provided by '
                             'Dr. A. Althobathi of the University of Queensland, with the files consisting of T1w '
                             'scans (along with manual spinal cord + gray matter segmentations) from 10 mice in total. '
                             'The dataset was comprised of both non-pathological (healthy) and pathological (diseased) '
                             'mice.',
         'url': 'https://github.com/ivadomed/mice_uqueensland_gm/',
         'models': ['mice_uqueensland_gm'],
         'citation': None,
         },
    'tumor_t2':
        {'description': 'Cord tumor segmentation on T2-weighted contrast',
         'long_description': 'This segmentation model for T2w spinal tumor segmentation uses the UNet '
                             'architecture, and was created with the `ivadomed` package. Training data consisted of '
                             '380 pathological subjects in total: 120 with tumors of type Astrocytoma, 129 with '
                             'Ependymoma, and 131 with Hemangioblastoma. This model is used in tandem with another '
                             'model for specialized cord localisation of spinal cords with tumors '
                             '(https://github.com/ivadomed/findcord_tumor).',
         'url': 'https://github.com/sct-pipeline/tumor-segmentation',
         'models': ['findcord_tumor', 't2_tumor'],
         'citation': None,
         },
    'sc_MS_mp2rage':
        {'description': 'Cord segmentation on MP2RAGE in MS patients',
         'long_description': 'This segmentation model for MP2RAGE spinal cord segmentation uses a Modified3DUNet '
                             'architecture, and was created with the `ivadomed` package. Training data consisted of '
                             'scans from 30 multiple sclerosis (MS) patients, and the dataset included manual '
                             'segmentations of MS lesions. This dataset was provided by the University of Basel.',
         'url': 'https://github.com/ivadomed/model_seg_ms_mp2rage',
         'models': ['model_seg_sc_MS_mp2rage'],
         'citation': None,
         },
    'tumor_edema_cavity_t1_t2':
        {'description': 'Multiclass cord tumor/edema/cavity segmentation',
         'long_description': 'This segmentation model for T1w and T2w spinal tumor, edema, and cavity segmentation '
                             'uses a 3D UNet architecture, and was created with the `ivadomed` package. Training data '
                             'consisted of a subset of the dataset used for the model `seg_tumor_t2`, with 243 '
                             'subjects in total: 49 with tumors of type Astrocytoma, 83 with Ependymoma, and 111 with '
                             'Hemangioblastoma. For each subject, the requisite parts of the affected region (tumor, '
                             'edema, cavity) were segmented individually for training purposes. This model is used in '
                             'tandem with another model for specialized cord localisation of spinal cords with tumors '
                             '(https://github.com/ivadomed/findcord_tumor).',
         'url': 'https://github.com/ivadomed/model_seg_sctumor-edema-cavity_t2-t1_unet3d-multichannel',
         'models': ['findcord_tumor', 'model_seg_sctumor-edema-cavity_t2-t1_unet3d-multichannel'],
         'citation': textwrap.dedent("""
             ```
             @article{LEMAY2021102766,
                      title={Automatic multiclass intramedullary spinal cord tumor segmentation on MRI with deep learning},
                      journal={NeuroImage: Clinical},
                      volume={31},
                      pages={102766},
                      year-2021},
                      issn-2213-1582},
                      doi-https://doi.org/10.1016/j.nicl.2021.102766},
                      url-https://www.sciencedirect.com/science/article/pii/S2213158221002102},
                      author-Andreanne Lemay and Charley Gros and Zhizheng Zhuo and Jie Zhang and Yunyun Duan and Julien Cohen-Adad and Yaou Liu},
                      keywords-Deep learning, Automatic segmentation, Spinal cord tumor, MRI, Multiclass, CNN}
             }
             ```
         """),
         },
    'gm_wm_exvivo_t2':
        {'description': 'Grey/white matter seg on exvivo human T2w',
         'long_description': 'This segmentation model for T2w human spinal gray and white matter uses a 2D Unet '
                             'architecture, and was created with the `ivadomed` package. Training data consisted '
                             'of 149 2D slices taken from the scan of an ex vivo spinal cord of a single subject, with '
                             'the gray and white matter from each slice manually segmented for training purposes. The '
                             'data was provided by the University of Queenland, and through this collaboration, the '
                             'model was subsequently applied to the development of an ex vivo spinal cord template '
                             '(https://archive.ismrm.org/2020/1171.html).',
         'url': 'https://github.com/ivadomed/model_seg_exvivo_gm-wm_t2_unet2d-multichannel-softseg',
         'models': ['model_seg_gm_wm_exvivo_t2_unet2d-multichannel-softseg'],
         'citation': None,
         },
    'gm_sc_7t_t2star':
        {'description': 'SC/GM seg on T2*-weighted contrast at 7T',
         'long_description': 'This multiclass model (SC/GM) was developed by N.J. Laines Medina, V. Callot and A. Le '
                             'Troter at the Center for Magnetic Resonance in Biology and Medicine (CRMBM-CEMEREM, UMR '
                             '7339, CNRS, Aix-Marseille University, France). Training data consisted of T2*w scans '
                             'acquired at 7T from 72 subjects: 34 healthy controls, 25 patients with ALS, 13 patients '
                             'with MS. The model was validated by comparing with single-class models using 9-fold '
                             'Cross-Validation. It was enriched by integrating a hybrid data augmentation (composed of '
                             'classical geometric transformations, MRI artifacts, and real GM/WM contrasts distorted '
                             'with anatomically constrained deformation fields). Finally, it was tested with an '
                             'external multicentric database. For more information, see the following URL.',
         'url': 'https://github.com/ivadomed/model_seg_gm-wm_t2star_7t_unet3d-multiclass',
         'models': ['model_7t_multiclass_gm_sc_unet2d'],
         'citation': textwrap.dedent("""
             ```
             @misc{medina20212d,
                   title={2D Multi-Class Model for Gray and White Matter Segmentation of the Cervical Spinal Cord at 7T},
                   author={Nilser J. Laines Medina and Charley Gros and Julien Cohen-Adad and Virginie Callot and Arnaud Le Troter},
                   year={2021},
                   eprint={2110.06516},
                   archivePrefix={arXiv},
                   primaryClass={eess.IV}
             }
             ```
         """),
         },
    'sc_lumbar_t2':
        {'description': 'Lumbar cord segmentation with 3D UNet',
         'long_description': 'This segmentation model for T2w spinal cord segmentation uses a 3D UNet architecture, '
                             'and was created with the ivadomed package. Training data was provided by Nawal Kinany '
                             'and Dimitry Van De Ville of EPFL, with the files consisting of lumbar T2w scans (and '
                             'manual spinal cord segmentations) of 11 healthy (non-pathological) patients.',
         'url': 'https://github.com/ivadomed/lumbar_seg_EPFL',
         'models': ['model_seg_epfl_t2w_lumbar_sc'],
         'citation': None,
         },
    'spinalcord':
        {'description': 'Spinal cord segmentation agnostic to MRI contrasts',
         'long_description': 'This model for contrast agnostic spinal cord segmentation uses an nnUNet '
                             'architecture, and was created with the MONAI package. Training data consists of healthy '
                             'controls from the open-source Spine Generic Multi Subject database and pathologies from '
                             'private datasets including DCM and MS patients. Segmentation has been tested with the '
                             'following contrasts: [T1w, T2w, T2star, MTon_MTS, GRE_T1w, DWI, mp2rage_UNIT1], but '
                             'other contrasts that are close visual matches may also work well with this model.',
         'url': 'https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord/',
         'models': ['model_seg_spinalcord_softseg_monai'],
         'citation': textwrap.dedent("""
             ```
             @misc{bédard2024contrastagnosticsoftsegmentationspinal,
                   title={Towards contrast-agnostic soft segmentation of the spinal cord},
                   author={Sandrine Bédard and Enamundram Naga Karthik and Charidimos Tsagkas and Emanuele Pravatà and Cristina Granziera and Andrew Smith and Kenneth Arnold Weber II au2 and Julien Cohen-Adad},
                   year={2024},
                   eprint={2310.15402},
                   archivePrefix={arXiv},
                   primaryClass={eess.IV},
                   url={https://arxiv.org/abs/2310.15402},
             }
             ```
         """),  # noqa E501 (line too long)
         },
    'lesion_sc_SCI_t2':
        {'description': 'Intramedullary SCI lesion and cord segmentation in T2w MRI',
         'long_description': 'This segmentation model for spinal cord injury segmentation uses a 3D U-Net '
                             'architecture, and was trained with the nnUNetV2 framework. It is a multiclass model, '
                             'outputting segmentations for both the hyperintense SCI lesions and spinal cord. Training '
                             'data consisted of T2w images from 7 sites with traumatic (acute pre-operative, intermediate, '
                             'chronic), non-traumatic (DCM) and ischemic SCI lesions spanning numerous resolutions, '
                             'orientations, as well as multiple scanner manufacturers and field strengths.',
         'url': 'https://github.com/ivadomed/model_seg_sci',
         'models': ['model_seg_SCI_multiclass_lesion_sc_nnunet'],
         'citation': textwrap.dedent("""
             ```
             @misc{karthik2024scisegv2universaltoolsegmentation,
                   title={SCIsegV2: A Universal Tool for Segmentation of Intramedullary Lesions in Spinal Cord Injury},
                   author={Enamundram Naga Karthik and Jan Valošek and Lynn Farner and Dario Pfyffer and Simon Schading-Sassenhausen and Anna Lebret and Gergely David and Andrew C. Smith and Kenneth A. Weber II au2 and Maryam Seif and RHSCIR Network Imaging Group and Patrick Freund and Julien Cohen-Adad},
                   year={2024},
                   eprint={2407.17265},
                   archivePrefix={arXiv},
                   primaryClass={cs.CV},
                   url={https://arxiv.org/abs/2407.17265},
             }
             ```
         """),  # noqa E501 (line too long)
         },
    'rootlets_t2':
        {'description': 'Segmentation of spinal nerve rootlets for T2w contrast',
         'long_description': 'This segmentation model for spinal nerve rootlets segmentation uses a 3D U-Net '
                             'architecture, and was trained with the nnUNetV2 framework. It is a multiclass model, '
                             'outputting a single segmentation image containing 8 classes representing the C2-C8 '
                             'dorsal spinal cord nerve rootlets. Training data consisted of 31 isotropic T2w images '
                             'from healthy subjects from two different open-access datasets.',
         'url': 'https://github.com/ivadomed/model-spinal-rootlets',
         'models': ['model_seg_spinal_rootlets_nnunet'],
         'citation': None,
         },
    'gm_wm_mouse_t1':
        {'description': 'Exvivo mouse GM/WM segmentation for T1w contrast',
         'long_description': 'This segmentation model for gray and white matter segmentation of exvivo mice spinal '
                             'cords uses an NNunet architecture, and was created with the nnUNetV2 package. It is a '
                             'multiclass model, outputting segmentations for both the grey matter and white matter.'
                             'Training data consisted of 22 mice with different numbers of chunks, for a total of 72 '
                             'MRI 3D images. Each training image was T2-weighted, had a size of 200x200x500, and had '
                             'a resolution of 0.05mm isotropic. Training data was provided by the Balgrist Center at'
                             'the University of Zurich.',
         'url': 'https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1',
         'models': ['model_seg_gm_wm_mouse_nnunet'],
         'citation': textwrap.dedent("""
             ```
             @software{cohen_adad_2024_10819207,
                       author={Cohen-Adad, Julien},
                       title={{Segmentation model of ex vivo mouse spinal cord white and gray matter}},
                       month=mar,
                       year=2024,
                       publisher={Zenodo},
                       version={v0.5},
                       doi={10.5281/zenodo.10819207},
                       url={https://doi.org/10.5281/zenodo.10819207}
             }
             ```
         """),
         },
    'lesion_sc_MS_stir_psir':
        {'description': 'Segmentation of spinal cord and MS lesions for STIR and PSIR contrasts',
         'long_description': 'This segmentation model for MS lesion segmentation uses a 2D U-Net architecture, and was '
                             'trained with the nnUNetV2 framework. It is a region-based model, outputting a single '
                             'segmentation image containing 2 classes representing the spinal cord and MS lesions. '
                             'Training data consisted of sagittal PSIR 0.7×0.7×3 mm3 (4 sites, 333 participants) multiplied '
                             'by -1 and sagittal STIR 0.7×0.7×3 mm3 (1 site, 92 participants) images of the cervical SC from '
                             'the Canadian Prospective Cohort Study (CanProCo).',
         'url': 'https://github.com/ivadomed/canproco',
         'models': ['model_seg_lesion_sc_canproco_nnunet'],
         'citation': None,
         },
    'sc_epi':
        {'description': 'Spinal cord segmentation for EPI-BOLD fMRI data',
         'long_description': 'This segmentation model for spinal cord on EPI data (single 3D volume) uses a 3D UNet model built from '
                             'the nnUNetv2 framework. The training data consists of 3D images (n=192) spanning numerous resolutions '
                             'from multiple sites like Max Planck Institute for Human Cognitive and Brain Sciences - Leipzig, '
                             'University of Geneva, Stanford University, Kings College London, Universitätsklinikum Hamburg. The '
                             'dataset has healthy control subjects. The model has been trained in a human-in-the-loop active learning fashion.',
         'url': 'https://github.com/sct-pipeline/fmri-segmentation',
         'models': ['model_seg_sc_epi_nnunet'],
         'citation': None,
         },
    'lesion_MS_mp2rage':
        {'description': 'MS lesion segmentation on cropped MP2RAGE data',
         'long_description': 'This segmentation model for multiple sclerosis lesion segmentation on cropped MP2RAGE-UNIT1 spinal cord data. '
                             'Uses a 3D U-Net, trained with the nnUNetV2 framework. It is a single-class model outputting binary MS lesions '
                             'segmentations. Training consisted of MP2RAGE data on UNIT1 contrast at 1.0 mm3 isotropic (322 subjects from '
                             '3 sites: National Institutes of Health, Bethesda, USA, University Hospital Basel and University of Basel, Basel, '
                             'Switzerland and Center for Magnetic Resonance in Biology and Medicine, CRMBM-CEMEREM, UMR 7339, CNRS,  '
                             'Aix-Marseille University, Marseille, France). '
                             'To crop the data you can first segment the spinal cord using the contrast agnostic model, This could be '
                             'done using: "sct_deepseg spinalcord -i IMAGE_UNIT1 -o IMAGE_UNIT1_sc", then crop the '
                             'IMAGE_UNIT1 image with 30 mm of dilation on axial orientation around the spinal cord. This could be done using: '
                             '"sct_crop_image -i IMAGE_UNIT1 -m IMAGE_seg -dilate 30x30x5" . Note that 30 is only for 1mm isotropic '
                             'resolution, for images with another resolution divide 30/your_axial_resolution.',
         'url': 'https://github.com/ivadomed/model_seg_ms_mp2rage',
         'models': ['model_seg_lesion_MS_mp2rage'],
         'citation': textwrap.dedent("""
             ```
             @article{10.1162/imag_a_00218,
                      author = {Valošek, Jan and Mathieu, Theo and Schlienger, Raphaëlle and Kowalczyk, Olivia S. and Cohen-Adad, Julien},
                      title = "{Automatic Segmentation of the Spinal Cord Nerve Rootlets}",
                      journal = {Imaging Neuroscience},
                      year = {2024},
                      month = {06},
                      issn = {2837-6056},
                      doi = {10.1162/imag_a_00218},
                      url = {https://doi.org/10.1162/imag_a_00218},
             }
             ```
         """),
         },
}


def get_required_contrasts(task):
    """
    Get required contrasts according to models in tasks.

    :return: list: List of required contrasts
    """
    contrasts_required = set()
    for model in TASKS[task]['models']:
        for contrast in MODELS[model]['contrasts']:
            contrasts_required.add(contrast)

    return list(contrasts_required)


def folder(name_model):
    """
    Return absolute path of deep learning models.

    :param name: str: Name of model.
    :return: str: Folder to model
    """
    return os.path.join(__deepseg_dir__, name_model)


def install_model(name_model, custom_url=None):
    """
    Download and install specified model under SCT installation dir.

    :param name: str: Name of model.
    :return: None
    """
    logger.info("\nINSTALLING MODEL: {}".format(name_model))
    url_field = MODELS[name_model]['url'] if not custom_url else [custom_url]  # [] -> mimic a list of mirror URLs
    # List of mirror URLs corresponding to a single model
    if isinstance(url_field, list):
        model_urls = url_field
        # Make sure to preserve the internal folder structure for nnUNet-based models (to allow re-use with 3D Slicer)
        urls_used = download.install_data(model_urls, folder(name_model), dirs_to_preserve=("nnUNetTrainer",))
    # Dict of lists, with each list corresponding to a different model seed for ensembling
    else:
        if not isinstance(url_field, dict):
            raise ValueError("Invalid url field in MODELS")
        urls_used = {}
        for seed_name, model_urls in url_field.items():
            logger.info(f"\nInstalling '{seed_name}'...")
            urls_used[seed_name] = download.install_data(model_urls,
                                                         folder(os.path.join(name_model, seed_name)), keep=True)
    # Write `source.json` (for model provenance / updating)
    source_dict = {
        'model_name': name_model,
        'model_urls': urls_used,
        # NB: If a custom URL is used, then it would just get overwritten as "out of date" when running the task
        #     So, we add a flag to tell `sct_deepseg` *not* to reinstall the model if a custom URL was used.
        'custom': bool(custom_url)
    }
    with open(os.path.join(folder(name_model), "source.json"), "w") as fp:
        json.dump(source_dict, fp, indent=4)


def install_default_models():
    """
    Download all default models and install them under SCT installation dir.

    :return: None
    """
    for name_model, value in MODELS.items():
        if value['default']:
            install_model(name_model)


def is_up_to_date(path_model):
    """
    Determine whether an on-disk model is up-to-date by comparing
    the URL used to download the model with the latest mirrors.

    :return: bool: whether the model is up-to-date
    """
    source_path = os.path.join(path_model, "source.json")
    if not os.path.isfile(source_path):
        logger.warning("Provenance file 'source.json' missing!")
        return False  # NB: This will force a reinstall
    with open(source_path, "r") as fp:
        source_dict = json.load(fp)
    model_name = source_dict["model_name"]
    if model_name not in MODELS:
        logger.warning(f"Model name '{model_name}' from source.json does not match model names in SCT source code.")
        return False

    expected_model_urls = MODELS[model_name]['url'].copy()
    actual_model_urls = source_dict["model_urls"]

    if "custom" in source_dict and source_dict["custom"] is True:
        logger.warning(f"Using custom model from URL '{actual_model_urls}'.")
        return True  # Don't reinstall the model if the 'custom' flag is set (since custom URLs would fail comparison)

    # Single-seed models
    if isinstance(expected_model_urls, list) and isinstance(actual_model_urls, str):
        if actual_model_urls not in expected_model_urls:
            return False
    # Multi-seed, ensemble models
    elif isinstance(expected_model_urls, dict) and isinstance(actual_model_urls, dict):
        for seed, url in actual_model_urls.items():
            if seed not in expected_model_urls:
                logger.warning(f"unexpected seed: {seed}")
                return False
            if url not in expected_model_urls.pop(seed):
                logger.warning(f"wrong version for {seed}: {url}")
                return False
        if expected_model_urls:
            logger.warning(f"missing seeds: {list(expected_model_urls.keys())}")
            return False
    else:
        logger.warning("Mismatch between 'source.json' URL format and SCT source code URLs")
        return False
    logger.info(f"Model '{model_name}' is up to date (Source: {actual_model_urls})")
    return True


def is_valid(path_models):
    """
    Check if model paths have the necessary files and follow naming conventions:
    - Folder should have the same name as the enclosed files.

    :param path_models: str or list: Absolute path(s) to folder(s) that enclose the model files.
    """
    def _is_valid(path_model):
        return has_ivadomed_files(path_model) or has_ckpt_files(path_model) or has_pth_files(path_model)
    # Adapt the function so that it can be used on single paths (str) or lists of paths
    if not isinstance(path_models, list):
        path_models = [path_models]
    return all(_is_valid(path) for path in path_models)


def has_ivadomed_files(path_model):
    """
    Check if model path contains A) a named .pt/.onnx model file and B) a named ivadomed json configuration file
    """
    name_model = Path(path_model).name
    path_pt = os.path.join(path_model, name_model + '.pt')
    path_onnx = os.path.join(path_model, name_model + '.onnx')
    path_json = os.path.join(path_model, name_model + '.json')
    return (os.path.exists(path_pt) or os.path.exists(path_onnx)) and os.path.exists(path_json)


def has_ckpt_files(path_model):
    """
    Check if model path contains any checkpoint files (used by non-ivadomed MONAI models)
    """
    return bool(glob.glob(os.path.join(path_model, '**', '*.ckpt'), recursive=True))


def has_pth_files(path_model):
    """
    Check if model path contains any serialized PyTorch state dictionary files (used by non-ivadomed nnUNet models)
    """
    return bool(glob.glob(os.path.join(path_model, '**', '*.pth'), recursive=True))


def check_model_software_type(path_model):
    """
    Determine the software used to train the model based on the types of files in the model folder
    """
    if has_ivadomed_files(path_model):
        return 'ivadomed'
    elif has_ckpt_files(path_model):
        return 'monai'
    elif has_pth_files(path_model):
        return 'nnunet'
    else:
        raise ValueError("Model type cannot be determined.")


def list_tasks():
    """
    Display available tasks with description.
    :return: dict: Tasks that are installed
    """
    return {name: value for name, value in TASKS.items()}


def list_tasks_string():
    tasks = list_tasks()
    # Display coloured output
    color = {True: 'LightGreen', False: 'LightRed'}
    table = f"{'TASK':<30s}{'DESCRIPTION':<50s}\n"
    table += f"{'-' * 80}\n"
    for name_task, value in tasks.items():
        path_models = [folder(name_model) for name_model in value['models']]
        path_models = [find_model_folder_paths(path) for path in path_models]
        are_models_valid = [is_valid(path_model) for path_model in path_models]
        task_status = stylize(name_task.ljust(30),
                              color[all(are_models_valid)])
        description_status = stylize(value['description'].ljust(50),
                                     color[all(are_models_valid)])

        table += "{}{}".format(task_status, description_status) + "\n"

    table += '\nLegend: {} | {}\n\n'.format(
            stylize("installed", color[True]),
            stylize("not installed", color[False]))

    table += 'To read in-depth descriptions of the training data, model architecture, '
    table += 'etc. used for these tasks, type the following command:\n\n'
    table += '    {}'.format(stylize('sct_deepseg -list-tasks', ['LightBlue', 'Bold']))
    return table


def display_list_tasks():
    for name_task, value in list_tasks().items():
        indent_len = len("LONG_DESCRIPTION: ")
        print("{}{}".format("TASK:".ljust(indent_len), stylize(name_task, 'Bold')))

        input_contrasts = str(', '.join(model_name for model_name in
                                        get_required_contrasts(name_task))).ljust(15)
        print("{}{}".format("CONTRAST:".ljust(indent_len), input_contrasts))

        path_models = [folder(name_model) for name_model in value['models']]
        path_models = [find_model_folder_paths(path) for path in path_models]
        are_models_valid = [is_valid(path_model) for path_model in path_models]
        models_status = ', '.join([model_name
                                   for model_name, validity in zip(value['models'], are_models_valid)])
        print("{}{}".format("MODELS:".ljust(indent_len), models_status))
        print('\n'.join(textwrap.wrap(value['long_description'],
                        width=shutil.get_terminal_size()[0]-1,
                        initial_indent="LONG_DESCRIPTION: ",
                        subsequent_indent=' '*indent_len)))

        print("{}{}".format("URL:".ljust(indent_len), stylize(value['url'], 'Cyan')))
        if all(are_models_valid):
            installed = stylize("Yes", 'LightGreen')
        else:
            installed = stylize("No", 'LightRed')
        print("{}{}\n".format("INSTALLED:".ljust(indent_len), installed))
    exit(0)


def get_metadata(folder_model):
    """
    Get metadata from json file located in folder_model

    :param path_model: str: Model folder
    :return: dict
    """
    fname_metadata = os.path.join(folder_model, os.path.basename(folder_model) + '.json')
    with open(fname_metadata, "r") as fhandle:
        metadata = json.load(fhandle)
    return metadata


def find_model_folder_paths(path_model):
    """
    Search for the presence of model subfolders within the main model folder. If they exist,
    then the model folder is actually an ensemble of models, so return a list of folders.
    If they don't exist, then return the original `path_model` (but as a list, to ensure code compatibility).

    :param path_model: Absolute path to folder that encloses the model files.
    :return: list: Either a list of ensemble subfolders, or a list containing the original model folder path.
    """
    name_model = path_model.rstrip(os.sep).split(os.sep)[-1]
    # Check to see if model folder contains subfolders with the model name (i.e. ensembling)
    model_subfolders = [folder[0] for folder in os.walk(path_model)  # NB: `[0]` == folder name for os.walk
                        if folder[0].endswith(name_model) and folder[0] != path_model]
    # If it does, then these are the "true" model subfolders. Otherwise, return the original path as a list.
    return model_subfolders if model_subfolders else [path_model]
