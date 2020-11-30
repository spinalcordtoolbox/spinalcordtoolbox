
.. _command-line-tools:

Command-Line Tools
##################

.. contents::
   :local:
   :depth: 1
..


Summary of Tools
****************

The command-line tools are named ``sct_*``; to see all the commands
available from SCT, start a new Terminal and type ``sct`` then press
"tab".


Segmentation
============

- sct_create_mask_ - Create mask along z direction.
- sct_deepseg_ - Segment an anatomical structure or pathologies according using a deep learning model created with
  `ivadomed <http://ivadomed.org/>`_.
- sct_deepseg_gm_ - Segment spinal cord gray matter using deep learning.
- sct_deepseg_lesion_ - Segment multiple sclerosis lesions.
- sct_deepseg_sc_ - Segment spinal cord using deep learning.
- sct_get_centerline_ - Extracts the spinal cord centerline.
- sct_propseg_ - Segment spinal cord using propagation of deformation model (PropSeg).

Segmentation analysis
=====================

- sct_analyze_lesion_ - Compute statistics on segmented lesions.
- sct_compute_hausdorff_distance_ - Compute the Hausdorff's distance between two binary images.
- sct_compute_mscc_ - Compute Maximum Spinal Cord Compression (MSCC).
- sct_dice_coefficient_ - Compute the Dice Coefficient to estimate overlap between two binary images.
- sct_process_segmentation_ - Perform various types of processing from the spinal cord segmentation.

Labeling
========

- sct_detect_pmj_ - Detection of the Ponto-Medullary Junction (PMJ).
- sct_label_vertebrae_ - Label vertebral levels
- sct_label_utils_ - Collection of tools to create or process labels

Registration
============

- sct_apply_transfo_ - Apply transformations.
- sct_get_centerline_ - Reconstruct spinal cord centerline.
- sct_register_multimodal_ - Register two images together (non-linear, constrained in axial plane)
- sct_register_to_template_ - Register an image with an anatomical template (eg. the `PAM50 template
  <https://pubmed.ncbi.nlm.nih.gov/29061527/>`_).
- sct_straighten_spinalcord_ - Straighten spinal cord from centerline
- sct_warp_template_ - Warps the template and all atlases to a destination image.

Diffusion MRI
=============

- sct_dmri_compute_bvalue_ - Calculate b-value (in mm^2/s).
- sct_dmri_concat_bvals_ - Concatenate bval files in time.
- sct_dmri_concat_bvecs_ - Concatenate bvec files in time.
- sct_dmri_compute_dti_ - Compute Diffusion Tensor Images (DTI) using `dipy <https://dipy.org/>`_.
- sct_dmri_display_bvecs_ - Display scatter plot of gradient directions from bvecs file.
- sct_dmri_moco_ - Slice-wise motion correction of DWI data.
- sct_dmri_separate_b0_and_dwi_ - Separate b=0 and DW images from diffusion dataset.
- sct_dmri_transpose_bvecs_ - Transpose bvecs file.

Magnetization transfer
======================

- sct_compute_mtr_ - Compute magnetization transfer ratio (MTR).
- sct_compute_mtsat_ - Compute MTsat and T1map `[Helms et al. Magn Reson Med 2008]
  <https://pubmed.ncbi.nlm.nih.gov/19025906/>`_.

Functional MRI
==============

- sct_fmri_compute_tsnr_ - Compute the temporal signal-to-noise ratio from fMRI nifti files.
- sct_fmri_moco_ - Correct fMRI data for motion.

Metric processing
=================

- sct_analyze_texture_ - Extraction of grey level co-occurence matrix (GLCM) texture features from an image within a
  given mask.
- sct_extract_metric_ - Estimate metric value within tracts, taking into account partial volume effect.

Image manipulation
==================

- sct_convert_ - Convert image file to another type.
- sct_crop_image_ - Tools to crop an image, either via command line or via a Graphical User Interface (GUI).
- sct_denoising_onlm_ - Utility function to denoise images.
- sct_flatten_sagittal_ - Flatten the spinal cord in the sagittal plane (to make nice pictures).
- sct_image_ - Performs various operations on images (split, pad, etc.).
- sct_maths_ - Performs mathematical operations on images (threshold, smooth, etc.).
- sct_merge_images_ - Merge images to the same space.
- sct_resample_ - Anisotropic resampling of 3D or 4D data.
- sct_smooth_spinalcord_ - Smooth the spinal cord along its centerline.

Miscellaneous
=============

- sct_compute_ernst_angle_ - Compute Ernst angle.
- sct_compute_snr_ - Compute SNR using methods described in `[Dietrich et al. JMRI 2007]
  <https://pubmed.ncbi.nlm.nih.gov/17622966/>`_.
- sct_download_data_ - Download binaries from the web.
- sct_qc_ - Generate Quality Control (QC) report following SCT processing.
- sct_run_batch_ - Wrapper to processing scripts, which loops across subjects.

System tools
============

- sct_check_dependencies_ - Check installation and compatibility of SCT.
- sct_testing_ - Runs complete testing to make sure SCT is working properly.
- sct_version_ - Display SCT version.


Main Tools
**********


sct_analyze_lesion
==================

.. program-output:: sct_analyze_lesion -h


sct_analyze_texture
===================

.. program-output:: sct_analyze_texture -h


sct_apply_transfo
=================

.. program-output:: sct_apply_transfo -h


sct_compute_ernst_angle
=======================

.. program-output:: sct_compute_ernst_angle -h


sct_compute_hausdorff_distance
==============================

.. program-output:: sct_compute_hausdorff_distance -h


sct_compute_mscc
================

.. program-output:: sct_compute_mscc -h


sct_compute_mtr
===============

.. program-output:: sct_compute_mtr -h


sct_compute_mtsat
=================

.. program-output:: sct_compute_mtsat -h


sct_compute_snr
===============

.. program-output:: sct_compute_snr -h


sct_convert
==============

.. program-output:: sct_convert -h


sct_create_mask
===============

.. program-output:: sct_create_mask -h


sct_crop_image
==============

.. program-output:: sct_crop_image -h


sct_deepseg
===========

.. program-output:: sct_deepseg -h


sct_deepseg_gm
==============

.. program-output:: sct_deepseg_gm -h


sct_deepseg_lesion
==================

.. program-output:: sct_deepseg_lesion -h


sct_deepseg_sc
==============

.. program-output:: sct_deepseg_sc -h


sct_denoising_onlm
==================

.. program-output:: sct_denoising_onlm -h


sct_detect_pmj
==============

.. program-output:: sct_detect_pmj -h


sct_dice_coefficient
====================

.. program-output:: sct_dice_coefficient -h


sct_dmri_compute_bvalue
=======================

.. program-output:: sct_dmri_compute_bvalue -h


sct_dmri_compute_dti
====================

.. program-output:: sct_dmri_compute_dti -h


sct_dmri_concat_bvals
=====================

.. program-output:: sct_dmri_concat_bvals -h


sct_dmri_concat_bvecs
=====================

.. program-output:: sct_dmri_concat_bvecs -h


sct_dmri_display_bvecs
======================

.. program-output:: sct_dmri_display_bvecs -h


sct_dmri_moco
=============

.. program-output:: sct_dmri_moco -h


sct_dmri_separate_b0_and_dwi
============================

.. image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/command-line/sct_dmri_separate_b0_and_dwi_example.png

.. program-output:: sct_dmri_separate_b0_and_dwi -h


sct_dmri_transpose_bvecs
========================

.. program-output:: sct_dmri_transpose_bvecs -h


sct_download_data
=================

.. program-output:: sct_download_data -h


sct_extract_metric
==================

.. program-output:: sct_extract_metric -h


sct_flatten_sagittal
====================

.. program-output:: sct_flatten_sagittal -h


sct_fmri_compute_tsnr
=====================

.. program-output:: sct_fmri_compute_tsnr -h


sct_fmri_moco
=============

.. program-output:: sct_fmri_moco -h


sct_get_centerline
==================

.. program-output:: sct_get_centerline -h


sct_image
=========

.. program-output:: sct_image -h


sct_label_utils
===============

.. program-output:: sct_label_utils -h


sct_label_vertebrae
===================

.. program-output:: sct_label_vertebrae -h


sct_maths
=========

.. program-output:: sct_maths -h


sct_merge_images
================

.. program-output:: sct_merge_images -h


sct_process_segmentation
========================

.. program-output:: sct_process_segmentation -h


sct_propseg
===========

.. image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/command-line/sct_propseg_example.png

.. program-output:: sct_propseg -h


sct_qc
======

.. program-output:: sct_qc -h


sct_register_multimodal
=======================

.. program-output:: sct_register_multimodal -h


sct_register_to_template
========================

.. program-output:: sct_register_to_template -h


sct_resample
============

.. program-output:: sct_resample -h


sct_run_batch
=============

.. program-output:: sct_run_batch -h


sct_smooth_spinalcord
=====================

.. program-output:: sct_smooth_spinalcord -h


sct_straighten_spinalcord
=========================

.. program-output:: sct_straighten_spinalcord -h


sct_warp_template
=================

.. program-output:: sct_warp_template -h




System Commands
***************


sct_check_dependencies
======================

.. program-output:: sct_check_dependencies -h


sct_testing
===========

.. program-output:: sct_testing -h

sct_version
===========

.. program-output:: sct_version




Internal Commands
*****************

These scripts are tailored to the developers.


isct_convert_binary_to_trilinear
================================

.. program-output:: isct_convert_binary_to_trilinear -h


isct_minc2volume-viewer
=======================

.. program-output:: isct_minc2volume-viewer -h


isct_test_ants
==============

.. program-output:: isct_test_ants -h
