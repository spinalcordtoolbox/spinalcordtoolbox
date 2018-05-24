
.. _command-line-tools:

Command-Line Tools
##################

.. contents::
   :local:
   :depth: 1
..


Summary of Tools
****************

The command-line tools are named `sct_*`; to see all the commands
available from SCT, start a new Terminal and type `sct` then press
"tab".


Segmentation:

- sct_create_mask_ - Create mask along z direction.
- sct_label_vertebrae_ - Label vertebral levels
- sct_propseg_ - Segment spinal cord using propagation of deformation model (PropSeg)
- sct_process_segmentation_ - Perform various types of processing from the spinal cord segmentation.
- sct_segment_graymatter_ - Segment the white and gray matter on the cervical spinal cord (available in v2.1 and higher).

Registration:

- sct_apply_transfo_ - Apply transformations.
- sct_concat_transfo_ - Concatenate transformations.
- sct_get_centerline_ - Reconstruct spinal cord centerline.
- sct_register_multimodal_ - Register two images together (non-linear, constrained in axial plane)
- sct_register_to_template_ - Register the template to an anatomical image (t1, t2).
- sct_smooth_spinalcord_ - Smooth the spinal cord along its centerline.
- sct_straighten_spinalcord_ - Straighten spinal cord from centerline
- sct_warp_template_ - Warp all the spinal cord tracts of the atlas according to the warping field given as input.

Metric processing:

- sct_average_data_within_mask_ - Average data within a mask.
- sct_extract_metric_ - Estimate metric value within tracts, taking into account partial volume effect.

Diffusion MRI:

- sct_dmri_concat_bvals_ - Concatenate bval files in time.
- sct_dmri_concat_bvecs_ - Concatenate bvec files in time.
- sct_dmri_moco_ - Slice-wise motion correction of DWI data.
- sct_dmri_separate_b0_and_dwi_ - Separate b=0 and DW images from diffusion dataset.
- sct_dmri_transpose_bvecs_ - Transpose bvecs file.

Magnetization transfer:

- sct_compute_mtr_ - Register image without (MTC0) and with magnetization transfer contrast (MTC1) and compute MTR

Functional MRI:

- sct_fmri_compute_tsnr_ - Compute the temporal signal-to-noise ratio from fMRI nifti files.
- sct_fmri_moco_ - Correct fMRI data for motion.

Miscellaneous:

- sct_check_dependencies_ - Check installation and compatibility of SCT.
- sct_testing_ - Runs complete testing to make sure SCT is working properly.
- sct_compute_ernst_angle_ - Compute Ernst angle.
- sct_dice_coefficient_ - Compute 2D or 3D DICE coefficient between two binary images.
- sct_flatten_sagittal_ - Flatten the spinal cord in the sagittal plane (to make nice pictures).
- sct_flip_data_ - Flip data in a specified dimension (x,y,z or t). N.B. This script will NOT modify the header but the way the data are stored (so be careful!!)
- sct_label_utils_ - Utility function for label images.
- sct_image_ - Performs various operations on images (split, pad, etc.).
- sct_maths_ - Performs mathematical operations on images (threshold, smooth, etc.).
- sct_pipeline_ - Runs one sct tool on many subjects in one command.




Main Tools
**********


sct_analyze_lesion
=====================

.. program-output:: ../../../bin/sct_analyze_lesion -h


sct_analyze_texture
======================

.. program-output:: ../../../bin/sct_analyze_texture -h


sct_apply_transfo
====================

.. program-output:: ../../../bin/sct_apply_transfo -h


sct_average_data_within_mask
===============================

.. program-output:: ../../../bin/sct_average_data_within_mask -h


sct_change_image_type
========================

.. program-output:: ../../../bin/sct_change_image_type -h


sct_check_atlas_integrity
============================

.. program-output:: ../../../bin/sct_check_atlas_integrity -h



sct_compute_ernst_angle
==========================

.. program-output:: ../../../bin/sct_compute_ernst_angle -h


sct_compute_hausdorff_distance
=================================

.. program-output:: ../../../bin/sct_compute_hausdorff_distance -h


sct_compute_mscc
===================

.. program-output:: ../../../bin/sct_compute_mscc -h


sct_compute_mtr
==================

.. program-output:: ../../../bin/sct_compute_mtr -h


sct_compute_snr
==================

.. program-output:: ../../../bin/sct_compute_snr -h


sct_concat_transfo
=====================

.. program-output:: ../../../bin/sct_concat_transfo -h


sct_convert
==============

.. program-output:: ../../../bin/sct_convert -h


sct_create_mask
==================

.. program-output:: ../../../bin/sct_create_mask -h


sct_crop_image
=================

.. program-output:: ../../../bin/sct_crop_image -h


.. _sct_deepseg_gm:

sct_deepseg_gm
=================

.. program-output:: ../../../bin/sct_deepseg_gm -h


.. _sct_deepseg_sc:

sct_deepseg_sc
=================

.. program-output:: ../../../bin/sct_deepseg_sc -h


sct_denoising_onlm
=====================

.. program-output:: ../../../bin/sct_denoising_onlm -h


sct_detect_pmj
=================

.. program-output:: ../../../bin/sct_detect_pmj -h


sct_dice_coefficient
=======================

.. program-output:: ../../../bin/sct_dice_coefficient -h


sct_dmri_compute_bvalue
==========================

.. program-output:: ../../../bin/sct_dmri_compute_bvalue -h


sct_dmri_compute_dti
=======================

.. program-output:: ../../../bin/sct_dmri_compute_dti -h


sct_dmri_concat_bvals
========================

.. program-output:: ../../../bin/sct_dmri_concat_bvals -h


sct_dmri_concat_bvecs
========================

.. program-output:: ../../../bin/sct_dmri_concat_bvecs -h


sct_dmri_create_noisemask
============================

.. program-output:: ../../../bin/sct_dmri_create_noisemask -h


sct_dmri_display_bvecs
=========================

.. program-output:: ../../../bin/sct_dmri_display_bvecs -h


sct_dmri_eddy_correct
========================

.. program-output:: ../../../bin/sct_dmri_eddy_correct -h


sct_dmri_moco
================

.. program-output:: ../../../bin/sct_dmri_moco -h


sct_dmri_separate_b0_and_dwi
===============================

.. program-output:: ../../../bin/sct_dmri_separate_b0_and_dwi -h


sct_dmri_transpose_bvecs
===========================

.. program-output:: ../../../bin/sct_dmri_transpose_bvecs -h


sct_download_data
====================

.. program-output:: ../../../bin/sct_download_data -h


sct_extract_metric
=====================

.. program-output:: ../../../bin/sct_extract_metric -h


sct_flatten_sagittal
=======================

.. program-output:: ../../../bin/sct_flatten_sagittal -h


sct_fmri_compute_tsnr
========================

.. program-output:: ../../../bin/sct_fmri_compute_tsnr -h


sct_fmri_moco
================

.. program-output:: ../../../bin/sct_fmri_moco -h


sct_get_centerline
=====================

.. program-output:: ../../../bin/sct_get_centerline -h


sct_image
============

.. program-output:: ../../../bin/sct_image -h


sct_invert_image
===================

.. program-output:: ../../../bin/sct_invert_image -h


sct_label_utils
==================

.. program-output:: ../../../bin/sct_label_utils -h


sct_label_vertebrae
======================

.. program-output:: ../../../bin/sct_label_vertebrae -h


sct_maths
============

.. program-output:: ../../../bin/sct_maths -h


sct_merge_images
===================

.. program-output:: ../../../bin/sct_merge_images -h


sct_nifti_tool
=================

.. program-output:: ../../../bin/sct_nifti_tool -h


sct_pipeline
===============

.. program-output:: ../../../bin/sct_pipeline -h


sct_process_segmentation
===========================

.. program-output:: ../../../bin/sct_process_segmentation -h


.. _sct_propseg:

sct_propseg
==============

.. program-output:: ../../../bin/sct_propseg -h

Notes:

- https://sourceforge.net/p/spinalcordtoolbox/wiki/correction_PropSeg/

  .. TODO



sct_register_graymatter
==========================

.. program-output:: ../../../bin/sct_register_graymatter -h


sct_register_multimodal
==========================

.. program-output:: ../../../bin/sct_register_multimodal -h


sct_register_to_template
===========================

.. program-output:: ../../../bin/sct_register_to_template -h


sct_resample
===============

.. program-output:: ../../../bin/sct_resample -h


.. _sct_segment_graymatter:

sct_segment_graymatter
=========================

.. program-output:: ../../../bin/sct_segment_graymatter -h


sct_smooth_spinalcord
========================

.. program-output:: ../../../bin/sct_smooth_spinalcord -h


sct_straighten_spinalcord
============================

.. program-output:: ../../../bin/sct_straighten_spinalcord -h


sct_testing
==============

.. program-output:: ../../../bin/sct_testing -h


sct_utils
============

.. program-output:: ../../../bin/sct_utils -h


sct_viewer
=============

.. program-output:: ../../../bin/sct_viewer -h


sct_warp_template
====================

.. program-output:: ../../../bin/sct_warp_template -h


System Commands
***************


sct_check_dependencies
======================

.. program-output:: ../../../bin/sct_check_dependencies -h




Internal Commands
*****************



isct_check_detection
=======================

.. program-output:: ../../../scripts/isct_check_detection.py -h


isct_get_fractional_volume
=============================

.. program-output:: ../../../scripts/isct_get_fractional_volume.py -h


isct_minc2volume-viewer
==========================

.. program-output:: ../../../scripts/isct_minc2volume-viewer.py -h


isct_test_ants
=================

.. program-output:: ../../../scripts/isct_test_ants.py -h


isct_warpmovie_generator
===========================

.. program-output:: ../../../scripts/isct_warpmovie_generator.py -h


msct_base_classes
====================

.. program-output:: ../../../scripts/msct_base_classes.py -h


msct_gmseg_utils
===================

.. program-output:: ../../../scripts/msct_gmseg_utils.py -h


msct_image
=============

.. program-output:: ../../../scripts/msct_image.py -h


msct_moco
============

.. program-output:: ../../../scripts/msct_moco.py -h


msct_multiatlas_seg
======================

.. program-output:: ../../../scripts/msct_multiatlas_seg.py -h


msct_nurbs
=============

.. program-output:: ../../../scripts/msct_nurbs.py -h


msct_parser
==============

.. program-output:: ../../../scripts/msct_parser.py -h


msct_pca
===========

.. program-output:: ../../../scripts/msct_pca.py -h


msct_register
================

.. program-output:: ../../../scripts/msct_register.py -h


msct_register_landmarks
==========================

.. program-output:: ../../../scripts/msct_register_landmarks.py -h


msct_shape
=============

.. program-output:: ../../../scripts/msct_shape.py -h


msct_smooth
==============

.. program-output:: ../../../scripts/msct_smooth.py -h


msct_types
=============

.. program-output:: ../../../scripts/msct_types.py -h




