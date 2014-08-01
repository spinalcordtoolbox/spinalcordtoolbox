
# CHANGES TO RELEASE

1.0.4
- NEW: sct_crop: function to quickly crop an image.
- REF: sct_estimate_MAP_tracts.py now called sct_estimate_tracts.py. New functionalities added (maximum likelihood estimation and tract grouping).

1.0.3 (2014-07-30)

- BUG: fixed bug in sct_process_segmentation.py related to import of scipy.misc imsave,imread in miniconda distrib (issue #62)
- BUG: fixed bug in sct_process_segmentation.py related to import of PIL/Pillow module (issue #58)
- OPT: sct_register_multimodal now working for images with non-axial orientation (issue #59)
- OPT: sct_register_straight_spinalcord_to_template has now been replaced by sct_register_multimodal in sct_register_to_template.
- OPT: major improvements for sct_dmri_moco, including spline regularization, eddy correction, group-wise registration, gaussian mask.
- OPT: sct_check_dependencies.py can now output log file and do extensive tests (type -h for more info)
- NEW: sct_apply_transfo.py: apply warping field (wrapper to ANTs WarpImageMultiTransform)
- NEW: sct_concat_transfo.py: concatenate warping fields (wrapper to ANTs ComposeMultiTransform)
- NEW: batch_processing.sh: example batch for processing multi-parametric data

1.0.2 (2014-07-13)

- NEW: virtual machine
- BUG: fixed sct_check_dependencies for Linux
- BUG: fix VM failure of sct_register_to_template (issue #41)
- OPT: sct_register_to_template.py now registers the straight spinal cord to the template using sct_register_multimodal.py, which uses the spinal cord segmentation for more accurate results.

1.0.1 (2014-07-03)

- INST: toolbox now requires matplotlib

1.0 (2014-06-15)

- first public release!

0.7 (2014-06-14)

- NEW: dMRI moco
- INST: libraries are now statically compiled
- OPT: propseg: now results are reproducible (i.e. removed pseudo-randomization)

0.6 (2014-06-13)

- Debian + OSX binaries
- BUG: fixed registration2template issue when labels were larger than 9
- BUG: fixed bug on PropSeg when the image contains a lot of null slices
- INST: now installer write on bashrc and links bash_profile to bashrc
- BUG: removed random parts in PropSeg

0.5 (2014-06-03)

- now possible to get both template2EPI and EPI2template warping fields
- fixed major bug in registration (labels were cropped)
- NEW: probabilistic location of spinal levels
- NEW: binaries for Debian/Ubuntu

0.4 (2014-05-28)

- NEW: installer for ANTs (currently only for OSX)
- fixed bugs

0.3 (2014-05-26)

- major changes in sct_register_multimodal
- fixed bugs

0.2 (2014-05-18)

- NEW: nonlocal means denoising filter
- NEW: sct_smooth_spinalcord --> smoothing along centerline
- fixed bugs

0.1 (2014-05-03)

- first beta version!
