# CHANGES TO RELEASE

##3.0 (TBD)
- NEW: PAM50 template now used as the default template
- NEW: **sct_compute_snr**: compute SNR using several methods (Dietrich et al. 2007)
- NEW: **sct_propseg**: now accepts a correction solution for sct_propseg in case of missing CSF/SC contrast and/or artefacts (see issue #664 for details)
- NEW: **sct_propseg**: added flag to open a viewer for initializing spinal cord segmentation by manually providing a few points (issue #741)
- NEW: **install_sct**: new installer, which downloads the necessary data (i.e., lighter package).
- NEW: SCT now includes its own python (from miniconda), which simplifies the installation and allows users to have another Python installed without generating conflicts.
- NEW: **sct_dmri_create_noisemask**: Identification and estimation of noise in the diffusion signal, implemented by the Dipy software project (http://nipy.org/dipy/), based on the PIESNO method
- NEW: **sct_register_graymatter**: Multi-label registration that accounts for gray matter shape.
- NEW: **sct_register_multimodal**: features two new transformations: centermassrot and columnwise.
- BUG: **sct_straighten_spinalcord**: Fixed #917, #924
- BUG: Fixed issues #715, #719
- BUG: **sct_propseg**: fixed issues #147, #242, #309, #376, #501, #544, #674, #680
- BUG: **sct_segment_graymatter**: fixed issues #782, #813, #815
- BUG: Fixed incompatibility with CENTOS 6.X (issue #776)
- BUG: Binaries now hosted on Gihub for accessibility from China (#927)
- OPT: **sct_check_dependencies**: Made test more sentitive to OS incompatibilities (issue #771)
- OPT: **sct_register_multimodal**: major changes. Simplified flags. Fixed issues #350, #404, #414, #499, #650, #735, #737, #749, #807, #818
- OPT: **sct_register_to_template**: now uses slicewise rigid transfo at first step (instead of slicereg), which improves accuracy (issue #666)
- OPT: **sct_label_vertebrae**: now fully automatic (although unstable-- work in progress).

##2.2.3 (2016-02-04)
- BUG: **sct_straighten_spinalcord**: fixed instabilities related to generation of labels (issue #722)

##2.2.2 (2016-01-31)
- OPT: **sct_dmri_moco**: added flag "-bvalmin" to specify b=0 threshold and improved reading of bval file.

##2.2.1 (2016-01-29)
- BUG: **sct_dmri_moco**: fixed bug related to the use of mask
- BUG: **sct_dmri_moco**: fixed bug in the algorithm (iterative average of target DWI volume)

##2.2 (2016-01-23)
- BUG: Fixed major issue during installation (issue #708)
- BUG: **sct_process_segmentation**: fixed bug occuring with small FOV (issue #706)

##2.1.1 (2016-01-15)
- BUG: **sct_resample**: fixed issue #691
- OPT: **sct_segment_graymatter**: improved robustness of normalization
- OPT: **sct_process_segmentation**: default parameter does not smooth CSA results anymore

##2.1 (2015-12-01)
- NEW: **sct_testing**: test SCT functions and their integrity
- NEW: **sct_maths**: performs basic operations on images. Similar to fslmaths.
- NEW: **sct_get_centerline -method auto**: uses advanced image processing methods for finding the spinal cord centerline automatically on any type of contrast. This script should be followed by sct_propseg for finer cord segmentation.
- NEW: **sct_label_vertebrae**: can automatically label vertebral levels given an anatomical scan, a centerline and few prior info.
- NEW: **sct_segment_graymatter**: segment spinal cord gray matter using multi-atlas approach from Asman et al.
- NEW: **sct_process_segmentation**: feature to estimate CSA based on labels
- NEW: **sct_label_utils**: new functionality for creating labels based on vertebral labeling
- NEW: added "-qc" flag to some functions to output png images for quality control.
- BUG: **install_patch**: now possible to install as non-admin (issues #380, #434)
- BUG: **sct_extract_metric**: fix the case when averaging labels from different clusters with method map
- INST: no more dependence with FSL
- INST: no more dependence with c3d
- OPT: **sct_straighten_spinalcord**: improved accuracy (issues #371, #425, #452, #472)
- OPT: **sct_registration_to_template**: improved accuracy
- REF: harmonization of flags. Most flags from v2.0 still work but a message of deprecation is sent.

##2.1_beta21 (2015-11-30)
- **sct_process_segmentation**: fixed issue with computation of volume based on vertebral level (slice selection now using centerline)

##2.1_beta20 (2015-11-30)
- fixed compatibility with new PAM50 template

##2.1_beta19 (2015-11-25)
- harmonized flags
- **sct_process_segmentation**: now computes volume

##2.0.6 (2015-06-30)
- BUG: **sct_process_segmentation**: fixed bug of output file location (issue #395)

##2.0.5 (2015-06-10)
- BUG: **sct_process_segmentation**: fixed error when calculating CSA (issue #388)

##2.0.4 (2015-06-06)
- BUG: **sct_process_segmentation**: fixed error when calculating CSA (issue #388)
- BUG: Hanning smoothing: fixed error that occurred when window size was larger than data (issue #390)
- OPT: **sct_check_dependencies**: now checks if git is installed
- OPT: simplified batch_processing.sh

##2.0.3 (2015-05-19)
- BUG: **sct_register_to_template**: fixed issue related to appearance of two overlapped templates in some cases (issue #367)
- BUG: **sct_register_to_template**: now all input data are resampled to 1mm iso to avoid label mismatch (issue #368)
- BUG: **sct_resample**: fixed bug when user specified output file
- OPT: **sct_create_mask**: improved speed

##2.0.2 (2015-05-16)
- BUG: **sct_fmri_compute_tsnr**: fixed issue when input path includes folder
- BUG: **sct_orientation**: now possibility to change orientation even if no qform in header (issue #360)
- BUG: **msct_smooth**: fixed error with small Hanning window (issue #363)
- BUG: **sct_straighten_spinalcord**: fixed issue with relative path (issue #365)
- NEW: **sct_label_utils**: added new function to transform group of labels into discrete label points
- NEW: **sct_orientation**: added a tool to fix wrong orientation of an image (issue #366)
- OPT: **sct_register_to_template**: twice as fast! (issue #343)

##2.0.1 (2015-04-28)
- BUG: **sct_extract_metric**: MAP method did not scale properly with the data. Now fixed (issue #348)
- BUG: fixed issue with parser when typing a command to see usage (it crashed)

##2.0 (2015-04-17)

- NEW: **sct_fmri_compute_tsnr**: new function to compute TSNR from fMRI data (performs moco before)
- OPT: **sct_straighten_spinalcord**: now MUCH faster and more accurate (issue #240)
- OPT: **sct_register_to_template**: allows more flexibility by allowing multiple steps for registration (flag -p).
  - N.B. flag "-m" has been replaced by "-s"
- OPT: **sct_register_multimodal**: allows more flexibility by imposing only one stage. Several stages can be run sequentially and then transformations can be concatenated.
  - N.B. flags "-s" and "-t" were replaced with "-iseg" and "-dseg" respectively
- OPT: **sct_extract_metric**: 
  - new methods for extraction: maximum likelihood and maximum a posteriori, which take into account partial volume effect
  - now possible to specify global regions for extraction with flag -l: wm, gm, sc
  - now possible to include a bunch of labels using ":". Example: 2:29
- NEW: **sct_get_centerline_from_labels**: obtain a centerline using a combination of labels and/or segmentations
  - N.B. sct_get_centerline was renamed for sct_get_centerline_automatic
- NEW: **sct_compute_ernst_angle**: new script to compute and display Ernst angle depending on T1 and TR
- OPT: **sct_process_segmentation**:
  - can compute average CSA across vertebral levels or slices
  - can compute length of segmentation
  - can compute CSA on non-binary images such as probabilistic gray/white matter maps
  - N.B. process names were simplified to: "csa", "length" and "centerline"
- OPT: **sct_crop_image**: now possible to crop an image based on a reference space
- OPT: new WM atlas: added gray matter and CSF for computing partial volume
- OPT: now use all available cores for ANTs and adjust variable when running dmri_moco (issue #238)
- INST: new installer in python, simpler to use and check for latest patches
- REF: msct_parser: new parser that generate documentation/usage
- REF: msct_image, sct_label_utils: smoothly converting the toolbox to objet-oriented, some scripts can be used as python module

##1.1.2_beta (2014-12-25)

- BUG: sct_dmri_moco: fixed crash when using mask (issue # 245)
- OPT: sct_create_mask: (1) updated usage (size in vox instead of mm), (2) fixed minor issues related to mask size.
- INST: links are now created during installation of release or patch (issue ).

##1.1.1 (2014-11-13)

- FIX: updated ANTs binaries for compatibility with GLIBC_2.13 (issue: https://sourceforge.net/p/spinalcordtoolbox/discussion/help/thread/e00b2aeb/)

##1.1 (2014-11-04)

- NEW: sct_crop: function to quickly crop an image.
- NEW: sct_extract_metric (replaces the old sct_estimate_MAP_tracts.py). New functionalities added (maximum likelihood estimation and tract grouping). More flexible with label files.
- NEW: sct_convert_mnc2nii
- NEW: sct_create_mask: create mask of different shapes (cylinder, box, gaussian). Useful for moco.
- NEW: sct_fmri_moco: motion correction function for fMRI data. Uses regularization along z.
- NEW: sct_compute_mtr: compute MTR
- NEW: sct_otsu: OTSU segmentation (usefull for DWI data)
- NEW: sct_resample: quick upsample/downsample 3D or 4D data
- NEW: sct_segment_greymatter: function to segment the grey matter by warping that one from the atlas
- OPT: sct_orientation can now be applied to 4d data
- OPT: sct_register_multimodal now using the new antsSliceReg method that regularizes along z.
- OPT: new version of the white matter atlas: more accurate, deformation accounting for internal structure (use BSplineSyN instead of SyN).
- OPT: sct_dmri_moco now using the new antsSliceReg method that regularizes along z.
- OPT: removed all .py extensions for callable functions (created links)
- OPT: sct_label_utils: now possible to create labels. Also added other useful features.
- INST: now possible to specify installation path for the toolbox
- INST: conda dependences are now automatically installed by the installer.
- INST: added pillow (fixed issue #117)
- INST: "getting started" now provided via example commands in batch_processing.sh
- REF: sct_straighten_spinalcord (fixed issues #56, #116)
- TEST: major changes on the testing framework for better modularity with Travis. Now using separate small dataset.

##1.0.3 (2014-07-30)

- BUG: fixed bug in sct_process_segmentation.py related to import of scipy.misc imsave,imread in miniconda distrib (issue #62)
- BUG: fixed bug in sct_process_segmentation.py related to import of PIL/Pillow module (issue #58)
- OPT: sct_register_multimodal now working for images with non-axial orientation (issue #59)
- OPT: sct_register_straight_spinalcord_to_template has now been replaced by sct_register_multimodal in sct_register_to_template.
- OPT: major improvements for sct_dmri_moco, including spline regularization, eddy correction, group-wise registration, gaussian mask.
- OPT: sct_check_dependencies.py can now output log file and do extensive tests (type -h for more info)
- NEW: sct_apply_transfo.py: apply warping field (wrapper to ANTs WarpImageMultiTransform)
- NEW: sct_concat_transfo.py: concatenate warping fields (wrapper to ANTs ComposeMultiTransform)
- NEW: batch_processing.sh: example batch for processing multi-parametric data

##1.0.2 (2014-07-13)

- NEW: virtual machine
- BUG: fixed sct_check_dependencies for Linux
- BUG: fix VM failure of sct_register_to_template (issue #41)
- OPT: sct_register_to_template.py now registers the straight spinal cord to the template using sct_register_multimodal.py, which uses the spinal cord segmentation for more accurate results.

##1.0.1 (2014-07-03)

- INST: toolbox now requires matplotlib

##1.0 (2014-06-15)

- first public release!

##0.7 (2014-06-14)

- NEW: dMRI moco
- INST: libraries are now statically compiled
- OPT: propseg: now results are reproducible (i.e. removed pseudo-randomization)

##0.6 (2014-06-13)

- Debian + OSX binaries
- BUG: fixed registration2template issue when labels were larger than 9
- BUG: fixed bug on PropSeg when the image contains a lot of null slices
- INST: now installer write on bashrc and links bash_profile to bashrc
- BUG: removed random parts in PropSeg

##0.5 (2014-06-03)

- now possible to get both template2EPI and EPI2template warping fields
- fixed major bug in registration (labels were cropped)
- NEW: probabilistic location of spinal levels
- NEW: binaries for Debian/Ubuntu

##0.4 (2014-05-28)

- NEW: installer for ANTs (currently only for OSX)
- fixed bugs

##0.3 (2014-05-26)

- major changes in sct_register_multimodal
- fixed bugs

##0.2 (2014-05-18)

- NEW: nonlocal means denoising filter
- NEW: sct_smooth_spinalcord --> smoothing along centerline
- fixed bugs

##0.1 (2014-05-03)

- first beta version!
