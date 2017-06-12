# CHANGES TO RELEASE

## v3.0.5 (2017-06-09)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.0.4...v3.0.5)

**BUG**

 - Force numpy 1.12.1 on osx and linux [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1352)
 - Use a different function to identify if a file exists [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1341)
 - Fixing an issue introduced with the sct_get_centerline. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1338)

**ENHANCEMENT**

 - Binarize GM seg after warping back result to original space [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1347)
 - Generation of centerline as ROI [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1344)

**FEATURE**

 - Introduce a pipeline to use the HPC architecture [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1260)

## v3.0.4 (2017-05-19)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.0.3...v3.0.4)

**BUG**

 - Normalize the init value to between 0 and 1 for propseg [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1324)
 - Moved the QC assets into the spinalcordtoolbox package [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1311)
 - Improved the formatting of the changelog generator [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1303)
 - Show remaining time status for downloads [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1299)

**ENHANCEMENT**

 - Added the command parameter `-noqc` [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1313)
 - Add dimension sanity checking for input file padding op [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1304)

 - **FEATURE**

 - Introducing spinal cord shape symmetry [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1332)

 - **TESTING**

 - Validate the function name in sct_testing [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1305)
 - Fix regression bug in sct_testing [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1310)

## v3.0.3 (2017-04-26)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.0.2...v3.0.3)

**BUG**

 - Fixes case if data image, segmentation and labels are not in the same space [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1292)
 - Fix the handling of the path of the QC report. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1285)
 - Change the format of the SCT version. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1284)
 - changed the DISPLAY variable due to conflicts with FSLView in batch_processing.sh [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1281)

**INSTALLATION**

 - Added course_hawaii17 into the list of available dataset from sct_download_data [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1291)
 - Incorrect variable when installing SCT in a different directory [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1289)

**DOCUMENTATION**

 - Added description with examples in the register_to_template command (#1262) [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1287)
 - Fixed typo in register_multimodal command [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1276)

## v3.0.2 (2017-04-20)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.0.1...v3.0.2)

**BUG**

 - Force the SCT environment to use only the python modules installed by SCT [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1266)
 - Fixing disabling options on straightening [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1255)
 - Fixed tSNR computation of the mean and std of the input image [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1245)
 - Increased the data type size from the default int16 to int32 to avoid overflow issues in sct_process_segmentation [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1224)
 - Fixed data type issue in sct_process_segmentation [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1223)

**ENHANCEMENT**

 - Improvements to denoising on sct_segment_graymatter [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1265)
 - Extend the functionality of sct_viewer [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1254)
 - Add OptiC for improved spinal cord detection [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1253)
 - Introduction spinalcordtoolbox python setup file [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1243)

**FEATURE**

 - Add option -rms to perform root mean square (instead of mean) in sct_maths [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1256)
 - Introduce QC report generation [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1251)
 - Introduce the QC html viewer [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1250)

**TESTING**

 - Introduce the QC html viewer [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1250)
 - Introduce python package configuration file (setup.cfg) [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1242)

## v3.0.1 (2017-03-22)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.0.0...v3.0.1)
### FEATURE
 - Merge multiple source images onto destination space. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1220)

## v3.0.0 (2017-03-15)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.0_beta32...v3.0.0)
### BUG
 - Modifying the type of coordinates for vertebral matching [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1206)
 - Removing discontinuities at edges on segmentation [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1196)
 - BUG: computing centreline using physical coordinates instead of voxel… [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1189)
 - Fix issue #1172: -vertfile as an optional parameter [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1173)
 - Improvements to the viewer of sct_propseg [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1169)
 - Removed confusion with command variables when using PropSeg viewer [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1161)
 - Patch sct_register_to_template with -ref subject [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1158)
 - zero voxels no more included when computing MI + new flag to compute normalized MI [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1136)
### ENHANCEMENT
 - Changed default threshold_distance from 2.5 to 10 to avoid edge effect [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1183)
 - Adapt sct_create_mask and sct_label_utils to 2D data [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1180)
 - Improvements to the viewer of sct_propseg [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1169)
### TESTING
 - OPT: display mean and std instead of mean twice [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1187)

## 3.0_beta32 (2017-02-10)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.0_beta31...v3.0_beta32)
### BUG
 - BUG: install_sct: fixed PATH issue (#1153): closed at 2017-02-08 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1154)
 - BUG: compute_snr: fixed variable name: closed at 2017-02-03 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1148)
 - Changed the algorithm to fetch the download filename: closed at 2017-02-03 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1146)
 - Copy header of input file to ensure qform is unchanged: closed at 2017-01-31 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1137)
 - zero voxels no more included when computing MI + new flag to compute normalized MI: closed at 2017-02-01 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1136)
 - Downloading the binaries using the python module instead of CURL: closed at 2017-01-30 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1134)
 - [sct_segment_graymatter] correct background value: closed at 2017-01-31 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1133)
 - Fixing indexes issue on Travis OSX: closed at 2017-01-17 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1115)
 - REF: display spinal cord length when required (full spinal cord): closed at 2017-01-17 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1112)
 - Adding rules for in-segmentation errors: closed at 2017-01-17 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1110)
### ENHANCEMENT
 - Generate a changelog from GitHub: closed at 2017-02-10 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1152)
 - OPT: maths: visu only produced if verbose=2: closed at 2017-02-02 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1147)
### TESTING
 - Add message to user when spinal cord is not detected and verbose improvement for testing: closed at 2017-02-01 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1145)
 - Display results of isct_test_function: closed at 2017-01-20 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1117)

## 3.0_beta31 (2017-01-16)
- BUG: **sct_process_segmentation** display spinal cord length when required (full spinal cord) (#1112)
- BUG: **sct_propseg** Adding rules for in-segmentation errors (#1110)
- BUG: PAM50: probabilist templates of WM and GM not sum to 1 (#411)
- BUG: **sct_propseg**: fixed edge issue (#1074)
- BUG: **sct_label_vertebrae**: fixed temporary folder not created (#1083)
- BUG: **isct_test_function**: fixed temp folder of subject being created inside test folder of another subject (#1084)
- BUG: **sct_apply_transfo**: fixed TR field on NIFTI is lost (#1013)
- BUG: **sct_register_graymatter**: fixed empty inverse warping field (#1068)
- OPT: **sct_label_vertebrae**: now outputing verbose=2 pics in ofolder (#1094)
- OPT: **sct_straighten_spinalcord**: fixed Reduce required RAM (#979)
- OPT: **sct_straighten_spinalcord**: removes accuracy computation by default (#1075)
- OPT: **sct_propseg**: improve robustness towards exception (#207)
- OPT: **isct_test_function**: send email when finished (#1081)
- OPT: **isct_test_function**: removed color tags on log entries  (#1035)

## 3.0_beta29 (2016-12-13)
- NEW: PAM50 template now used as the default template
- NEW: **sct_compute_snr**: compute SNR using several methods (Dietrich et al. 2007)
- NEW: **sct_propseg**: now accepts a correction solution for sct_propseg in case of missing CSF/SC contrast and/or artefacts (see issue #664 for details)
- NEW: **sct_propseg**: added flag to open a viewer for initializing spinal cord segmentation by manually providing a few points (issue #741)
- NEW: **install_sct**: new installer, which downloads the necessary data (i.e., lighter package).
- NEW: SCT now includes its own python (from miniconda), which simplifies the installation and allows users to have another Python installed without generating conflicts.
- NEW: **sct_dmri_create_noisemask**: Identification and estimation of noise in the diffusion signal, implemented by the Dipy software project (http://nipy.org/dipy/), based on the PIESNO method
- NEW: **sct_register_graymatter**: Multi-label registration that accounts for gray matter shape.
- NEW: **sct_register_multimodal**: features two new transformations: centermassrot and columnwise.
- NEW: **sct_register_multimodal**: flag smoothWarpXY: regularization of warping field (only for algo=columnwize)
- NEW: **sct_register_multimodal**: flag pca_eigenratio_th: Min ratio between the two eigenvalues for PCA-based angular adjustment (only for algo=centermassrot).
- NEW: **sct_create_mask**: now compatible with 2D data (#1066)
- NEW: **sct_maths**: computes mutual information and cross-correlation between images (#1054)
- BUG: **sct_straighten_spinalcord**: Fixed #917, #924, #1063
- BUG: Fixed issues #715, #719
- BUG: **sct_propseg**: fixed issues #147, #242, #309, #376, #501, #544, #674, #680
- BUG: **sct_segment_graymatter**: fixed issues #782, #813, #815
- BUG: **sct_register_graymatter**: fixed issue #1068
- BUG: Fixed incompatibility with CENTOS 6.X (issue #776)
- BUG: Binaries now hosted on Gihub for accessibility from China (#927)
- BUG: **sct_resample**: Fixed slight image shift caused by resampling (#612)
- OPT: **sct_check_dependencies**: Made test more sentitive to OS incompatibilities (issue #771)
- OPT: **sct_register_multimodal**: major changes. Simplified flags. Fixed issues #350, #404, #414, #499, #650, #735, #737, #749, #807, #818, #1033, #1034
- OPT: **sct_register_to_template**: now uses slicewise rigid transfo at first step (instead of slicereg), which improves accuracy (issue #666)
- OPT: **sct_register_to_template**: added contrast for registration: t2s
- OPT: **sct_label_vertebrae**: now fully automatic (although unstable-- work in progress).
- OPT: **sct_testing**: added integrity testing for CSA computation (#1031)
- REF: **sct_testing**: sct_testing_data is now hosted on GitHub-release for better tracking and across-version compatibility.

## 3.0_beta28 (2016-11-25)
- BUG: **sct_process_segmentation**: Fixed issue related to calculation of CSA (#1022)
- BUG: **sct_label_vertebrae**: Fixed Vertebral labeling removes first vertebrae in the labelled segmentation (#700)
- OPT: **sct_register_multimodal**: Now possible to input initial warping field (#1049)
- OPT: **sct_register_multimodal**: Added feature to be able to input two pairs of label image for estimating affine/rigid/nonrigid transformation (#661)
- OPT: **sct_extract_metric**: Added weighted-Maximum a posteriori extraction method (#1018)
- OPT: Remove color tags on log entries (#1035)

## 3.0_beta27 (2016-10-23)
- NEW: **sct_extract_metric**: method "max" to extract CSA value form interpolated volume (e.g. PAM50 space) without partial volume bias

## 3.0_beta26 (2016-10-05)
- INST: Fixed #992, #1004, #1008, #1012

## 3.0_beta25 (2016-09-30)
- OPT: Fixed #875
- INST: Fixed #1007, #1009

## 3.0_beta24 (2016-09-28)
- BUG: Fixed #870, #898, #859, #871, #1005, #750, #444, #878, #1000
- INST: Fixed issue with matplotlib version 1.5.3

## 3.0_beta23 (2016-09-18)
- BUG: Fixed #984, #983, #954, #978, #987, #938, #964, #638, #969, #922, #855
- OPT: **sct_register_to_template**: added a flag "-ref" to be able to register to anisotropic data

## 3.0_beta22 (2016-09-09)
- BUG: Fixed #994, #989, #988, #976, #968

## 2.2.3 (2016-02-04)
- BUG: **sct_straighten_spinalcord**: fixed instabilities related to generation of labels (issue #722)

## 2.2.2 (2016-01-31)
- OPT: **sct_dmri_moco**: added flag "-bvalmin" to specify b=0 threshold and improved reading of bval file.

## 2.2.1 (2016-01-29)
- BUG: **sct_dmri_moco**: fixed bug related to the use of mask
- BUG: **sct_dmri_moco**: fixed bug in the algorithm (iterative average of target DWI volume)

## 2.2 (2016-01-23)
- BUG: Fixed major issue during installation (issue #708)
- BUG: **sct_process_segmentation**: fixed bug occuring with small FOV (issue #706)

## 2.1.1 (2016-01-15)
- BUG: **sct_resample**: fixed issue #691
- OPT: **sct_segment_graymatter**: improved robustness of normalization
- OPT: **sct_process_segmentation**: default parameter does not smooth CSA results anymore

## 2.1 (2015-12-01)
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

## 2.1_beta21 (2015-11-30)
- **sct_process_segmentation**: fixed issue with computation of volume based on vertebral level (slice selection now using centerline)

## 2.1_beta20 (2015-11-30)
- fixed compatibility with new PAM50 template

## 2.1_beta19 (2015-11-25)
- harmonized flags
- **sct_process_segmentation**: now computes volume

## 2.0.6 (2015-06-30)
- BUG: **sct_process_segmentation**: fixed bug of output file location (issue #395)

## 2.0.5 (2015-06-10)
- BUG: **sct_process_segmentation**: fixed error when calculating CSA (issue #388)

## 2.0.4 (2015-06-06)
- BUG: **sct_process_segmentation**: fixed error when calculating CSA (issue #388)
- BUG: Hanning smoothing: fixed error that occurred when window size was larger than data (issue #390)
- OPT: **sct_check_dependencies**: now checks if git is installed
- OPT: simplified batch_processing.sh

## 2.0.3 (2015-05-19)
- BUG: **sct_register_to_template**: fixed issue related to appearance of two overlapped templates in some cases (issue #367)
- BUG: **sct_register_to_template**: now all input data are resampled to 1mm iso to avoid label mismatch (issue #368)
- BUG: **sct_resample**: fixed bug when user specified output file
- OPT: **sct_create_mask**: improved speed

## 2.0.2 (2015-05-16)
- BUG: **sct_fmri_compute_tsnr**: fixed issue when input path includes folder
- BUG: **sct_orientation**: now possibility to change orientation even if no qform in header (issue #360)
- BUG: **msct_smooth**: fixed error with small Hanning window (issue #363)
- BUG: **sct_straighten_spinalcord**: fixed issue with relative path (issue #365)
- NEW: **sct_label_utils**: added new function to transform group of labels into discrete label points
- NEW: **sct_orientation**: added a tool to fix wrong orientation of an image (issue #366)
- OPT: **sct_register_to_template**: twice as fast! (issue #343)

## 2.0.1 (2015-04-28)
- BUG: **sct_extract_metric**: MAP method did not scale properly with the data. Now fixed (issue #348)
- BUG: fixed issue with parser when typing a command to see usage (it crashed)

## 2.0 (2015-04-17)

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

## 1.1.2_beta (2014-12-25)

- BUG: sct_dmri_moco: fixed crash when using mask (issue # 245)
- OPT: sct_create_mask: (1) updated usage (size in vox instead of mm), (2) fixed minor issues related to mask size.
- INST: links are now created during installation of release or patch (issue ).

## 1.1.1 (2014-11-13)

- FIX: updated ANTs binaries for compatibility with GLIBC_2.13 (issue: https://sourceforge.net/p/spinalcordtoolbox/discussion/help/thread/e00b2aeb/)

## 1.1 (2014-11-04)

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

## 1.0.3 (2014-07-30)

- BUG: fixed bug in sct_process_segmentation.py related to import of scipy.misc imsave,imread in miniconda distrib (issue #62)
- BUG: fixed bug in sct_process_segmentation.py related to import of PIL/Pillow module (issue #58)
- OPT: sct_register_multimodal now working for images with non-axial orientation (issue #59)
- OPT: sct_register_straight_spinalcord_to_template has now been replaced by sct_register_multimodal in sct_register_to_template.
- OPT: major improvements for sct_dmri_moco, including spline regularization, eddy correction, group-wise registration, gaussian mask.
- OPT: sct_check_dependencies.py can now output log file and do extensive tests (type -h for more info)
- NEW: sct_apply_transfo.py: apply warping field (wrapper to ANTs WarpImageMultiTransform)
- NEW: sct_concat_transfo.py: concatenate warping fields (wrapper to ANTs ComposeMultiTransform)
- NEW: batch_processing.sh: example batch for processing multi-parametric data

## 1.0.2 (2014-07-13)

- NEW: virtual machine
- BUG: fixed sct_check_dependencies for Linux
- BUG: fix VM failure of sct_register_to_template (issue #41)
- OPT: sct_register_to_template.py now registers the straight spinal cord to the template using sct_register_multimodal.py, which uses the spinal cord segmentation for more accurate results.

## 1.0.1 (2014-07-03)

- INST: toolbox now requires matplotlib

## 1.0 (2014-06-15)

- first public release!

## 0.7 (2014-06-14)

- NEW: dMRI moco
- INST: libraries are now statically compiled
- OPT: propseg: now results are reproducible (i.e. removed pseudo-randomization)

## 0.6 (2014-06-13)

- Debian + OSX binaries
- BUG: fixed registration2template issue when labels were larger than 9
- BUG: fixed bug on PropSeg when the image contains a lot of null slices
- INST: now installer write on bashrc and links bash_profile to bashrc
- BUG: removed random parts in PropSeg

## 0.5 (2014-06-03)

- now possible to get both template2EPI and EPI2template warping fields
- fixed major bug in registration (labels were cropped)
- NEW: probabilistic location of spinal levels
- NEW: binaries for Debian/Ubuntu

## 0.4 (2014-05-28)

- NEW: installer for ANTs (currently only for OSX)
- fixed bugs

## 0.3 (2014-05-26)

- major changes in sct_register_multimodal
- fixed bugs

## 0.2 (2014-05-18)

- NEW: nonlocal means denoising filter
- NEW: sct_smooth_spinalcord --> smoothing along centerline
- fixed bugs

## 0.1 (2014-05-03)

- first beta version!
