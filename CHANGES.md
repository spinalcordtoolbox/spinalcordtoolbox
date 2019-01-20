# CHANGES TO RELEASE

## v4.0.0 (2019-01-20)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.2.7...v4.0.0)

**BUG**

 - **sct_extract_metric:** Fixed bug in method max. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2087)
 - **sct_flatten_sagittal:** Fix bugs related to image scaling. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2070)
 - **sct_label_vertebrae:** Fixed path issue when using -initlabel flag. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2068)
 - **sct_get_centerline:** Convert data to float before intensity rescaling (in optic). [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2065)
 - **sct_deepseg_lesion,sct_deepseg_sc:** Fixed ValueError and IndexError. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2060)
 - **sct_register_to_template:** Fixed regression bugs. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2052)

**ENHANCEMENT**

 - **sct_label_vertebrae:** Removed support for -initc2 flag because there is an alternative approach with sct_label_utils. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2125)
 - **sct_extract_metric:** Expose aggregate_slicewise() API and various improvements. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2115)
 - **sct_register_to_template:** Updated PAM50 template header to be in the same coordinate system as the MNI template. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2102)
 - **sct_qc:** Various improvements. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2097)
 - **sct_deepseg_lesion,sct_deepseg_sc:** deepseg_sc: Speed processing up . [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2094)
 - **sct_qc:** QC now scales images based on physical dimensions (previously based on voxels). [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2089)
 - **sct_process_segmentation:** Major refactoring to bring few improvements. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1931)

**FEATURE**

 - **sct_register_to_template:** Now possible to specify the type of algorithm used for cord straightening. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2101)
 - **sct_label_vertebrae:** spinalcordtoolbox/vertebrae/detect_c2c3 -- New module. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2084)
 - **sct_compute_snr:** Now possible to output SNR map, removed requirement for inputing mask, and few other improvements. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2080)
 - **sct_dmri_separate_b0_and_dwi:** sct_dmri_separate_b0_and_dwi: Now append suffix to input file name to prevent conflicts. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2075)
 - **sct_smooth_spinalcord:** Enable to set smoothing parameters in all axes. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2073)

**DOCUMENTATION**

 - **sct_label_vertebrae:** Updated documention on how to create vertebral and disc labels. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2107)
 - **sct_changelog:** Few improvements on automatic Changelog generation. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2077)


## v3.2.7 (2018-10-29)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.2.6...v3.2.7)

**BUG**

 - sct_fmri_moco: Fixed regression bug related to the use of a mask [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2047)
 - msct_nurbs: Fixed singular matrix error [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2042)

**ENHANCEMENT**

 - sct_extract_metric: Do not zero negative values [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2049)


## v3.2.6 (2018-10-16)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.2.5...v3.2.6)

**BUG**

 - sct_propseg: Reordered variable assignment [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2039)
 - sct_straighten_spinalcord: Fixed AttributeError related to conversion of numpy array to list [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2026)
 - sct_create_mask: Few fixes and improvements [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2021)

**ENHANCEMENT**

 - sct_get_centerline: Use the new viewer [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2035)
 - sct_fmri_moco: Generalize motion correction for sagittal acquisitions and other improvements [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2022)

**FEATURE**

 - sct_straighten_spinalcord: Few fixes and improvements [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2033)
 - sct_deepseg_sc/lesion: Allow to input manual or semi-manual centerline [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2020)


## v3.2.4 (2018-08-24)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.2.3...v3.2.4)

**BUG**

 - Updated URL for PAM50 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1997)
 - sct_register_to_template: Fixed wrong projection in case labels not in same space [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1978)
 - sct_extract_metric: Fixed recently-introduced bug related to output results [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1965)

**ENHANCEMENT**

 - Few fixes in sct_extract_metric and batch processing outputs [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/2008)
 - sct_dmri_compute_dti: Output tensor eigenvectors [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1975)
 - Totally pimp the Image Slicer (to act like a sequence, to slice many images), and add unit tests for the slicer [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1956)
 - Second pass at image refactoring [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1948)


## v3.2.3 (2018-07-21)
This release notably brings a useful feature, which now makes it possible to use single-label with -l flag for registration to the template. This feature was required by the recently-introduced [analysis pipeline for multi-parametric data when FOV is systematically centered at a particular disc or mid-vertebral level](https://github.com/sct-pipeline/multiparametric-fixed-fov). [View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.2.2...v3.2.3)

**BUG**

 - `sct_register_multimodal`: Fixed bug when using partial mask with algo=slicereg [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1917)

**ENHANCEMENT**

 - `sct_propseg`: Labels and centerline are now output with correct header if -rescale is used [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1926)
 - Adding a batch size of 4 for all deep learning methods. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1924)

**FEATURE**

 - `sct_register_to_template`: Enable single-label with -l flag [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1955)


## v3.2.2 (2018-07-08)
Major changes to this release include a fix to SCT installation on OSX laptops with non-English encoding language. Another important fix is the inclusion of the link in `sct_download_data` for downloading the Paris'18 SCT course material. A nice enhancement is the possibility to calculate metrics slice-wise or level-wise in `sct_extract_metric`. View detailed changelog
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.2.1...v3.2.2)

**BUG**

 - sct_label_vertebrae: Added subcortical colormap for fslview [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1883)
 - sct_flatten_sagittal: Fixed wrong indexation [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1882)

**ENHANCEMENT**

 - sct_deepseg_gm: Lazy loading module: now faster when calling usage [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1908)
 - sct_propseg: Now possible to rescale data header to be able to segment non-human spinal cord (mice, rats, etc.) [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1903)
 - sct_deepseg_gm: Adding TTA (test-time augmentation) support for better segmentation results [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1894)
 - sct_deepseg_gm: Removed restriction on the network input size (small inputs): Fixes bug that appeared when inputting images with small FOV [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1877)
 - sct_deepseg_sc: Reducing TensorFlow cpp logging verbosity level [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1876)
 - sct_extract_metric: Now possible to calculate metrics slice-wise or level-wise [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1875)

**DOCUMENTATION**

 - Added documentation for installing SCT on Windows using Docker [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1879)
 - Added information on the README about how to update SCT from git install [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1867)
 - Updated documentation and added link to the data for the SCT course in Paris [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1858)

**INSTALLATION**

 - Use pip install -e spinalcordtoolbox to gain flexibility [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1892)
 - Local language support (LC_ALL) added to installation& launcher on macOS [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1881)

**TESTING**

 - Removed sct_register_graymatter (obsolete old code) from sct_testing functions [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1904)
 - Implemented multiprocessing and argparse in sct_testing, and other improvements related to Sentry [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1872)


## 3.2.1 (2018-06-12)
This release fixes a few bugs, notably one related to template registration when using disc-based alignment. It also features an improved version of sct_deepseg_sc with the introduction of 3D kernel models, as well as a more accurate segmentation on T1-weighted scans. The main documentation now includes a link to a new collection of repositories: sct-pipeline, which gathers examples of personalized analysis pipelines for processing spinal cord MRI data with SCT. [View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.2.0...3.2.1)

**BUG**

 - Skip URL if filename isn't provided by HTTP server; catch anything in URL try loop [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1843)
 - Fixed registration issue caused by labels far from cord centerline [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1828)
 - Fixed wrong disc labeling and other minor improvements [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1814)
 - Added test to make sure not to crop outside of slice range [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1813)
 - Forcing output type to be float32 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1800)
 - Fixed z_centerline_voxel not defined if -no-angle is set to 1 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1797)

**ENHANCEMENT**

 - Adding threshold (or not) option for the sct_deepseg_gm [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1846)
 - Manual centerline is now output when using viewer [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1840)
 - Added CNN for centerline detection, brain detection and added possibility for 3d CNN kernel [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1822)
 - Fixed verbose in QC, integrated coveralls [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1808)
 - Now possible to specify a vertebral labeling file when using -vert [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1798)

**DOCUMENTATION**

 - Added link to github.com/sct-pipeline [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1796)

**INSTALLATION**

 - Adapted final verbose if user decided to not modify the .bashrc [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1832)

**TESTING**

 - Coveralls added to Travis to prevent build failure [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1847)



## 3.2.0 (2018-05-29)
This release includes: a new example dataset (now includes T2*-w and fMRI data) with an updated batch_processing.sh, a new function to compute MT-saturation effect (sct_compute_mtsat), an improved straightening that can account for inter-vertebral disc positions to be used alongside sct_register_to_template for more accurate registration, and few improvements on sct_pipeline and quality control (QC) report generation. [View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.1.1...3.2.0)

**BUG**

 - Fixed sct_pipeline if more than two -p flags are used [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1760)
 - Fixed re-use of the same figure during QC generation [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1711)
 - sct_deepseg_sc - Issue when input is .nii instead of .nii.gz [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1706)
 - Fslview no more called at the end of process if it it deprecated [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1648)
 - Fixing the TensorFlow installation for some old platforms. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1646)
 - Re-ordering of 4th dimension when apply transformation on 4D scans [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1638)
 - Fix "-split" option issues on sct_image [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1627)

**ENHANCEMENT**

 - Updated batch_processing and sct_example_data with new features [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1779)
 - Various fixes for sct_pipeline [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1771)
 - sct_pipeline: store metadata in Pickle report [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1761)
 - Adding volume-wise standardization normalization for the sct_deepseg_gm [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1757)
 - Make sct_get_centerline robust to intensities with range [0, 1] [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1746)
 - Improved doc and minor fixes with centerline fitting [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1736)
 - Make sct_process_segmentation compatible with the new ldisc convention [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1735)
 - Removed flirt dependency [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1729)
 - More pessimistic caching of outputs [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1719)
 - Slice counting fixed [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1687)
 - output of -display ordered per label value [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1686)
 - Improvements in straightening and registration to the template [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1669)
 - The QC report is now a standalone html file. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1668)
 - Adding a port option for the qc server [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1657)
 - Make QC generation opt-in [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1653)
 - Fixing cropping issue in sct_straighten_spinalcord [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1652)
 - Set MeanSquares the default metric for sct_fmri_moco [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1649)
 - Now possible to change data orientation on 4D data [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1637)
 - Use python  concurrent.futures instead of multiprocessing  [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1587)

**FEATURE**

 - New function to create violin plots from sct_pipeline results [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1759)
 - Enable input file with label at a specific disc [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1698)
 - Control the brightness of the image in the GUI. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1684)
 - Implements MTsat function [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1672)
 - Improvements in straightening and registration to the template [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1669)
 - Integration of SCT into fsleyes UI [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1629)
 - Add Sentry error reporting [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1544)
 
## 3.1.1 (2018-02-16)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.1.0...3.1.1)

**BUG**

 - Fix TensorFlow installation on Debian [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1601)
 - BUG: Fixed a small bug on None condition [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1594)
 - Fixed missing output [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1581)
 - Bug fix and various improvements [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1571)
 - Now working for 2d data [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1565)
 - Fix Timer with progress  [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1554)
 - BUG: concat_transfo: fixed wrong catch of dimension [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1534)
 - reinstall only current numpy version [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1520)
 - Enable the calculation of spinal cord shape at the edge of the image [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1517)
 - Disabling rotation in register to template [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1516)

**ENHANCEMENT**

 - Adding minimal Dockerfile for SCT. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1600)
 - Bug fix and various improvements [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1571)
 - Find mirror servers in case OSF is not accessible [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1567)
 - Now supports fsleyes when displaying viewer syntax at the end of a process [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1533)

**FEATURE**

 - sct_deepseg_sc implementation [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1586)
 - sct_deepseg_gm implementation [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1564)

**TESTING**

 - Fixed minor verbose issues during testing [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1582)

## v3.1.0 (2017-10-27)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.0.8...v3.1)

**BUG**

 - Fix errors in create_atlas.m [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1499)
 - Fix a regression bug. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1490)
 - Used the absolute path to create the temporary label file in propseg [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1487)
 - Fixed: Optic is used by default if -init-mask is used with external file provided [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1485)
 - Fixed global dependency in sct_process_segmentation call [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1481)
 - Fixed z-regularization for slicereg [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1465)

**ENHANCEMENT**

 - Fixed: Raise in sct.run in bad order. Also added specific sct errors [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1503)
 - More improvements to the viewer [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1496)
 - Refactored WM atlas creation pipeline and improved documentation  [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1492)
 - Option to install SCT in development mode [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1491)
 - Added key bindings to the undo, save and help actions. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1480)
 - Introduced the zoom functionality to the anatomical canvas [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1477)
 - Improvements on centerline for template generation [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1473)
 - Major refactoring of testing framework [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1453)

**FEATURE**

 - Improvement of sct_analyze_lesions: compute percentage of a given tract occupied by lesions [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1500)
 - sct_get_centerline: new manual feature [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1467)
 - sct_detect_pmj: new_feature [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1466)

**TESTING**

 - Major refactoring of testing framework [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1453)

## v3.0.8 (2017-09-13)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.0.7...v3.0.8)

**BUG**

 - Added try/except for QC report [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1457)
 - Conversion issue for float32 images with large dynamic [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1450)
 - (Partly-)Fixed bug related to memory issue with diagonalization [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1446)
 - DEV: fixed bug on centerline when referencing to the PMJ [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1436)

**ENHANCEMENT**

 - Now possible to input single label at disc (instead of mid-body) [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1451)
 - Now using N-1 instead of N as denominator for computing the STD. [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1441)

**FEATURE**

 - Function to analyze lesions #1351 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1439)

## v3.0.7 (2017-08-02)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.0.6...v3.0.7)

**BUG**

 - The params attributes are initialized to the type integer [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1431)
 - Fixing stdout issue on sct_testing [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1430)
 - Changed destination image for concatenation of inverse warping field [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1413)
 - crashes if apply transfo on 4d images [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1408)
 - Allow the -winv parameter to write a file to disk [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1404)
 - Change import path of resample [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1401)
 - Precision error while calculating Dice coefficient #1098 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1398)

**ENHANCEMENT**

 - Enables to set Gaussian weighting of mutual information for finding C2-C3 disk [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1422)
 - Adapt concat and apply transfo to work on 2d images [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1420)
 - Fixed small issues in pipeline [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1417)
 - Use custom template for sct_register_graymatter [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1407)
 - compute_ernst_angle: set the parameter t1 default value to optional value of 850ms [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1400)
 - Improvements on centerline and template generation [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1378)

**FEATURE**

 - NEW: dmri_display_bvecs: new function to display bvecs [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1394)
 - Function to extract texture features #1350 [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1376)

**TESTING**

 - Various fixes to pipeline and testing [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1424)
 - New test for sct_label_utils compatible with sct_pipeline [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1402)

**DOCUMENTATION**

 - Changed default values and clarified doc [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1405)


## v3.0.6 (2017-07-04)
[View detailed changelog](https://github.com/neuropoly/spinalcordtoolbox/compare/v3.0.5...v3.0.6)

**BUG**

 - Catch the OSError exception thrown when the git command is missing [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1396)
 - BUG: register_multimodal: fixed typo when calling isct_antsRegistration [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1392)
 - BUG: fix bug when slice is empty [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1390)
 - Ignore using user packages when install with conda and pip [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1384)
 - Fix referential for JIM centerline [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1377)
 - image/pad: now copy input data type (fixes issue 1362) [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1363)
 - Use a pythonic way to compare a variable as  None [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1361)
 - The init-mask accepts "viewer" as a value [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1357)
 - Fixed unassigned variable in case -z or -vert is not used [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1354)

**ENHANCEMENT**

 - Restrict deformation for ANTs algo [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1388)
 - Made error message more explicit if crash occurs [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1387)
 - Insert previous and next buttons in the qc reports page [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1358)
 - integrate new class for multiple stdout inside sct_check_dependencies [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1342)

**DOCUMENTATION**

 - Update README.md [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1346)

**INSTALLATION**

 - Ignore using user packages when install with conda and pip [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1384)
 - Update sct testing data [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1370)
 - Added the dependency psutil in the conda requirements [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1359)
 - Added egg files in the list of gitignore [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1355)

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

**FEATURE**

 - Introducing spinal cord shape symmetry [View pull request](https://github.com/neuropoly/spinalcordtoolbox/pull/1332)

**TESTING**

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
