# Changelog

## 7.2 (2025-11-28)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/7.1...7.2)

**FEATURE**
 - **sct_compute_ascor**: Add function to compute aSCOR. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5007)
 - **sct_deepseg**: Add new `-test-time-aug` flag for `nnunetv2` models to turn on `use_mirroring` behavior. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5082)
 - **sct_fmri_moco**: Allow specifying a reference scan as the registration target with new flag `-ref`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5043)
 - **sct_process_segmentation**: Update AP diameter to be based on rotated seg + add new `-anat` flag (for symmetry, quadrant area, and HOG angle properties). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4958)
 - **sct_run_batch**: Add new `-yml` arguments to include/exclude both subjects and files based on YAML lists. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5030)

**ENHANCEMENT**
 - **sct_analyze_lesion**: Add `vert_level` column to `-f` output spreadsheet. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5089)
 - **sct_apply_transfo, sct_label_vertebrae, sct_register_multimodal**: Improve performance of `-x label` when applying a warping field to single-voxel point labels. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5025)
 - **sct_compute_ascor, sct_process_segmentation**: Update `-centerline` argument to allow for consistent `-discfile` projection (used in aSCOR script). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5075)
 - **sct_deepseg**: Rework `totalspineseg` inference (rename task to `spine`, set `-step1-only` behavior as default, improve output filenames). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5097)
 - Hide developer-specific arguments in argparse help. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5035)

**BUG**
 - **sct_compute_ascor, sct_process_segmentation**: Update how `-discfile` intermediate images are saved (output directory, relative paths). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5074)
 - **sct_deepseg, sct_qc**: Fix cropping and labeling for DeepSeg axial mosaic QC Reports. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5069)
 - **sct_process_segmentation**: Solve projection anomaly on anisotropic data. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5067)
 - **sct_register_multimodal**: Fix logging message to correctly print voxel size in `centermassrot` . [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5022)
 - Only create "detailed summary" test files in CI (and only when the directory exists). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5031)

**INSTALLATION**
 - **sct_deepseg**: Switch to `neuropoly` fork of `nnunetv2` to support multi-fold "custom trainer" version of `lesion_ms`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5027)
 - **sct_deepseg**: Pin `blosc2<3.9.0` to avoid failing PRs due to incompatibility wiht `numpy<2.0`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5044)
 - **sct_deepseg**: Make `spinalcord` a default model to solve `PyTorchStreamReader` error in `sct_run_batch`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5076)
 - Allow `blosc2>=3.9.1` due to re-added support for `numpy==1.26`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5046)
 - Restrict `numexpr!=2.14.0` and `h5py!=3.15.0` to address recent installation failures. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5054)
 - Avoid `libffi` bug by installing all of the `conda` packages from `conda-forge` in a single step. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5102)

**DOCUMENTATION**
 - **sct_deepseg**: Added ref for `lesion_ms` ESMRMB 2025 model. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5014)
 - **sct_deepseg**: Update spinal rootlets model description. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5019)
 - **sct_deepseg**: Add citation for RootletSeg model in models.py. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5032)
 - **sct_dmri_moco, sct_fmri_moco**: Improve moco argparse usage for `-param {poly,iter}` (help descriptions) and `-g` (positive integer requirement). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5040)
 - **sct_register_to_template**: Add tutorial for rootlets-based registration to template. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5010)
 - Replace PaperPile links with PubMed links in `pam50.rst` doc page. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5018)
 - Update "Citing SCT" page with missing references and better formatting. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5095)
 - Tweak `-ref subject`-related documentation for 3+ labels. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5109)
 - Update `sct_tutorial_data` links to point to the newest release (SCT Course 2025). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5114)
 - Fix typos in MS lesion tutorial (`ms_lesion` -> `lesion_ms`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5115)

**REFACTORING**
 - **sct_compute_ascor, sct_process_segmentation**: Clarify `ValueError` if the `-centerline` does not cover all the slices in the input mask. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5077)
 - **sct_deepseg, sct_fmri_compute_tsnr, sct_label_utils, sct_label_vertebrae, sct_qc, sct_register_multimodal, sct_register_to_template**: Refactor QC code for six scripts. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5050)

**CI**
 - Refactor the broken link checker. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5047)
 - Replace deprecated macOS 13 runners with new `macos-15-intel` runners. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5088)

**GIT/GITHUB**
 - Update pull request template to mention tests requirement (and improve readability). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5105)

## 7.1 (2025-08-15)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/7.0...7.1)

**FEATURE**
 - **sct_analyze_lesion**: Implement "tissue bridge ratio" metrics. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4933)
 - **sct_compute_compression**: Allow MSCC to be computed on lesion masks (as an alternative to compression labels). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4862)
 - **sct_maths**: Implement anisotropic dilation and erosion. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4966)
 - **sct_process_segmentation**: Add flag `-discfile` to allow for determining `VertLevel` using projected disc labels. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4948)

**ENHANCEMENT**
 - **sct_deepseg**: Update `lesion_ms` DeepSeg model from `r20241101` to `r20250626` (ESMRMB 2025). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4945)
 - **sct_deepseg**: Add slice numbers to axial mosaic QC reports. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4981)
 - **sct_deepseg**: Tweak `totalspineseg` axial mosaic QC report to make it easier to read. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4993)
 - **sct_extract_metric**: Use centerline of `-vertfile` for more accurate mapping of vert levels to `z` slices. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4996)
 - **sct_label_vertebrae**: Speed up labeling for `-discfile` by replacing the straightening step with disc projection. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4896)
 - **sct_register_multimodal**: Mask output of `sct_apply_transfo` when applying `-initwarp` during multimodal registration. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4927)
 - **sct_run_batch**: Improve clarity of terminal errors when a batch script fails. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4950)
 - [EXPERIMENTAL] Add Apptainer installation for use on HPC systems. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4857)
 - Allow the use of custom mapping for `sct_label_vertebrae`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4921)
 - Expand `LazyLoader` usage to improve import times across scripts. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4938)

**BUG**
 - **sct_analyze_lesion**: Change `interpolated_midsagittal_slice` to be saved in the original orientation (instead of `RPI`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4969)
 - **sct_analyze_lesion, sct_qc**: Fix `IndexError` by more thoroughly applying the SC mask to the lesion image. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4961)
 - **sct_deepseg**: Fix two small crashes when generating QC reports using `-qc-seg`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4956)
 - **sct_qc**: Update report-ui to v0.1.1. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4892)
 - **sct_qc**: Use `Path.resolve()` instead of `Path.absolute()`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4917)
 - **sct_register_multimodal, sct_register_to_template**: Add safeguards to handle when `type=imseg` is used with `rot_method=pca`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4924)
 - **sct_register_to_template**: Fix check for user's labels against template's labels by adopting `issubset`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4925)
 - **sct_register_to_template, sct_straighten_spinalcord**: Prevent repeated slices during straightening by correctly zeroing out duplicate slices. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4932)
 - Update PAM50 template to `r20250730` to fix pixel error in `PAM50_rootlets.nii.gz`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4992)

**INSTALLATION**
 - Update SCT's virtual environment to use Python 3.10 instead of 3.9. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4869)
 - Limit `onnx<1.16.2` to fix CI failures on Windows runners. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4901)
 - Add fixes for `PyQt5` and `onnxruntime` to amend our previous switch to Python 3.10. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4906)
 - Validate `$SCT_DIR` for path-length issues (even for the default directory + non-interactive mode). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4935)
 - Fix failing CI runs for Arch (`rl_print_keybindings`) and Debian 10 (EOL). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4987)

**DOCUMENTATION**
 - **sct_deepseg**: Clean up argparse help to only display arguments and descriptions relevant to specific tasks. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5008) 
 - **sct_dmri_moco, sct_fmri_moco**: Make sure that all useful parameters in `ParamMoco` are documented in the `-h`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4979)
 - Move `graymatter` model to `gray_matter` group. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4868)
 - Update and reorganize Docker installation instructions. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5001)

**REFACTORING**
 - Retire our obsolete shim for `str.removesuffix`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4904)

**CI**
 - Enable `macos-15` and `windows-2025` in all CI workflows. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4856)
 - Handle `windows-2019` incompatibility with `onnxruntime==1.22.0` (drop `2019`, limit `<1.22.0`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4907)
 - Add a blanket exclusion for `403 - Forbidden` responses to our broken link checker. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4908)
 - Filter out `Version` and `CodeURL` when diffing JSON sidecars. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4909)
 - Fix `check-url` Github CI failing when no new URLs are present. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4910)
 - Remove default value from Apptainer bundling script to fix CI error. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4911)
 - Fix hanging WSL runners by replacing `windows-2022` with `windows-2025`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4915)
 - Address new `DeprecationWarnings` thrown during `pytest` test suite. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4940)
 - Automatically reset the `stable` docs branch when creating a new release. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4942)
 - Make it easier to manually check QC reports produced by `pytest`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4983)
 - Remove hardcoded path to `D:\` drive for older Windows tests. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4989)
 - `test_cli.py`: Increase threshold for macOS duration failures. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4990)
 - Fix failing tests due to faulty `platform` syntax. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4991)
 - Disable false-positive shell check for `SC2329` in `install_sct`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5000)
 - Tag QC reports from automated tests with the OS used. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/5002)

**GIT/GITHUB**
 - Simplify SCT's pull request template to match the new Contributing Guidelines. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4912)

## 7.0 (2025-04-25)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/6.5...7.0)

**FEATURE**
 - **sct_deepseg**: Add the bavaria-quebec nnUNet model for MS lesion and cord segmentation to sct_deepseg model suite. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4692)
 - **sct_deepseg**: Allow adding `-qc-seg` to crop the QC generated by `sct_deepseg`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4754)
 - **sct_deepseg**: Add new contrast-agnostic `graymatter` model. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4832)
 - **sct_deepseg**: Add `-step1-only` argument to the `totalspineseg` task's subparser. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4860)
 - **sct_detect_compression**: Add new CLI script to predict compression probability. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4760)
 - **sct_qc**: Add new column to QC report to rank images numerically from 0-9. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4764)
 - **sct_register_to_template**: Add `-lrootlet` argument to enable rootlets-informed registration to the PAM50 template. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4647)
 - Add detailed time and memory profiling for CLI scripts. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4844)

**ENHANCEMENT**
 - **sct_compute_compression, sct_download_data**: Update PAM50 normalized metrics URL to latest release (`r20250321`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4846)
 - **sct_analyze_lesion**: Update interpolation logic used when determining the midsagittal slice of lesions. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4702)
  - **sct_deepseg**: Update contrast-agnostic model from v2.5 to v3.0 . [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4843)
 - **sct_deepseg**: Updated `sct_deepseg` usage to improve clarity and ease of use. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4852)
 - **sct_qc**: Overhaul the QC report backend to use modern web technologies (Vite/React/Tailwind/Typescript). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4827)
 - **sct_resample**: Improve clarity of `ZeroDivisionError` message when resampling. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4821)
  - **sct_warp_template**: Update PAM50 release links to `r20250422` (ventral rootlets, Th1 level). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4865)
 - Change over to MiniForge as the package manager (from MiniConda prior). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4768)
 - Add basic automatic time profiling for all CLI scripts by default. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4836)

**BUG**
 - **sct_deepseg**: Preserve the `_seg` suffix for the contrast-agnostic model. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4809)
 - **sct_get_centerline**: Update `get_centerline` to ensure that output arrays match the orientation of the output centerline image. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4622)
 - **sct_process_segmentation**: Fix slice indexing for `-angle-corr-centerline`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4850)
 - **sct_qc**: Replace `pyplot` usage in `qc.py` to address crashing in Jupyter notebooks. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4838)
 - **sct_qc**: Implement QC outlines for multi-valued segmentations. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4723)

**INSTALLATION**
 - Change over to MiniForge as the package manager (from MiniConda prior). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4768)
 - Pin `torch` to `<2.3` to keep versions in sync with Intel Mac platforms. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4788)
 - Transfer hotfixes from the `test-past-releases.yml` CI workflow to the 5.x-6.x series of SCT releases. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4780)
 - Allow ANTs binaries to access `msvc-runtime` DLLs to avoid DLL not found errors. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4784)
 - Gracefully handle permission issues on RC files during installation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4812)

**DOCUMENTATION**
 - **sct_apply_transfo**: Amend argparse help of transform tool to make `-d` and `-o` arguments clearer. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4828)
 - **sct_deepseg**: Update `sct_deepseg` documentation and arg help. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4763)
 - **sct_deepseg**: Fix missed rename of the contrast agnostic model (`_monai` -> `_nnunet`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4808)
 - Update `courses.rst` with SCT Course 2024 links. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4796)
 - Fix wording in opening paragraph within `courses.rst` documentation page. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4802)
 - Update web tutorials to match the changes made for the 2024 SCT Course. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4804)
 - Expand "batch processing" tutorial with steps for manual correction. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4829)

**REFACTORING**
 - **sct_deepseg**: Use argparse subparsers for `sct_deepseg`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4685)
 - **sct_deepseg**: Retire the model `sc_t2star` from `sct_deepseg` (and move its model gallery entry). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4840)
 - Consolidate common elements (`-h`/`-v`/`-r` + argument groups) into `SCTArgumentParser`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4819)

**CI**
 - Improve how we track changes to SCT's outputs by exporting a summary of all output files. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4759)
 - Add new GitHub Actions workflow to test if old releases install without error. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4761)
 - Upgrade Ubuntu 20.04 -> 22.04/24.04 in test suite. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4793)
 - Compare new `batch_processing.sh` file summaries between PRs and `master` branch. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4803)
 - Add automated testing for new profiling utilities. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4830)
 - Remove `--verbose` from default `pytest` config. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4841)

## 6.5 (2024-11-21)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/6.4...6.5)

**FEATURE**
 - **sct_analyze_lesion**: Output lesion length and width for the midsagittal slice. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4617)
 - **sct_deepseg**: Add `-custom-url` arg to allow users to install specific `.zip`s for existing deepseg models. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4618)
 - **sct_deepseg**: Update contrast agnostic model to r20241024 (improved for SCI and whole-spine T1/T2 images). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4661)
 - **sct_deepseg**: Add contrast-agnostic MS lesion segmentation model. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4686)
 - **sct_deepseg**: Add canal segmentation model. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4687) and [follow-up pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4719)
 - **sct_deepseg**: Add TotalSpineSeg model (vertebrae, intervertebral discs, spinal cord, and spinal canal). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4690)
 - **sct_deepseg, sct_qc**: Add `-qc-plane` flag to allow switching the QC view to `'Sagittal'`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4708)
 - **sct_fmri_compute_tsnr, sct_qc**: Add `-m` (mask) and `-qc` options to fMRI TSNR script. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4602)
 - **sct_get_centerline**: Expose `space` API parameter in centerline CLI to allow user to specify `-space phys`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4620)

**ENHANCEMENT**
 - **sct_analyze_lesion**: Update extension of lesion analysis spreadsheet from `.xls` to `.xlsx`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4677)
 - **sct_analyze_lesion, sct_qc**: Swap x-axis in tissue bridges QC. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4609)
 - **sct_apply_transfo**: Preserve integer dtype when warping an image using `NearestNeighbour` interpolation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4680)
 - **sct_apply_transfo, sct_warp_template**: Use faster dilation algorithm when dilating "mostly zero" point label images (`-x label`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4668)
 - **sct_dmri_moco, sct_fmri_moco, sct_register_multimodal**: Convert softmask to binary mask before passing to ANTs, rather than applying softmask to input data. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4611)
 - **sct_download_data**: Add an interactive check before deleting output directory when downloading data. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4650)
 - **sct_download_data**: Update `sct_course_data` URL to point to `SCT-Course-20241209`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4706)

**BUG**
 - **sct_analyze_lesion**: Fix crash during lesion analysis if there is no midsagittal lesion. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4595)
 - **sct_apply_transfo**: Properly maintain verbose status for internal ANTs `run_proc` call (to silence logging for `-v 0`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4603)
 - **sct_deepseg**: Test EPI model to ensure that it has the correct `nnUNetTrainer` model structure. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4615)
 - **sct_deepseg**: Remove faulty `zip()` that breaks `-install` option. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4639)
 - **sct_deepseg**: Fix the suffix of the rootlets model (`_seg` -> `_rootlets`) to prevent overwriting. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4672)
 - **sct_deepseg**: Fix rootlets QC for anisotropic images. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4722)
 - **sct_extract_metric**: Add more intuitive feedback when no metrics can be extracted from input data. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4606)
 - **sct_maths**: Fix incorrectly lazy-loaded `dipy` imports. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4638)
 - **sct_maths**: Fix dtype mismatch error when mixing integers and floating point numbers. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4660)
 - Add clearer instructions to fix issue when $SCT_DIR isn't writeable during installation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4646)
 - Fix bug due to discrepancy introduced by `set_qform` and `set_sform` methods. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4697)
 - Update warnings filter to properly filter `Private repos` warning from `requirements.parse`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4720)

**INSTALLATION**
 - Skip `acvl_utils==0.2.1` due to buggy interaction with `nnunetv2`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4688)
 - Install `monai[cucim]` extra when specifying GPU SCT `-g` flag. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4634)
 - Allow `nnunetv2>=2.5.1` due to bugfix for previous issue with `2.4.2`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4699)

**DOCUMENTATION**
 - **sct_analyze_lesion**: Add `lesion-analysis` tutorial. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4586)
 - **sct_deepseg**: Add spinal nerve rootlets segmentation tutorial. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4629)
 - **sct_deepseg**: Add example commands for the `-custom-url` arg. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4676)
 - **sct_deepseg**: Retire outdated DeepSeg models (`seg_sc_ms_lesion_stir_psir`, `ms_sc_mp2rage`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4707)
 - Address deprecation by manually performing previous RTD Sphinx context injection . [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4604)
 - Split "Command Line Tools" page into multiple individual pages (with markdown formatting for `argparse`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4625)
 - Move "Labeling Conventions" page to "Concepts", while preserving tutorial using `.. include::` directive. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4640)
 - Remove `/en/latest/` slugs from `spinalcordtoolbox.com` URLs. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4651)
 - Handle newly-broken links (retries, FSLWiki changes). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4657)
 - Apply minor documentation feedback from 2023 SCT course. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4678)
 - Add model gallery for current version of DeepSeg (`-task` syntax). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4684)
 - Fix hyperlink syntax in documentation (Markdown -> RST). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/commit/653c3607b23bc33887bc2ae2c033ba7aa66c370e)

**CI**
 - Skip `-h` duration test on macOS CI runners due to sporadic runtimes. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4605)
 - Allow error code 406 in broken link checker. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4610)
 - Be good netizens when checking broken links (HEAD request only, 30s retries, respect `Retry-All`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4649)
 - Bump oldest macOS version used by test runners (12 -> 13). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4675)

## 6.4 (2024-08-01)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/6.3...6.4)

**FEATURE**
 - **sct_analyze_lesion**: Automatically compute midsagittal tissue bridges. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4489)
 - **sct_analyze_lesion**: Support `CombinedLabels` from `info_label.txt` when computing lesion distributions with `-f`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4584)
 - **sct_analyze_lesion**: Add `-perslice` option (plus performance improvements to make it feasible). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4588)
 - **sct_compute_flow**: New function to compute velocity from VENC sequence. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4302)
 - **sct_deepseg**: Track `sct_deepseg` model provenance with `source.json` (in model folder) and JSON sidecar (in output). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4466)
 - **sct_deepseg**: Add postprocessing functionality for non-ivadomed models. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4482)
 - **sct_download_data**: Add T2w dog template to available data downloads. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4561)

**ENHANCEMENT**
 - **sct_deepseg**: Change `-install-task` arg to `-install`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4473)
 - **sct_deepseg**: Update contrast agnostic model to v2.4 (now improved for lumbar t2w + PSIR/STIR images). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4510)
 - **sct_deepseg**: Indicate model version in "model is up to date" log message. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4519)
 - **sct_deepseg**: Replace `seg_ms_lesion_mp2rage` model with a nnUnet based model on `sct_deepseg`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4554)
 - **sct_deepseg**: Fix rootlets QC by improving the cropping, centering, and colormap. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4555)
 - **sct_deepseg**: Set `edge` padding as default for contrast-agnostic monai model. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4576)
 - **sct_deepseg**: Update SCI model to SCIsegV2 and standardize on `nnUNetTrainer` folder structure. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4583)
 - **sct_maths**: Allow multiple `sct_maths` operations to be run sequentially on an image in one command. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4485)
 - **sct_maths**: Add `-volumewise` to process the individual 3D volumes of 4D input images. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4499)
 - Run all tests in temporary directories (to preserve a clean copy of `sct_testing_data`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4474)

**BUG**
 - **sct_analyze_lesion**: Fix crashing if there is no midsagittal lesion. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4595)
 - **sct_analyze_lesion**: Silence `pandas` warning by explicitly using `copy` to remove view/copy ambiguity [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4596)
 - **sct_analyze_lesion**: Fix slice numbering in QC and CLI output [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4597)
 - **sct_apply_transfo**: Make sure that `fname_out` is set properly for `fsleyes` command when `-o` is not passed. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4480)
 - **sct_deepseg**: Ensure that using `-thr 0` with the contrast agnostic model generates a useful softseg. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4508)
 - **sct_deepseg**: Avoid buggy version of `nnunetv2` that produces empty output predictions. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4564)
 - **sct_dmri_moco, sct_fmri_moco**: Output ANTs syntax in moco functions with `-v 2` for easier debugging. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4580)
 - **sct_extract_metric**: Ignore case where user selects 'combine=1' with CombinedLabels. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4469)
 - **sct_extract_metric**: Skip empty slicegroups to avoid creating blank rows in output metric CSV file. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4487)
 - **sct_flatten_sagittal**: Rescale flattened values from `[-1.0, 1.0]` back to their original range and datatype. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4573)
 - **sct_label_vertebrae, sct_register_to_template, sct_smooth_spinalcord**: Include the SCT version in the straightening.cache. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4537)
 - **sct_qc**: Fix tiny sagittal mosaic QC Report for large images with many slices. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4479)
 - **sct_qc**: Properly detect softsegs for QC when binary values are *not* the most common values.. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4494)
 - **sct_qc**: Align the lock names for `qc.py` and `qc2.py`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4541)
 - **sct_qc**: Adjust QC canvas size by physical resolution to avoid tiny QC with anisotropic T2* images. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4566)
 - **sct_qc, sct_run_batch**: Increase timeout of `mutex` to avoid `AlreadyLocked` errors with many concurrent processes. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4511)
 - **sct_register_to_template**: Add rotation to landmark-based preregistration (`step=0`) during template registration. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4533)
 - **sct_straighten_spinalcord**: Fix straightening transformations for images with "tilted" qform/sform. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4514)
 - Fix small argparse bug when wrapping help text with whitespace-only lines. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4472)
 - Add `matplotlib-inline` backend for compatibility with Jupyter notebooks. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4525)

**INSTALLATION**
 - Make `sct_download_data` errors non-fatal during installations of SCT. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4484)
 - Restrict `numpy<2` to avoid upstream errors. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4536)

**DOCUMENTATION**
 - **sct_analyze_lesion**: Improve documentation of the `-f` argument to make the percentages clearer. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4496)
 - **sct_analyze_lesion**: Improve formatting of bullet points in CLI help. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4585)
 - **sct_apply_transfo**: Rewrite help for `-x label` to clarify the difference between single-voxel and multi-voxel labels. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4498)
 - **sct_label_utils**: Clarify that `-disc` does not involve orthogonal projection. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4550)
 - **sct_run_batch**: Clarified how many subjects are processed in parallel. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4470)
 - `sct_label_utils`: Clarify argument order for `-disc`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4483)
 - Update url to slicer documentation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4486)
 - Update redirected citation link. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4559)

**REFACTORING**
 - **sct_deepseg, sct_deepseg_sc, sct_register_to_template**: Remove duplicate functions in `sct_deepseg` and `sct_register_to_template`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4546)
 - **sct_maths**: Minor improvements to PR #4485. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4557)
 - **sct_qc**: Refactor `qc.py` to use `create_qc_entry`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4574)
 - Use Tensorflow's simple `LazyLoader` class to minimize import times of expensive packages. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4493)
 - Simplify calls to `np.einsum`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4529)

**CI**
 - Update github action to Node.js 20. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4462)
 - Replace `libglib2.0-0` with `t64` version to fix Debian Rolling Release test failure. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4463)
 - Update CentOS Stream 8 runner to work past EOL. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4517)
 - Set `norecursedirs` in `setup.cfg` to avoid pytest collecting tests from our dependencies within our `venv`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4571)

## 6.3 (2024-04-25)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/6.2...6.3)

**FEATURE**
 - **sct_deepseg**: Add CanProCo-based MS lesion segmentation model. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4443)
 - **sct_deepseg**: Add EPI-BOLD fMRI spinal cord segmentation model. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4454)
 - **sct_deepseg**: Update `contrast-agnostic` SC segmentation model to the latest v2.3 version. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4451)
 - **sct_deepseg, sct_qc**: Add QC report for `sct_deepseg`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4446)

**ENHANCEMENT**
 - **sct_fmri_moco**: Switch to using mean magnitude for output `moco_params.tsv` file used for QC. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4396)
 - **sct_process_segmentation**: Add improvements to CSA calculation for GM/WM masks (`-angle-corr-centerline`, `float32` precision fix, doc warnings). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4409)
 - **sct_qc**: Add padding to the crop used for sagittal mosaic QC report. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4392)
 - **sct_qc**: Isolate `index.html`-writing code into reusable script + replace `Lock` with Semaphore-based `mutex`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4439)
 - **sct_qc**: Switch from two-tone colormap to transparency for `sct_deepseg` QC. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4460)

**BUG**
 - **sct_deepseg**: Address upstream breaking API change by renaming `_gpu` input param to `_device`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4381)
 - **sct_deepseg**: Avoid `nnunetv2=={2.4.0,2.4.1}` to mitigate upstream bug for `predict_single_npy_array`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4448)
 - **sct_qc**: Fix `-p sct_label_vertebrae` QC report when providing TotalSegmentator labels. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4383)
 - **sct_qc**: Fix detection of PSIR images for QC resampling to avoid unnecessary thresholding. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4414)
 - **sct_qc**: Keep CSS/JS/etc. assets up to date when QC report is regenerated with a newer version of SCT. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4427)
 - **sct_qc**: Avoid nibabel crashes due to `int64` arrays by explicitly passing `dtype`/`header` to `Nifti1Image`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4433)
 - **sct_run_batch**: Filter out color codes from log files. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4288)
 - Update QC assets when saving the QC report. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4432)
 - Replace remaining usages of `nib.save()` with `Image.save()` to mitigate scaling issues. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4416)

**INSTALLATION**
 - Pull upstream `ivadomed` changes that let us upgrade previously-pinned versions of `dipy`/`numpy`/`nibabel`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4332)
 - Address test warnings by pinning `pyqt5-sip<12.13.0` and updating `setup.cfg` ignore entries (`pkg_resources`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4355)
 - Pin to `dipy<1.6` until conflicts with `dipy>=1.8` can be resolved. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4398)

**DOCUMENTATION**
 - **sct_extract_metric**: Fixed path to PAM50 in help example command. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4385)
 - **sct_image**: Clarify argparse help for `-copy-header` option to provide caution about header/data mismatches. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4429)
 - Add links to 2024 review in prominent locations in documentation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4418)
 - Fix inaccurate line in docs regarding the method used for spinal level estimation (Frostell et al.). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4449)

**REFACTORING**
 - Restructure QC report code to use `create_qc_entry` context manager (plus 1 PoC: `multimodal`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4224)

**CI**
 - Update GitHub Actions versions (macOS runners, Windows runners, Node.js 16 actions -> 20). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4353)
 - Update GitHub Actions versions (Debian 9->11, CentOS 7->9, WSL 2022->2019) to fix CI failures. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4391)
 - Add CI workflow step to detect broken links in repository files. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4395)
 - Reduce concurrent macOS runners on PRs from 6 to 2 (and remove push-to-master triggers). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4402)
 - Replace `matrix.os` with `runner.os` to more robustly detect Win OSs in GHA workflows. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4403)
 - Move most broken link checking to a daily job. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4428)
 - Fix broken link checker. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4430)

## 6.2 (2024-02-15)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/6.1...6.2)

**FEATURE**
 - **sct_deepseg**: Integrate NNUnet/MONAI models into DeepSeg CLI (contrast-agnostic, SCI, rootlets, Zurich mouse). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4345)
 - **sct_qc**: Save QC records to browser local storage. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4317)
 - **sct_warp_template**: Update PAM50 link to include new `template/PAM50_rootlets.nii.gz` file. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4373)

**ENHANCEMENT**
 - Convert integer images to floating point when resampling with linear interpolation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4230)

**BUG**
 - **sct_analyze_lesion**: Set `minmax=False` to prevent cropping during angle correction, avoiding slice number mismatches. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4316)
 - **sct_compute_compression**: Fix reversed slice numbering (`S->I` => `I->S`) in output CSV file. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4326)
 - **sct_compute_mtsat**: Apply missing B1 correction to `r1map` and `a` calculations to fix MTsat bug. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4306)
 - **sct_deepseg**: Reverse spacings to match nnUNet's `SimpleITK`-based conventions. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4369)
 - **sct_deepseg, sct_label_vertebrae**: Mitigate scaling issues (`1.0` -> `0.999`) due to float/int datatype mismatches between header and array. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4346)
 - **sct_label_vertebrae**: Fix path splitting when space (`' '`) is present in `isct_spine_detect` model path. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4328)
 - **sct_register_to_template**: Ensure reorientation is performed consistently before any resampling. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4300)
 - **sct_register_to_template**: Fix straightening error during registration if 3+ labels are supplied and topmost disc label is not C1. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4347)

**INSTALLATION**
 - Temporary fix for dependency issue with dipy 1.8.0. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4321)
 - Fix broken pip installer by forcing it to detect macOS 11 as `11.0` and not `10.16`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4359)
 - Document Rosetta 2 as a requirement for installation on Apple silicon (M1, M2, M3). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4368)

**DOCUMENTATION**
 - **sct_compute_compression**: Update argparse descriptions to make it clear that the script can also be used to compute MCC. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4331)
 - **sct_label_utils**: Clarify the argparse help description for the `-disc` argument. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4325)
 - **sct_run_batch**: Add example syntax for YAML and JSON config files to the argparse help description for `exclude_list`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4304)
 - **sct_run_batch**: Add examples to `-include-list` and `-exclude-list` flags. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4361)
 - Add links to both the slide deck and YouTube video for the 2023 SCT Course. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4307)
 - Port changes from SCT Course 2023 Google Slides to the web tutorials. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4311)
 - Update `sphinx` and `furo` while removing "Edit on GitHub" hack in the process. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4341)

**REFACTORING**
 - Remove `utils` star imports and ensure we are importing directly from submodules `{fs,sys,shell}`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4289)
 - Remove unused `Image.verbose` attribute. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4292)
 - Replace assertions with more appropriate errors. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4294)

**CI**
 - Use the same Python version as SCT for linting. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4296)

## 6.1 (2023-11-03)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/6.0...6.1)

**FEATURE**
 - **sct_analyze_lesion**: Add function to output the axial damage ratio + minor improvements. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4207)
 - **sct_download_data**: Add current `sct_tutorial_data` release as a new course dataset. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4271)
 - **sct_download_data**: Update PAM50 template link to include cord and lumbar label changes. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4268)
 - **sct_run_batch**: Add start date information to the logs. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4218)

**ENHANCEMENT**
 - **sct_compute_compression**: Improvements of CLI output printed by the `sct_compute_compression`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4175)
 - **sct_deepseg**: Append `-list-tasks` to argparse help and update `-list-tasks-long`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4212)
 - **sct_label_utils**: Speed up `-remove-reference` by removing unnecessary iteration. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4194)
 - **sct_run_batch**: Raise error if `-config` file has wrong suffix. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4201)

**BUG**
 - **sct_compute_compression**: Use pandas for `.csv` saving to correctly merge existing output metric columns. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4172)
 - **sct_deepseg**: Add checks for empty arrays post-segmentation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4199)
 - **sct_deepseg**: Update model releases (`lumbar_seg`, `t2star_sc`) to fix output suffix (`_seg-manual` -> `_seg`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4235)
 - **sct_dmri_compute_dti**: Avoid `dipy` versions 1.6.0 + 1.7.0 that contain a `method=restore` bug. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4232)
 - **sct_dmri_display_bvecs**: Fix `ValueError` by passing a tuple to `color=` instead of a numpy array. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4221)
 - **sct_label_vertebrae**: Check whether the provided discfile is empty. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4200)
 - **sct_propseg**: Remove parent parts from `fname_out` to fix buggy `-o` behavior. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4170)
 - **sct_qc**: Fix QC for soft segmentations that have light regions close to `1.0` (e.g. `0.999`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4193)
 - **sct_register_to_template**: Raise exception if sform/qform don't match during registation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4211)
 - **sct_register_to_template**: Avoid duplicate orthogonal labels during registration by checking existing labels first. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4233)
 - **sct_register_to_template**: Switch matmul syntax (`*` -> `@`) to follow-up previous `np.matrix` -> `np.array` change. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4267)
 - **sct_straighten_spinalcord**: Fix `IndexError` during straightening if `-dest` image is shorter than centerline image (`-s`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4192)
 - Gracefully handle infinity as well as NaN values in QC. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4189)
 - Fix distorted registration due to straightening bug in `get_closest_to_absolute`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4206)
 - Temporarily pin `onnxruntime` to <1.16. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4226)
 - Avoid onnxruntime 1.16.0 but allow onnxruntime >=1.16.1. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4256)

**INSTALLATION**
 - Inline small `.yml` "model" files. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4171)
 - Update Docker installation instructions for Linux/macOS/Windows. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4196)

**DOCUMENTATION**
 - **sct_compute_compression**: Add tutorial for `sct_compute_compression` in documentation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4162)
 - **sct_compute_compression, sct_process_segmentation**: Add references for PAM50 normalized metrics. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4259)
 - **sct_download_data**: List `sct_download_data` datasets at the end of `-h`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4255)
 - **sct_download_data**: Color installed datasets for `sct_download_data -h`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4257)
 - **sct_run_batch**: Clarification of `sct_run_batch -script-args` input flag help. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4190)
 - **sct_run_batch**: Add note that `~` should not be used in paths passed using the `-script-args` arg. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4208)
 - Update installation docs for Windows (Miniconda) and Linux/macOS (standalone installer). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4060)
 - Migrate old ReadTheDocs settings to `.readthedocs.yaml`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4213)
 - Update documentation for new spinal levels. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4249)
 - Disable PDF and EPUB documentation builds. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4261)
 - Update figure and references for "Other shape metrics" tutorial. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4262)
 - Fix typo in "Other shape metrics" tutorial. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4273)
 - Add tutorial for lumbar segmentation and registration. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4250)

**REFACTORING**
 - **sct_analyze_lesion**: Simplify lesion volume computation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4219)
 - **sct_compute_compression**: Use the `pandas.DataFrame.combine_first` method. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4176)
 - **sct_warp_template**: Remove `-s` functionality and add a deprecation warning. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4252)

**CI**
 - Exit on errors anywhere in the linter script. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4161)
 - Update changelog PR workflow to fix milestone title error and to also change `version.txt`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4165)

## 6.0 (2023-07-14)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/5.8...6.0)

**FEATURE**
 - **sct_analyze_lesion**: Allow lesion volume calculation without providing SC seg. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4112)
 - **sct_compute_compression**: Add new CLI script to compute normalized metric ratios (MSCC, etc.) for compressed levels. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4003)
 - **sct_crop_image**: Add `-dilate` option to enable cropping around a spinal cord segmentation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4028)
 - **sct_deepseg**: Add support for model ensembles to `sct_deepseg`, update `mp2rage_lesion` model. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4042)
 - **sct_deepseg_lesion, sct_qc**: Add sagittal mosaic QC and use it for `sct_deepseg_lesion` QC report. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4102)
 - **sct_get_centerline**: Add `-centerline-soft` option to output a non-binary "soft" centerline. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4026)
 - **sct_image**: Replace `-setorient-data` with `-flip` and `-transpose` options. **WARNING: Breaks compatibility with previous version.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4071)
 - **sct_label_utils**: Add new `project_centerline` option in `sct_label_utils` to project an image on the spinal cord centerline . [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4080)
 - **sct_process_segmentation**: Bring metrics in PAM50 anatomical dimensions in `sct_process_segmentation`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3977)
 - **sct_process_segmentation**: Display vertebral levels when provided with `-perslice 1`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4004)
 - **sct_warp_template**: Add new `-histo` option to warp the newly-added PAM50 histology files. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3444)
 - **sct_register_to_template**: List resampled labels in order. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4088)

**ENHANCEMENT**
 - **sct_detect_pmj**: Add `-qc-dataset` and `-qc-subject` flags to `sct_detect_pmj.py`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4053)
 - **sct_image**: Add `-qc-subject` and `-qc-dataset` flags to `sct_image`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4052)
 - **sct_label_utils**: Update `labelize_from_discs()` to add level labels above the top disc and below the bottom disc. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4072)
 - **sct_maths**: Use `filters.rank` instead of `dilate`/`erode` for `uint8`/`uint16` images. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4021)
 - **sct_propseg**: Silence unnecessary "overwriting" messages when updating output image header. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4110)
 - Fix indentation for wrapped lines in argparse help. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4008)
 - Fix broken `tqdm` behavior for Windows and older Unix platforms. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4093)
 - Use `sct_progress_bar` for `algo=columnwise` and `register2d` slicewise loops. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4094)
 - Replace `algo=rigid` with `algo=slicereg` for mt0-mt1 registration. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4095)
 - Selectively display `relpath` or `abspath` in viewer syntax based on CWD. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4109)

**BUG**
 - **sct_analyze_texture**: Change `grey` --> `gray` in imports of `skimage.feature` functions. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4057)
 - **sct_compute_ernst_angle**: Fix upper/lower TR bounds used when plotting Ernst angles (argument `-b`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4131)
 - **sct_deepseg**: Clean up improper submodule access (via imported parent modules). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3973)
 - **sct_deepseg_lesion**: Fix broken `-v 2` case by making sure `fname_res_ctr` exists prior to resampling. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3987)
 - **sct_deepseg_sc**: Ensure that qform/sform codes are preserved when generating segmentation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4135)
 - **sct_get_centerline**: Prevent soft centerline being used with non fitseg methods. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4063)
 - **sct_get_centerline**: Add extrapolation for `sct_get_centerline` fitseg. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4079)
 - **sct_image**: Swap order of `sct_image -stitch` QC images to fix incorrect YML path. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4157)
 - **sct_label_utils**: Fix buggy `-create-viewer` behavior on initialization (undo crash, blank checkboxes). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4108)
 - **sct_plugin.py (FSLeyes)**: Catch errors from outdated `wxpython` versions bundled with FSL, then suggest a solution. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4048)
 - **sct_qc**: Faithfully display `sct_qc` arguments, rather than making a fake command. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4016)
 - **sct_qc**: Ensure that sagittal images (and softsegs) are resampled correctly for lesion QC. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4133)
 - **sct_qc**: Fix errors in the list of available QC processes (`index.html`, `sct_qc.py`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4151)
 - **sct_register_to_template**: Ensure that the `straightening.cache` file is output to the PWD during template registration. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4156)
 - **sct_resample**: Update `-mm` and `-ref` argument descriptions, fix `parse.error` logic, and add tests. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4081)
 - Pin `numpy<1.24` to mitigate incompatibility with `pystrum==0.2`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3981)
 - Better isolate the SCT conda env by using `-p` instead of `-n` during creation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3989)
 - Make matplotlib less verbose. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4089)
 - Use `importlib.metadata` to fetch list of scripts instead of `pkg_resources`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4124)

**INSTALLATION**
 - **sct_segment_graymatter**: Remove `gm_model` from `download.py` module and install scripts. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3995)
 - Replace `version.txt` contents (`dev`) with PEP440-compliant version (`6.0.dev0`). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3999)
 - Remove outdated/unused settings from `setup.cfg`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4000)
 - Update `version.txt` on master branch during release workflow. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4001)
 - Use Miniconda instead of built-in Python for Windows installations. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4027)
 - Clean up `.egg-info` and `python/` folders when reinstalling SCT. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4043)
 - Allow `install_sct` to be run standalone (without downloading "Source code" archive). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4049)
 - Add missing features from `install_sct` to `install_sct.bat`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4061)
 - Remove Sentry code. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4103)
 - Upgrade SCT's conda environment to Python 3.9. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4136)
 - `install_sct`: Use explicit paths to python/pip executables. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4143)
 - `install_sct`: Install `libffi` from `conda-forge`, rather than `defaults`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4144)
 - Make sure that `$SCT_DIR` is set for both Windows installations and `batch_processing.sh`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4150)

**DOCUMENTATION**
 - **sct_deepseg**: Move MP2RAGE cropping comment from "model" description to "task" description. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4065)
 - **sct_deepseg_lesion**: Update description of `-file_centerline` flag to mention `-centerline file` instead of `-centerline manual`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4018)
 - **sct_register_multimodal**: Add an example `-param` configuration to the argparse help description of `algo=dl`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4111)
 - Improve the readability of temporary directory names throughout SCT. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3986)
 - Add link to mailing list. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4014)
 - Improve the appearance of `spinalcordtoolbox.com` in search results. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4069)
 - Standardize Python file headers for copyright and license information. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4128)
 - Encode emoji in utf-8. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4141)
 - Clarify documentation for `compute_vertebral_distribution`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4154)

**REFACTORING**
 - **sct_plugin.py (FSLeyes)**: Fix `flake8` linting issues in the `sct_plugin.py` script. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4074)
 - **sct_plugin.py (FSLeyes)**: Rename FSLeyes script from `sct_plugin.py` to `sct_fsleyes_script.py`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4092)
 - **sct_register_multimodal**: Simplify weight-loading model code for `algo='dl'`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3996)
 - Refactor `get_centerline` function to distinguish between OptiC and non-OptiC methods. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3975)
 - Remove `_get_centerline()` and put functionality into `centerline/core.py`'s `get_centerline()`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4010)
 - Remove `round_and_clip` function from `centerline/core.py`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4031)
 - Add `im_type` argument in `display_viewer_syntax` to facilitate colormap selection. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4037)
 - Fix flake8 lints. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4041)
 - Remove each instance of the `-igt` argument from SCT scripts. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4068)
 - Fix `flake8` linting issues in the `testing/` directory. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4073)
 - Code cleanup in `types.py`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4075)
 - Refactor `get_dimension`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4097)
 - Refactor `find_and_sort_coord`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4098)
 - Delete `create_test_data.py` and move `dummy_` functions into their respective test files. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4115)
 - Refactor `reports/qc.py`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4126)
 - Remove unused QC report assets. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4129)

**CI**
 - List changed files for flake8. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3963)
 - Improve `release-body.md` contents to address feedback from v5.8 release. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3985)
 - Use milestone due date. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3990)
 - Add shellcheck to CI (and fixup shell scripts to get the check to pass). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4015)
 - Attach `install_sct` as an asset during release creation CI workflow. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4055)
 - Remove all 70,000+ warnings from test suite. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4091)
 - Fix Debian 9 CI failures by adding `archive` links to `/etc/apt/sources.list`. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4106)
 - Update outdated GitHub Actions to versions that use Node.js 16. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4107)
 - Ensure CI avoids older ReadTheDocs docker images to address OpenSSL mismatch. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4114)
 - Test standalone installations via `tests.yml`, but only on the `master` branch. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4121)
 - Add warning filters for upstream `pkg_resources` warnings. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4127)

**GIT/GITHUB**
 - Update PR template links to testing and documentation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/4033)

## 5.8 (2022-12-02)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/5.7...5.8)

**FEATURE**

 - **sct_image:** Add new `-stitch` option that wraps stitching functionality from Glocker et al..  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3865)
 - **sct_process_segmentation:** Add DistancePMJ when perslice flag in `sct_process_segmentation`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3856)
 - **sct_run_batch:** Add `-ignore-ses` flag to process `sub-` directories even when `ses-` subdirectories are present.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3862)

**ENHANCEMENT**

 - Update header `dtype` property on save/load to match the datatype of the data array.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3944)
 - **sct_analyze_lesion, sct_analyze_texture, sct_denoising_onlm:** Enforce use of `display_viewer_syntax` in four scripts using hardcoded `fsleyes` commands.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3895)
 - **sct_get_centerline:** Ensure that the output of `get_centerline()` can be saved using `-r 0`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3960)
 - **sct_image:** Update `binaries_linux` to include a rebuilt version of `stitching` that targets `centos7`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3939)
 - **sct_label_vertebrae, sct_qc:** Add readability fixes for QC reports (sagittal view scaling, label text, label colormaps).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3889)
 - **sct_register_multimodal:** Add dimensions of data to registration logging.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3899)
 - **sct_register_multimodal:** Minimize memory usage for `algo=dl` (and add a warning for potential OOM killer issues).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3953)
 - **sct_register_to_template:** Throw error when labels are lost during the straightening transform.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3880)

**BUG**

 - Properly escape `\` in the regex for removing `sct_env` from old RC files.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3872)
 - Ensure that calling `printv(type='error')` actually exits the program.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3874)
 - Reduce logging from intermediate CLI scripts by setting `-v 0` in `main()` calls.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3881)
 - Address `[Errno 24] Too many open files` during motion correction.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3882)
 - Fix several bugs when loading headers in the `Image` constructor.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3931)
 - Set a more permissive `quaternion_threshold` inside `Image` constructor.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3936)
 - **sct_analyze_lesion:** Fix output filename spelling.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3909)
 - **sct_check_dependencies:**  Re-add `print_line` call for PropSeg check.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3911)
 - **sct_extract_metric:** Ensure that `-list-labels` is the last argument that is parsed.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3864)
 - **sct_get_centerline:** Exclude original file extension (e.g. `.nii.gz`) from the output `.csv` filename.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3917)
 - **sct_merge_images, sct_smooth_spinalcord:** Enforce `type=int` for arguments that use `choices=(0, 1)`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3887)
 - **sct_propseg, sct_straighten_spinalcord:** Pass `argv` to `generate_qc` functions (instead of `arguments` or `sys.argv[1:]`).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3863)
 - **sct_register_multimodal:** Improve handling of case where `algo` is specified alongside `type='label'`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3894)
 - **sct_run_batch:** Modify `-include-list` and `-exclude-list` to check against parts of a directory, too.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3859)
 - **sct_run_batch:** Fix `include_list`/`exclude` argument ordering in call to `_filter_directories`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3870)
 - **sct_run_batch:** Allow `path_output` parameter to start with `~`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3958)
 - **install_sct.bat:** Use delayed expansion (`!!`) for `git_ref`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3891)

**INSTALLATION**

 - Upgrade SCT from Python 3.7 to Python 3.8.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3916)
 - Limit `pyqt5` to >5.11 and <5.15 to avoid "Cannot load library libqxcb.so" error on Ubuntu.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3928)
 - Ensure bundled GLIBC is up-to-date in order to resolve `MESA-LOADER` errors.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3929)
 - Add more thorough check for `git` within the `PATH` for the Windows installer.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3945)

**DOCUMENTATION**

 - Remove API pages from the documentation.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3869)
 - Fix copy-paste typos in tutorial documentation.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3886)
 - Add Google Sheets summary graphs to `studies.rst`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3896)
 - Switch to higher-quality "Tools used" graph on the studies page.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3906)
 - Update outdated links to `atlas` and `spinal_level` scripts in `pam50.rst`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3923)
 - Clarify strengths of both `sct_propseg` and `sct_deepseg` in the documentation.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3941)
 - Fix typo in "Registration" tutorial (`slice-by-slide` -> `slice-by-slice`).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3954)
 - **sct_deepseg_lesion:** Add missing spaces in multiline strings, and missing commas in lists.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3933)
 - **sct_image:** `sct_image`: Clarify descriptions for `-setorient` commands.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3938)
 - **sct_maths:** Fix small typo in `-uthr` help description (`below` -> `above`).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3922)
 - **sct_process_segmentation:** Add references to Bédard and Cohen-Adad.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3934)
 - **sct_register_multimodal:** Add scaling to `sct_register_multimodal -dof` argument help.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3948)

**REFACTORING**

 - Replace `run_proc` calls with `main` calls across SCT's scripts.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3724)
 - Replace `argv=None` default by `argv: Sequence[str]` type hint.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3866)
 - Simplify `add_suffix`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3878)
 - Move 5 unused scripts to `spinalcordtoolbox-dev`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3892)
 - Remove unused `isct_` scripts from SCT's CLI and test suite.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3908)
 - **sct_check_dependencies:** Convert `isct_test_ants` script into a `pytest` test, then call via `sct_check_dependencies`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3932)
 - **sct_compute_snr, sct_dmri_moco, sct_fmri_moco, sct_maths, sct_process_segmentation:** Call `parser.error` properly.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3858)
 - **sct_dmri_denoise_patch2self:** Deal with `nibabel` deprecation of `get_header` and `get_data`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3879)
 - **sct_download_data:** Refactor data downloading API to remove dependency on CLI script.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3907)
 - **sct_merge_images:** Tidy up PEP8 warnings and other small style issues.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3930)
 - **sct_process_segmentation:** Improve handling of default arguments.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3860)
 - **math.py:** Address `selem` deprecation by using `footprint`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3898)
 - **shell.py:** Let `Metavar` inherit from `str` to silence PyCharm type warning.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3876)

**CI**

 - Upgrade Ubuntu 18.04 -> 20.04/22.04 in test suite.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3919)
 - Set `draft: true` in the release creation workflow.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3940)
 - Pin `flake8<6.0.0` in our linting workflow.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3962)

## 5.7 (2022-07-28)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/5.6...5.7)

**FEATURE**

 - **sct_process_segmentation,sct_extract_metric:** Combine conditions when slice number and vertebral levels are both specified.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3823)
 - **sct_register_to_template,sct_register_multimodal:** Integration of DL multimodal registration in SCT.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3807)
 - Allow in-place installations (to support PRs from forks).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3797)

**CI**

 - Update GitHub Actions runners to match current (non-beta) up-to-date versions.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3853)
 - Add `install_sct.bat` as a release asset during `create-release.yml` workflow.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3850)
 - Remove coverage reporting via `codecov`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3837)
 - Refactor release workflow into 3 stages to properly test PR branches.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3836)
 - Exclude `certifi @ file` from pip freeze to work around upstream conda bug.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3819)
 - Expand `batch_processing.sh` test to support macOS and Windows.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3770)

**BUG**

 - **sct_propseg:** Prevent from overwriting files by outputting to a tempdir prior to renaming.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3847)
 - **sct_maths:** Swap `if` for `elif` to prevent error from being thrown when calling `sct_maths -mi`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3846)
 - **sct_merge_images:** Remove catch-all exception handling.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3839)
 - **image.py:** Use `shutil.copyfile` for output files if src/dest are on different filesystems.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3833)
 - **sct_process_segmentation:** Move CSA normalization models to `./data/` at the root of the SCT repo.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3821)
 - **sct_maths:** Rewrite `-add`/`-sub`/`-mul`/`-div` to match expected behavior for 3D/4D images.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3808)
 - **image.py:** Use absolutepath when loading images.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3792)

**INSTALLATION**

 - Repair automated commenting out of pre-4.0 sct settings.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3805)
 - **requirements.txt:** Pin `protobuf` to fix upstream Keras issue.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3802)
 - Replace pushd/popd with cd-in-a-subshell.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3796)
 - **requirements.txt:** Use CPU `--extra-index-url` to remove strict dependency pinning for `torch`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3790)
 - **sct_download_data:** Use binaries that are packaged by the `spinalcordtoolbox-binaries` repo.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2668)

**DOCUMENTATION**

 - **sct_merge_images:** Clarify argparse help description.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3840)
 - Add new tutorial for contrast agnostic registration.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3831)
 - Update installation docs to make 'checkout' step less confusing.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3825)

**ENHANCEMENT**

 - **sct_check_dependencies:** Clean up FSLeyes version checking to be less verbose/confusing.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3820)
 - **fslhd:** Account for `None` orientation strings when printing image headers.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3815)
 - **sct_deepseg_sc,sct_deepseg_gm,sct_deepseg_lesion:** Replace Tensorflow/Keras-based inference (`.h5`) with onnxruntime (`.ONNX`).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3738)

**REFACTORING**

 - **utils.py:** Unify calls to `shutil.move` by using only `utils.fs.mv`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3848)
 - Reduce flake8 warnings throughout the codebase.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3842)
 - **sct_label_vertebrae:** Remove deprecation warning for previously-removed `gaussian_std` argument.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3841)
 - Fix executable file permissions and remove unnecessary header declarations.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3835)
 - **aggregate_slicewise.py:** Update docstring for the `aggregate_per_slice_or_level` function.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3830)
 - **sct_register_to_template,sct_register_multimodal:** Fix circular and star importing in registration CLI scripts.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3811)

**GIT/GITHUB**

 - Delete the entire `dev/` folder from the `master` branch.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3827)
 - **create-release.yml:** Release directly from the `master` branch.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3813)

## 5.6 (2022-04-29)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/5.5...5.6)

**FEATURE**

 - **sct_deepseg:** Add model for T2w lumbar SC segmentation.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3766)
 - Add new native Windows install script and GitHub Actions CI runner.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3701)

**CI**

 - Add temporary skip for hardware-specific GitHub Actions runner issue.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3688)
 - Test release path in CI.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3678)
 - Replace CentOS 8 with CentOS Stream 8 to address December 2021 EOL.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3676)

**BUG**

 - **sct_deepseg:** Rewrite ANSI color code snippets to support terminals limited to 16-colors.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3782)
 - **sct_label_vertebrae:** Output a better error message when the initial disc is invalid.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3775)
 - Set a more permissive threshold for reading the qform.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3771)
 - **sct_check_dependencies:** Skip checking `isct_propseg` on Windows.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3753)
 - **sct_analyze_texture:** Stop using a tempfile for data reorientation.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3727)
 - Open CSV files using `newline=''` to fix `\r\r\n` issue.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3726)
 - **sct_run_batch:** Check `isdir` directly, rather than trying to catch `IsADirectoryError`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3725)
 - **sct_dice_coefficient:** Update `CMakeLists.txt` to include bugfix for unresolved external symbol errors.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3723)
 - **sct_run_batch:** Account for non-UNIX platforms in `sys.platform` checks.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3722)
 - Replace hardcoded `/tmp` directory with `tmp_path`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3721)
 - Clean-up Windows-incompatible hardcoded `/` path separators.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3720)
 - **sct_run_batch:** Add better support for shell script batch processing on Windows.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3719)
 - Improve how image overwriting is handled for memory-mapped data arrays on Windows.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3718)
 - **sct_propseg:** Attempt to fixup isct_propseg build (but ultimately skip sct_propseg on Windows).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3717)
 - **sct_label_vertebrae:** Refactor `-param` parsing to address several bugs.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3700)
 - **sct_propseg:** Fix parsing of `-d` argument in script.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3687)
 - **sct_analyze_lesion:** Fix computation of estimated lesion length and diameter.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3681)
 - Rewrite QC test to use `Pool.close()` to avoid stalling with pytest-cov.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3680)
 - **sct_get_centerline:** Stop appending `.nii.gz` to the centerline output filepath if `-o` has an extension.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3677)

**INSTALLATION**

 - Suppress output of `where deactivate` check.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3781)
 - Allow `install_sct.bat` file execution from double click.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3778)
 - Isolate SCT script launchers from the `venv_sct/Scripts/` directory on Windows.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3756)
 - Add `msvc-runtime` to our requirements to avoid DLL load error on Windows.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3752)
 - `install_sct.bat` Only deactivate if script is available on PATH.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3750)
 - `install_sct.bat`: Use `requirements-freeze.txt` if present.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3749)
 - `install_sct.bat`: Add a default value for `git ref`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3748)
 - `install_sct.bat`: Allow overwriting of existing spinalcordtoolbox installations.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3747)
 - `install_sct.bat`: Use a non-admin way of adding SCT to the PATH.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3746)
 - `requirements.txt`: Update `-f` link for PyTorch CPU HTML page.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3679)
 - Revert "requirements.txt -> setup.py" PR to restore old dependency-checking behavior.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3671)

**DOCUMENTATION**

 - Fix inaccurate comment in `batch_processing.sh`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3773)
 - Update documentation build instructions.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3734)
 - Fix broken links to neuro.polymtl.ca.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3690)
 - Add links to the newly-updated 2021 SCT Course material.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3689)
 - Add new "Pipelines" page to the sidebar of SCT's RTD documentation.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3668)
 - Split "SCT Courses" and "Tutorials" into two separate documentation pages.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3666)
 - **sct_deepseg:** Add new option `-list-tasks-long` to print in-depth descriptions of deepseg tasks.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3541)

**ENHANCEMENT**

 - Cosmetic fixes on `-list-tasks`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3777)
 - Add support for ITK-Snap + multiple viewers to `display_viewer_syntax`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3754)
 - **sct_label_vertebrae:** Improve the label cleaning function.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3739)

**REFACTORING**

 - **sct_label_vertebrae:** Code cleanup.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3774)
 - Refactor `sys.platform` checks to include Windows and use `.startswith()` idiom.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3757)
 - **sct_label_vertebrae:** Refactor argument parsing code.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3731)
 - Rename and move CLI tests from `testing/api/` to `testing/cli/`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3729)

## 5.5 (2022-01-26)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/5.4...5.5)

**FEATURE**

 - **sct_deepseg:** Add models for MP2RAGE SC and MS lesion segmentation.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3637)
 - **sct_testing:** Bring back previously-removed command as a light wrapper for `pytest`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3605)
 - **sct_deepseg:** Add spinal cord/gray matter multiclass segmentation model for 7T data.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3527)
 - **sct_dmri_denoise_patch2self:** Add new Patch2Self CLI script for dMRI denoising.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3510)

**CI**

 - Switch to 'informational mode' to prevent Codecov failures for small changes.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3655)
 - `tests.yml`: Update runners to no longer use Ubuntu 16.04.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3538)
 - Add coverage via `codecov`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3463)

**BUG**

 - Add utility function to strip '.py' extensions.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3658)
 - **sct_maths:** Refactor `-symmetrize` function to fix dimension bug.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3657)
 - **sct_compute_mtr:** Convert input file to float32 before computing MTR.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3638)
 - **sct_get_centerline:** Make sure that the QC report displays properly.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3626)
 - Set `channel_axis=None` for `pyramid_expand` to fix bug in `skimage==0.19.0`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3616)
 - **sct_propseg:** Fix argument parsing to ensure only the requested files are output.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3611)
 - Restore the argparse linewrapping behavior we had prior to Python 3.7.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3604)
 - **sct_download_data:** Update outdated mirror links.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3596)
 - Upgrade QC reports to use Qt5.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3588)
 - **sct_resample:** Make sure dimensions are correct for 4D images resampled with `-mm`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3553)
 - **sct_qc:** Fix colormap for vertebral labeling in the QC report.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3534)

**INSTALLATION**

 - Allow `install_sct` to run from any directory.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3660)
 - Use `--ignore-installed` to preserve the version of `certifi` that gets installed during `conda create`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3618)
 - Use a more reliable way to disable USER_SITE.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3606)
 - `requirements.txt`: Pin `onnxruntime>=1.7.0` rather than `==1.4.0`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3595)
 - `install_sct`: Skip `pip==21.2` to avoid suboptimal dependency resolver.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3594)
 - `install_sct`: Replace `sed -i` with `perl -pi -e`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3591)
 - `install_sct`: Add `touch` fixes to address WSL connection issues.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3574)
 - Fix multiple WSL issues related to installation and display.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3573)
 - Check CPU attributes to determine whether to install an AVX-less version of TensorFlow (M1 Macs, older Linux CPUs).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3569)
 - Remove `futures` from our `requirements.txt` file.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3548)
 - Use Python 3.7 instead of Python 3.6 for SCT's conda environment.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3361)

**DOCUMENTATION**

 - Add studies citing SCT.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3656)
 - Replace `sct_maths` with `sct_separate_dmri_separate_dwi_and_b0` in tutorial dMRI preprocessing.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3643)
 - Help differenciate Linux and WSL user on install documentation.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3641)
 - Add Uhrenholt et al.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3639)
 - Use furo-theme sanctioned dark mode colour controls.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3632)
 - Restore "Edit on GitHub" button.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3624)
 - Replace SCT Course 2020 iCloud link with Google Drive link.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3623)
 - **sct_deepseg_sc:** Indicate default value for `-brain`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3617)
 - **sct_register_to_template:** Update help for 2+ labels usage, and change link from iCloud to RTD.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3603)
 - Fix typo in `studies.rst` (Kinawy -> Kinany).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3589)
 - Remove misleading `FSLeyes` installation instructions.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3577)
 - `README.rst`: Fix typo.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3576)
 - `windows.rst`: Recommend echoing DISPLAY to ~/.profile, not .profile.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3563)
 - Remove leftover unnecessary `R|` formatting.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3561)
 - Add Hernandez et al.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3557)
 - Add Staud et al.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3551)
 - Update testimonials and small documentation fixup.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3544)
 - Add studies.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3543)
 - Add PMJ-based CSA tutorial.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3542)
 - Update studies.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3535)
 - `fsleyes.rst`: Fix typo (Installating -> Installing).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3526)
 - Add tutorials for the remaining material from the SCT Course.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3493)

**ENHANCEMENT**

 - **sct_download_data:** Add default output folders for dataset downloads.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3650)

**REFACTORING**

 - Remove `channel=None` bugfix that was superseded by upstream `skimage=0.19.1` patch.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3652)

**GIT/GITHUB**

 - Update broken link in PULL_REQUEST_TEMPLATE.md.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3644)
 - Clean up `install/`, `issues/`, and `flirtsch/` folders in repo.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3461)

## 5.4 (2021-09-24)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/5.3.0...5.4)

**FEATURE**

 - **sct_process_segmentation:** Add CSA normalization in `sct_process_segmentation`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3515)
 - **sct_register_multimodal:** Add `-owarpinv` to `sct_register_multimodal`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3507)
 - **sct_maths:** Introduce `-uthr` (upper threshold) to complement `-thr` (lower threshold) in sct_maths.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3499)
 - **sct_process_segmentation:** Add PMJ-based CSA in `batch_processing.sh`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3497)
 - **sct_compute_snr:** Implement new method to compute SNR on a single 3D volume.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3479)
 - **sct_process_segmentation:** Measure CSA based on distance from pontomedullary junction (PMJ).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3478)
 - **sct_process_segmentation,sct_qc:** Add QC report for `sct_process_segmentation` for PMJ-based CSA.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3465)
 - **sct_compute_snr:** Add `-o` argument to output SNR value to a text file.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3455)
 - **sct_qc:** Display soft segmentation in qc report.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3418)
 - **sct_run_batch:** Make it possible to loop across "ses-" entity.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3414)
 - **sct_dmri_display_bvecs:** Allow `sct_dmri_display_bvecs.py` to display multi-shell acquisition.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3390)

**CI**

 - Upgrade Ubuntu versions in test suite.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3477)
 - Modify `lint_code.yml` to try to make it work for PRs from forks..  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3474)
 - Improve the readability of the test suite.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3423)
 - Add `workflow_dispatch` for test suite workflows.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3399)
 - Add GitHub Actions workflow to publish a new release.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3383)

**BUG**

 - **sct_check_dependencies:** Import `pyplot` before `PyQt` to mitigate a finicky `libgcc_s.so.1` error.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3512)
 - **sct_smooth_spinalcord:** Fix the output files of `sct_smooth_spinalcord` (`smooth.nii.gz`, `straight_ref.nii.gz`).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3506)
 - Fix-up broken test for `sct_compute_snr`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3472)
 - **sct_register_to_template,sct_register_multimodal:** Apply softmask workaround to `slicereg` algorithm.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3446)
 - **sct_qc:** Fix display of PMJ in QC report.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3442)
 - **sct_qc:** Check if input segmentation is binary for QC report.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3437)
 - **sct_plugin.py (FSLeyes):** Fix FSLeyes script to make it compatible with both FSLeyes v0.34 and v1.0.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3427)
 - Don't set the global loglevel when CLI scripts are called in-code.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3410)
 - **sct_register_multimodal:** Don't output `dest->src` files if registration is only performed one-way.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3407)
 - **sct_propseg:** Fix `rescale_header is not 1` because the default value is `1.0`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3405)
 - **sct_process_segmentation:** Prevent metric calculation for empty slices by checking if array is ~0.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3403)
 - **sct_label_utils:** Ensure a copy of the header is used in `zeros_like` to fix `-create-viewer` bug.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3398)
 - **sct_dmri_display_bvecs:** Add test for `sct_dmri_display_bvecs`, then add `-v` argument to make the test pass.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3387)
 - **sct_dmri_moco:** Fix `-bvals` filepath handling and update bvals argparse help.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3376)
 - **sct_resample:** Make sure TR parameter isn't lost when resampling 4D images.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3370)
 - **sct_apply_transfo:** Change `sct_apply_transfo -v 0` to match `isct_antsApplyTransform` output.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3368)

**INSTALLATION**

 - **sct_check_dependencies:** Don't crash on non-ImportError exceptions when checking dependencies.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3482)
 - Stop trying to detect headless systems using `DISPLAY`, and no longer set `MPLBACKEND`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3420)
 - Throw error during installation if `$SCT_DIR` contains spaces.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3366)

**DOCUMENTATION**

 - Update references.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3521)
 - Fix inconsistent description on `studies.rst` page.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3520)
 - Update references.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3517)
 - Replace blue/gray SCT logo images with transparent logo image.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3514)
 - Add studies.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3504)
 - Reformatting the references for SCT docs.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3502)
 - Fix hover colour on sidebar of docs.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3498)
 - Fix CSS colour formatting bug with new Furo theme.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3489)
 - **sct_detect_pmj:** Clarified usage in PMJ detection script.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3484)
 - Add reference Vallotton et al..  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3464)
 - Fix typo (`sct_deepsec_sc` -> `sct_deepseg_sc`).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3452)
 - Remove misleading info about unsupported installation/usage methods (pip, API).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3447)
 - **sct_denoising_onlm:** Added bibliography reference.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3425)
 - Fix missing words in testimonials.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3393)
 - Add dark red formatting for argparse errors.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3363)
 - Remove instructions for installing FSLeyes in the SCT conda environment.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3360)
 - New Theme for Docs.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3355)
 - **sct_register_to_template,sct_register_multimodal:** Add multiple tutorial pages covering the "Registration" material from the SCT course.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3094)

**ENHANCEMENT**

 - Skip redownloading `sct_testing_data/` if the directory already exists.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3480)
 - **sct_detect_pmj:** Removal of scary message for user by changing certain function call parameters.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3473)
 - **sct_detect_pmj:** Improve R-L placement in `sct_detect_pmj`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3468)
 - **sct_qc:** Add QC field in .yml list for `sct_detect_pmj`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3448)
 - Fix table size and scrolling in the QC report.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3389)

**REFACTORING**

 - Update outdated information + PEP8 fixes in `setup.py`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3460)
 - Clean up scattered/broken tests and move into a single folder.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3449)
 - Test `batch_processing.sh` values by indexing column name, not number.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3436)
 - Modify `zeros_like` to no longer make a copy, and to avoid rescaling.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3379)
 - **sct_testing:** Replace all 45 `sct_testing` tests with Pytest equivalents.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3373)
 - Remove generate_path from methods in Image class.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3371)

**GIT/GITHUB**

 - Replace `neuropoly/spinalcordtoolbox` with `spinalcordtoolbox/spinalcordtoolbox`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3381)

## 5.3.0 (2021-04-25)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/5.2.0...5.3.0)

**FEATURE**

 - **sct_image:** Handle affine matrix mismatches better by exposing `-set-qform-to-sform`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3332)
 - **sct_extract_metric:** Implemented weighted median.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3329)
 - **sct_dmri_moco,sct_fmri_moco,sct_qc:** Implement QC for sct_dmri_moco and sct_fmri_moco.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3320)
 - **sct_image:** Add flag to `sct_image` to print image headers.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3317)
 - **sct_download_data:** Add entry to download exvivo template.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3310)
 - **sct_register_to_template:** Added flag -s-template-id to use another segmentation (e.g. white matter) for registration to template.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3299)
 - **sct_qc:** Interactive QC assessment: Add Pass/Fail/Artifact and download YAML file.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3253)

**CI**

 - Remove remaining Travis-related files and replace badge.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3345)
 - Use 'extend-ignore' in flake8 config to preserve defaults.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3255)
 - Run and test batch_processing.sh using GitHub Actions.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3198)
 - Port CI to Github Actions.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3125)

**BUG**

 - **sct_run_batch:** Fully isolate the conda env by its site.py.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3339)
 - **sct_image:** Various fixes to sct_image -display-warp.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3337)
 - **sct_image:** Fix faulty check for `arguments.set_sform_to_qform`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3331)
 - **sct_deepseg:** Update model and fixed default output suffix.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3321)
 - Update outdated `sct_register_graymatter` command in `batch_processing.sh`.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3316)
 - **sct_run_batch:** Unset `PYTHONNOUSERSITE` in environment before calling batch script.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3313)
 - **sct_register_to_template:** Fixed right-left flip if template is not RPI and other minor improvements.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3305)
 - Use less strict value for `rel_tolerance` in the `batch_processing.sh` test.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3293)
 - **sct_deepseg:** Unpin ivadomed to get latest version and fix wrong q/sform_code output.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3291)
 - Replace instances of 'sct_convert.convert' with 'image.convert'.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3288)
 - **sct_label_vertebrae,sct_warp_template:** Replace ANTs binary call with sct_apply_transfo call to properly set sform.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3272)
 - **sct_create_mask:** Create 2d masks in memory instead of via intermediate files.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3270)
 - **sct_label_vertebrae:** Handle 'label_discs' case where SC segmentation has holes/discontinuities.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3260)
 - **sct_dice_coefficient:** Uniquely distinguish filenames for tmp files to prevent overwriting.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3258)
 - Decrease the sensitivity of the sform/qform mismatch check.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3257)

**DOCUMENTATION**

 - Added testimonials; improvements on the documentation.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3349)
 - Add ref Savini et al.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3348)
 - Add ref Zhang et al.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3343)
 - Add ref Tinnermann et al..  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3342)
 - Drop `yes |` from the docs.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3307)
 - Add ref Lee et al.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3287)
 - Add reference Azzarito et al.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3259)
 - Remove TODO from Introduction docs.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3247)

**ENHANCEMENT**

 - Update moco commands in `batch_processing.sh` to include QC.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3351)

**GIT/GITHUB**

 - Add GH Actions workflow that automates Changelog PR.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3264)
 - Add checklist item to pull request template for 'Milestone' label.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3263)

## 5.2.0 (2021-02-24)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/5.1.0...5.2.0)

**FEATURE**

 - **sct_deepseg:** New segmentation model: GM and WM for exvivo DWI data (University of Queensland).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3224)

**BUG**

 - **sct_register_to_template:** Enforce UINT8 when resampling labels for register to template.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3230)
 - Fix size calculation bug for '-method map' and '-method ml'.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3217)
 - Fix recently-introduced faulty slice index comparison.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3203)
 - Fix unit_testing/test_labels.py::test_remove_missing_labels.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3191)

**DOCUMENTATION**

 - Add reference Ost et al..  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3243)
 - Add reference Johnson.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3241)
 - Move course materials to "Tutorials" page for visibility.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3240)
 - Update Installation section in docs for MacOS Big Sur, add section for FSLeyes.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3228)
 - Add ref solanes.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3221)
 - Update FSLeyes install instructions for Win10/WSL.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3210)
 - Update references.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3204)
 - Fix line breaks in documentation.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3197)
 - **sct_extract_metric,sct_process_segmentation:** Validate '-vertfile' dimensions in aggregate_per_slice_or_level().  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3120)

**ENHANCEMENT**

 - Modify tests to clean up after themselves in the working directory.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3222)
 - Remove xdist from requirements and pytest config.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3212)
 - **sct_image,sct_straighten_spinalcord:** Check qform and sforms match first.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2858)

**REFACTORING**

 - **sct_image:** Refactor sct_image functions to accept Image objects as input .  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3208)
 - **sct_maths:** Refactor sct_maths callers to remove subprocess and use API.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2975)

**GIT/GITHUB**

 - Lint pull requests using flake8 GH Actions workflow.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3215)
 - Add config that links to SCT forum and removes blank issues.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3151)

## 5.1.0 (2021-01-22)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/5.0.1...5.1.0)

**FEATURE**

 - **sct_concat_transfo:** Restore previously deprecated sct_concat_transfo.py.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3181)
 - **sct_deepseg:** Support of multichannel and multiclass models.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3041)
 - **sct_label_utils:** Add function to detect missing label.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2992)

**DOCUMENTATION-INTERNAL**

 - Add script to automate requirements-freeze.txt generation.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3042)

**BUG**

 - **sct_compute_hausdorff_distance,sct_dmri_compute_dti:** Fix bugs introduced by recent init step refactoring pull request.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3177)
 - **sct_propseg:** Replace os.path.dirname with pathlib.Path().parent.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3166)
 - **sct_deepseg,sct_run_batch:** Make sure (most) scripts return error code and print usage when no args are passed.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3156)
 - **sct_label_utils:** Add new '-create-seg-mid' option to replace bugged '-create-seg -1' behavior.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3148)
 - Replace "argv if argv else '--help'" behavior with subclassed ArgumentParser.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3139)
 - Replace troublesome unicode quote characters with more friendly ones.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3135)
 - Fixup realpath polyfill.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3123)
 - Fix incorrect indexing in get_center_spit to prevent QC report cropping.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3116)
 - Forces output label image to be UINT8 with -create-viewer.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3104)
 - Check that conda actually activated during install..  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3103)
 - **sct_label_vertebrae:** Change Error type so program doesn't quit when labels are too high.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3073)
 - Set PYTHONNOUSERSITE=True to prevent user site packages from interfering.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3070)
 - Bump ivadomed version to 2.5.0.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3034)
 - **sct_label_vertebrae:** Fixed missing top disc label with using -discfile .  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2990)
 - **sct_label_vertebrae:** Obsolete -denoise functionality.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2985)
 - Fix API importing scripts.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2966)

**INSTALLATION**

 - Remove tensorboard==1.14.0 version pinning from requirements.txt.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3185)
 - Fixup realpath polyfill.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3123)
 - Check that conda actually activated during install..  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3103)
 - Turn on stricter shell rules + maintainence.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3102)
 - Add improvements to recent tensorflow-tensorboard fix.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3082)
 - Fix version parsing to support Big Sur (11.0).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3053)

**DOCUMENTATION**

 - Updated badge from travis.org to travis.com.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3165)
 - Add the Twitter badge.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3162)
 - Organize "Specific references" into clearer tables .  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3133)
 - Update references.rst.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3132)
 - **sct_deepseg_sc:** Update reference.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3117)
 - **sct_straighten_spinalcord:** Readability of parameters. Fixed -ldisc-input and -ldisc-dest typos.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3092)
 - Documentation fixes.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3069)
 - Update references.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3066)
 - Fix link reference.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3061)
 - **sct_run_batch:** Update sct_run_batch argparse descriptions to clarify '-itk-threads' usage.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3052)
 - Update references.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3028)
 - Enable fail on warning in RTD.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2980)

**ENHANCEMENT**

 - Various improvements for the manual labeling of cord centerline.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3175)
 - **sct_deepseg:** Add option to have a custom task.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3160)
 - **sct_run_batch:** Print actual numbers of jobs run in parallel..  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3109)
 - Update tumor segmentation models.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3098)
 - Change default option values to None.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3090)
 - **sct_deepseg_sc,sct_detect_pmj,sct_propseg:** Implement -o flag for a few functions.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3081)
 - **sct_label_utils:** Update -create-viewer argument to use parse_num_list function.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3076)
 - **sct_label_vertebrae:** Check that there is two inputs for initz.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3060)
 - **sct_register_multimodal:** Introduced flags samplingStrategy and samplingPercentage for ANTs calls; Set default to 'None' to ensure reproducible results.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3050)

**REFACTORING**

 - Refactor CLI init steps to be consistent across scripts.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3091)
 - Fix API importing scripts.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2966)

**GIT/GITHUB**

 - Add slightly modified PR template from shimming-toolbox.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3032)

## 5.0.1 (2020-11-12)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/5.0.0...5.0.1)

**CI**

 - Travis: Add 10.15 (Catalina), update 10.14 (Mojave) image.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3018)

**BUG**

 - **sct_process_segmentation:** Add missing type information to argument in sct_process_segmentation.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3039)
 - Pin onnxruntime==1.4.0 to avoid libomp issue on macOS.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3037)

**DOCUMENTATION**

 - **sct_label_utils:** sct_label_utils.py: Clarify -create-seg usage description.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3023)
 - RTD: Re-enable showing version text underneath logo.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3021)

**ENHANCEMENT**

 - **sct_run_batch:** sct_run_batch: handle the case of unexecutable script.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3031)
 - **sct_label_utils:** sct_label_utils.py: Add message for generated files.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3026)
 - Display command when scripts are called from the command-line.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3020)

**REFACTORING**

 - **sct_deepseg:** Refactor deepseg/core.py into sct_deepseg and update relevant test.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3022)

## 5.0.0 (2020-11-07)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/4.3...5.0.0)

**FEATURE**

 - **sct_get_centerline,sct_qc:** Implement QC sct_get_centerline.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2784)
 - **sct_run_batch:** Disabling progress bars.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2763)
 - **sct_label_utils,sct_qc:** Implemented QC report for sct_label_utils.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2734)

**BUG**

 - Logging and printing fixes.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3011)
 - Restore subpackage module imports to fix test errors.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3010)
 - Fix incorrect checking of input arguments.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2933)
 - **sct_register_to_template:** sct_register_to_template: Fix '%' in argparse iCloud help links.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2928)

**INSTALLATION**

 - **sct_utils:** Explicit listing of console scripts.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2759)
 - Retrieve data bundles from their new location.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2733)

**DOCUMENTATION**

 - Update sct_extract_metric help to fix RTD error + fix outdated usage.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3000)
 - Add "Segmentation" tutorial to RTD that mirrors SCT course contents.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2991)
 - fix doc build warnings + associated bug.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2955)
 - Update Win10 WSL installation information (Move from Wiki to RTD, update recommendations for FSLEyes/WSL1).  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2952)
 - Update LICENSE to LGPLv3.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2899)
 - Automatically show defaults in argparse help descriptions.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2886)

**ENHANCEMENT**

 - **sct_extract_metric:** Introduce flag to list labels.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/3012)
 - **sct_register_to_template:** Fixes inconsistencies between PAM50 levels and cord.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2998)
 - Use SystemExit not sys.exit & only on error paths..  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2983)
 - fsleyes: allow user to specify output folder.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2941)
 - **sct_image:** sct_image: -copy-header should use -o as output file.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2940)
 - **sct_concat_transfo:** Deprecate sct_concat_transfo + refactor callers.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2914)
 - **sct_maths:** sct_maths: Convert usage of convert_list_str to use list_type.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2901)
 - **sct_run_batch:** Prevent crash if folder already exists.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2867)
 - **msct_parser:** Convert 11-20 out of 20 remaining scripts from msct_parser to argparse.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2842)
 - **sct_run_batch:** Fix thread reporting, early termination, and indentation error in sct_run_batch.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2827)
 - **msct_parser:** Convert 1-10 out of 20 remaining scripts from msct_parser to argparse.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2819)
 - **sct_apply_transfo:** Clarified cropping strategy for sct_apply_transfo; remove warning.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2811)
 - **sct_run_batch:** Various improvements.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2795)
 - **sct_run_batch:** Introduced variable PATH_DATA_PROCESSED.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2787)
 - **sct_deepseg:** Accommodate a cascade of deep learning models.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2748)
 - Improvements for the FSLeyes plugin.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2741)
 - **sct_run_batch:** Various improvements: create log, send email, config file, include/exclude list of subjects.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2732)

**REFACTORING**

 - **msct_parser:** Remove msct_parser and clean up remaning usage.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2936)
 - Convert argparse '0'/'1' options to be typed as ints rather than strings.  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2900)



## 4.3 (2020-06-11)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/4.2.2...4.3)

**BUG**

 - **sct_label_vertebrae:** Fixed -initlabel problem with file naming. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2738)
 - **sct_utils:** Fix and move send_email. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2729)
 - **sct_qc:** Fix out of order plots. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2722)
 - **sct_label_utils:** sagittal dialog: fixes error when trying to access out of bound slice. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2694)
 - **sct_register_multimodal:** Fixed forgot to reorient mask. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2660)
 - **sct_label_vertebrae:** -initz flag and label value correction. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2641)
 - **sct_qc:** Test for parallel qc crash. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2628)
 - Replace parser.usage.generate with parser.error. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2621)

**ENHANCEMENT**

 - **sct_download_data:** Moved sct_download_data functions to new download module. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2700)
 - SCT logo fix in FSLeyes. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2657)
 - **sct_run_batch:** Flag -s added to force sequential analysis even if GNU parallel is installed. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2635)

**FEATURE**

 - Added useful formatting to compare SCT versions; introduced sct_version. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2723)
 - **sct_get_centerline:** Fit centerline across all slices with input segmentation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2646)
 - **sct_deepseg,sct_download_data:** Centralized all deep learning segmentation tasks with new function "sct_deepseg" and refactored sct_download_data. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2639)
 - **sct_dmri_moco,sct_fmri_moco:** Now possible to use soft mask, bug fixes and various improvements. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2634)
 - parameters_example.sh: Now defined a relative PATH_PARENT. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2619)
 - **sct_maths:** Enable 2D kernel for morpho math operations and various improvements. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2616)
 - **sct_apply_transfo:** Fixed compatibility between SCT/ANTs and FSL warping fields. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2529)

**DOCUMENTATION**

 - Fix argparse linewrap for R| strings. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2685)
 - Added video recording of the London 2020 course. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2659)
 - Update badge for discourse forum. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2656)
 - **sct_dmri_concat_b0_and_dwi:** Clarified documentation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2650)
 - Added tutorial to install SCT with WSL for Windows users. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2643)
 - Fixed README forum link. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2631)

**INSTALLATION**

 - Drop support for 2014-era Linux.. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2731)
 - Minimal Torch install.. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2715)
 - **sct_dmri_moco,sct_fmri_moco:** Updated ANTs binaries to solve slow processing in some systems. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2642)
 - Downgraded TensorFlow to 1.5 to fix AVX incompatibility. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2618)

**TESTING**

 - Added CI for Windows Subsystem for Linux (WSL). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2698)
 - **sct_get_centerline:** Relax test tolerance for centerline polyfit. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2686)
 - Add polynomial function to dummy_segmentation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2684)
 - create_test_data: move to spinalcordtoolbox.testing. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2638)



## 4.2.2 (2020-02-28)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/4.2.1...4.2.2)

**BUG**

 - **sct_maths:** Fixed missing type when using erode feature. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2613)
 - **sct_crop_image:** Fixed flag -b crops instead of masking. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2606)
 - **sct_crop_image:** Fixed problem with parameters xmax, ymax and zmax. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2604)
 - **sct_deepseg_sc,sct_utils:** Check if input data is 3D or 2D. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2598)

**ENHANCEMENT**

 - **sct_deepseg_gm,sct_deepseg_lesion,sct_deepseg_sc:** build(deps): bump tensorflow from 2.0.0 to 2.0.1. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2592)
 - Added more functions to the FSLeyes plugin. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2581)

**DOCUMENTATION**

 - **sct_qc:** Fixed QC display syntax for Docker users. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2589)



## 4.2.1 (2020-01-14)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/4.2.0...4.2.1)

**BUG**

 - **sct_warp_template:** Fixed generation of QC report. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2567)
 - **sct_register_multimodal:** Fixed bug related to missing output file. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2566)

**ENHANCEMENT**

 - sct_utils: Changed default open command for Linux. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2578)
 - **sct_deepseg_sc,sct_label_vertebrae:** Better error handling if installation files are missing and clarified help. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2560)

**INSTALLATION**

 - Added gcc as installation pre-requisite with useful instructions. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2577)



## 4.2.0 (2019-12-19)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/4.1.1...4.2.0)

**ENHANCEMENT**

 - **sct_register_multimodal,sct_register_to_template:** New method for detecting rotation in centermassrot. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2331)

**FEATURE**

 - **sct_register_to_template:** Spinal-level-based registration to the PAM50. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2513)

**DOCUMENTATION**

 - Added info for running SCT via Vbox. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2551)

**INSTALLATION**

 - Now using requirements-freeze.txt for installing stable releases. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2558)
 - install/sct_changelog: Fixed bug when fetching previous release tag. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2550)
 - Removed Darwin=15 case in requirements.txt and added OS checks during installation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2534)



## 4.1.1 (2019-11-28)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/4.1.0...4.1.1)

**BUG**

 - **sct_register_to_template:** Fixed cropping of registered image. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2532)
 - **sct_run_batch:** Fixed issue when passing absolute path to script. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2518)

**FEATURE**

 - **sct_apply_transfo:** Added an option for keypoints transformation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2533)
 - **sct_label_utils:** Now possible to add existing label from an external file in the create-viewer option. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2520)

**DOCUMENTATION**

 - README: Added link to Youtube tutorials. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2521)

**INSTALLATION**

 - **sct_viewer:** PyQt5 version downgrade to fix GUI on Debian 8.11 distros. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2524)
 - **sct_check_dependencies:** Increased sensitivity of dependency testing. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2522)



## 4.1.0 (2019-10-26)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/4.0.2...4.1.0)

**BUG**

 - **sct_compute_mtr:** Fixed aberrant mtr values. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2503)
 - **sct_process_segmentation:** Fixed wrong orientation with new version of scikit-image. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2495)
 - **sct_maths:** Fixed deprecation with adaptative thresholding. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2474)
 - **sct_deepseg_sc:** Fixed segmentation issue that mostly appeared on DWI data. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2467)
 - **sct_qc,sct_resample:** QC report: Fixed shift along slice direction between image and overlay. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2455)
 - **sct_straighten_spinalcord:** Fixed shape mismatch during straightening. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2444)
 - **sct_denoising_onlm:** Fix index error and display fsleyes command. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2432)

**ENHANCEMENT**

 - **sct_resample:** Raise error if trying to resampling to size zero. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2507)
 - **sct_deepseg_sc:** Fix deepseg threshold (again). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2490)
 - **sct_deepseg_sc:** Fine-adjustment of threshold for binarization of soft segmentation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2479)
 - **sct_qc:** Fixed up/down dysfunction in the qc report . [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2477)
 - **sct_crop_image:** Improved CLI, fixed bug with the GUI and refactored into module. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2459)
 - **sct_deepseg_sc:** Remove isolated voxels at the edge of the output segmentation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2437)
 - **sct_download_data:** Check if folder already exists by checking its actual name, not the name of the entry to -d flag. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2425)

**FEATURE**

 - **sct_qc:** Added button to toggle overlay and removed automatic fading. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2514)
 - **sct_dmri_concat_b0_and_dwi:** New script to concatenate b0 and dwi data. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2452)
 - **sct_process_segmentation:** Compute cord length. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2431)

**DOCUMENTATION**

 - Better management of CLI syntax in case mandatory arguments are missing. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2457)

**INSTALLATION**

 - **sct_download_data:** Fixed issue that appeared when trying to remove temporary folder from different file systems. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2473)


## 4.0.2 (2019-09-04)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/4.0.1...4.0.2)

**BUG**

 - **sct_straighten_spinalcord:** Fixed wrong input arguments. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2427)
 - **sct_fmri_moco:** Replaced sct.mv with shutil.copyfile for tmp space issue. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2420)



## 4.0.1 (2019-08-17)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/4.0.0...4.0.1)

**BUG**

 - **sct_dmri_compute_dti:** Fixed flag '-evecs' not detecting input as of type int. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2417)
 - **sct_image:** Fixed -setorient-data giving wrong results. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2401)
 - **sct_image:** Proper handling of int arguments contained in list type input. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2397)
 - **sct_process_segmentation:** Fixed wrong morphometric measures with anisotropic in-plane resolution. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2396)
 - Change canvas axes for image placement.. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2389)

**ENHANCEMENT**

 - **sct_process_segmentation:** Corrected wrong slice information on QC output. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2408)
 - **sct_apply_transfo:** Fixed q/sform on transformed image. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2400)

**INSTALLATION**

 - **sct_resample:** Dropped dependency to nipy. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2414)
 - Check if gcc is installed. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2410)
 - Modify bashrc on sudo. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2409)



## 4.0.0 (2019-08-04)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.2.7...4.0.0)

**BUG**

 - **sct_crop_image:** Fixed bug when using GUI (flag -g). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2325)
 - **sct_label_vertebrae:** Fixed misplaced label in non-RPI data for initializing vertebral labeling. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2289)
 - **sct_qc:** Fixed corruption of QC json file when running parallel jobs. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2250)
 - **sct_label_vertebrae:** Fixed bug that appeared when inputing uncompressed nifti file. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2202)
 - **sct_label_vertebrae:** Fixed bug in the post processing of detect_c2c3. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2201)
 - **sct_propseg:** Fixed ignored -init flag and minor improvements. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2198)
 - **sct_deepseg_lesion,sct_deepseg_sc:** Fixed bug and clarified usage of -centerline viewer. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2183)
 - **sct_dmri_moco,sct_fmri_moco:** Work around "too many open files" by slurping the data. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2163)
 - **sct_straighten_spinalcord:** Fixed crash caused by wrong estimation of centerline length in case of incomplete segmentation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2154)
 - **sct_extract_metric:** Fixed bug in method max. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2087)
 - **sct_flatten_sagittal:** Fix bugs related to image scaling. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2070)
 - **sct_label_vertebrae:** Fixed path issue when using -initlabel flag. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2068)
 - **sct_get_centerline:** Convert data to float before intensity rescaling (in optic). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2065)
 - **sct_deepseg_lesion,sct_deepseg_sc:** Fixed ValueError and IndexError. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2060)
 - **sct_register_to_template:** Fixed regression bugs. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2052)

**ENHANCEMENT**

 - batch_processing.sh: Replaced propseg by deepseg_sc. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2377)
 - batch_processing.sh: QC report is now generated locally. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2370)
 - **msct_parser:** Conversion from msct_parser to argparse. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2315)
 - **sct_qc:** Allow the possibility to discard column of choice on the output html QC report. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2301)
 - **sct_process_segmentation,sct_straighten_spinalcord:** Improve quality of straightening. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2299)
 - **sct_deepseg_lesion,sct_deepseg_sc:** Output segmentation in uint8 when input is float. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2297)
 - **sct_qc:** Added automatic data sorting in the QC report. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2295)
 - **sct_fmri_moco:** Enabling the extraction of fMRI motion correction parameters. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2293)
 - **sct_qc,sct_resample:** Fixed resampling method with reference image and improved speed for generating QC report. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2285)
 - **sct_compute_mtr:** Added output file and/or folder flag. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2264)
 - **sct_compute_snr:** Make consistent STD calculation between sct_fmri_compute_tsnr and sct_compute_snr. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2263)
 - Clarify handling of logger, error and exceptions. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2221)
 - **msct_register,sct_register_multimodal,sct_register_to_template:** Refactoring to allow use of im AND seg in the registration process. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2218)
 - **sct_get_centerline,sct_straighten_spinalcord:** Increased smoothness of default bspline centerline fitting algorithm . [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2214)
 - **sct_get_centerline:** Remove Optic temp files by default. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2212)
 - **sct_qc:** Lock qc report during generation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2206)
 - **sct_process_segmentation:** Major modifications to simplify usage and fix various issues with shape analysis. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2200)
 - **sct_process_segmentation:** Minor fix in usage and csv output. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2199)
 - **sct_warp_template:** Faster execution and other minor improvements. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2175)
 - **sct_qc:** Various improvements on the QC report and resampling module. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2157)
 - **sct_process_segmentation:** Major refactoring of centerline routine. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2147)
 - **sct_label_vertebrae:** Removed support for -initc2 flag because there is an alternative approach with sct_label_utils. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2125)
 - **sct_extract_metric:** Expose aggregate_slicewise() API and various improvements. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2115)
 - **sct_register_to_template:** Updated PAM50 template header to be in the same coordinate system as the MNI template. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2102)
 - **sct_qc:** Various improvements. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2097)
 - **sct_deepseg_lesion,sct_deepseg_sc:** deepseg_sc: Speed processing up . [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2094)
 - **sct_qc:** QC now scales images based on physical dimensions (previously based on voxels). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2089)
 - **sct_process_segmentation:** Major refactoring to bring few improvements. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1931)

**FEATURE**

 - **sct_qc:** Add CSA results on QC report . [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2306)
 - **sct_extract_metric:** Added flag to combine all labels. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2291)
 - **sct_dmri_compute_dti:** Output DTI Eigenvalues. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2274)
 - **sct_qc:** New API to generate QC reports. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2216)
 - **sct_label_vertebrae:** Added possibility to rescale intervertebral disc distance and various improvements. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2182)
 - **sct_register_to_template:** Now possible to specify the type of algorithm used for cord straightening. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2101)
 - **sct_label_vertebrae:** spinalcordtoolbox/vertebrae/detect_c2c3 -- New module. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2084)
 - **sct_compute_snr:** Now possible to output SNR map, removed requirement for inputing mask, and few other improvements. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2080)
 - **sct_dmri_separate_b0_and_dwi:** sct_dmri_separate_b0_and_dwi: Now append suffix to input file name to prevent conflicts. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2075)
 - **sct_smooth_spinalcord:** Enable to set smoothing parameters in all axes. **WARNING: Breaks compatibility with previous versions of SCT.** [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2073)

**DOCUMENTATION**

 - **sct_label_vertebrae:** Updated documentation on how to create vertebral and disc labels. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2107)
 - **sct_changelog:** Few improvements on automatic Changelog generation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2077)

**INSTALLATION**

 - Fixed compatibility with OSX 10.11 (El Capitan). [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2309)
 - Fixed Keras/Tensorflow compatibility with CentOS 7 by downgrading to Python 3.6. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2276)
 - Using Python 3.x for default installation. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2238)
 - Fixed installation error caused by old SSL module. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2223)
 - First pass at also supporting pip installations. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1996)

**TESTING**

 - Travis: Adding distribs and displaying allow_failures. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2298)
 - **sct_deepseg_lesion,sct_deepseg_sc:** Added new unit tests. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2159)



## v3.2.7 (2018-10-29)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.2.6...v3.2.7)

**BUG**

 - sct_fmri_moco: Fixed regression bug related to the use of a mask [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2047)
 - msct_nurbs: Fixed singular matrix error [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2042)

**ENHANCEMENT**

 - sct_extract_metric: Do not zero negative values [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2049)


## v3.2.6 (2018-10-16)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.2.5...v3.2.6)

**BUG**

 - sct_propseg: Reordered variable assignment [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2039)
 - sct_straighten_spinalcord: Fixed AttributeError related to conversion of numpy array to list [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2026)
 - sct_create_mask: Few fixes and improvements [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2021)

**ENHANCEMENT**

 - sct_get_centerline: Use the new viewer [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2035)
 - sct_fmri_moco: Generalize motion correction for sagittal acquisitions and other improvements [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2022)

**FEATURE**

 - sct_straighten_spinalcord: Few fixes and improvements [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2033)
 - sct_deepseg_sc/lesion: Allow to input manual or semi-manual centerline [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2020)


## v3.2.4 (2018-08-24)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.2.3...v3.2.4)

**BUG**

 - Updated URL for PAM50 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1997)
 - sct_register_to_template: Fixed wrong projection in case labels not in same space [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1978)
 - sct_extract_metric: Fixed recently-introduced bug related to output results [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1965)

**ENHANCEMENT**

 - Few fixes in sct_extract_metric and batch processing outputs [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/2008)
 - sct_dmri_compute_dti: Output tensor eigenvectors [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1975)
 - Totally pimp the Image Slicer (to act like a sequence, to slice many images), and add unit tests for the slicer [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1956)
 - Second pass at image refactoring [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1948)


## v3.2.3 (2018-07-21)
This release notably brings a useful feature, which now makes it possible to use single-label with -l flag for registration to the template. This feature was required by the recently-introduced [analysis pipeline for multi-parametric data when FOV is systematically centered at a particular disc or mid-vertebral level](https://github.com/sct-pipeline/multiparametric-fixed-fov). [View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.2.2...v3.2.3)

**BUG**

 - `sct_register_multimodal`: Fixed bug when using partial mask with algo=slicereg [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1917)

**ENHANCEMENT**

 - `sct_propseg`: Labels and centerline are now output with correct header if -rescale is used [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1926)
 - Adding a batch size of 4 for all deep learning methods. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1924)

**FEATURE**

 - `sct_register_to_template`: Enable single-label with -l flag [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1955)


## v3.2.2 (2018-07-08)
Major changes to this release include a fix to SCT installation on OSX laptops with non-English encoding language. Another important fix is the inclusion of the link in `sct_download_data` for downloading the Paris'18 SCT course material. A nice enhancement is the possibility to calculate metrics slice-wise or level-wise in `sct_extract_metric`. View detailed changelog
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.2.1...v3.2.2)

**BUG**

 - sct_label_vertebrae: Added subcortical colormap for fslview [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1883)
 - sct_flatten_sagittal: Fixed wrong indexation [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1882)

**ENHANCEMENT**

 - sct_deepseg_gm: Lazy loading module: now faster when calling usage [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1908)
 - sct_propseg: Now possible to rescale data header to be able to segment non-human spinal cord (mice, rats, etc.) [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1903)
 - sct_deepseg_gm: Adding TTA (test-time augmentation) support for better segmentation results [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1894)
 - sct_deepseg_gm: Removed restriction on the network input size (small inputs): Fixes bug that appeared when inputting images with small FOV [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1877)
 - sct_deepseg_sc: Reducing TensorFlow cpp logging verbosity level [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1876)
 - sct_extract_metric: Now possible to calculate metrics slice-wise or level-wise [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1875)

**DOCUMENTATION**

 - Added documentation for installing SCT on Windows using Docker [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1879)
 - Added information on the README about how to update SCT from git install [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1867)
 - Updated documentation and added link to the data for the SCT course in Paris [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1858)

**INSTALLATION**

 - Use pip install -e spinalcordtoolbox to gain flexibility [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1892)
 - Local language support (LC_ALL) added to installation& launcher on macOS [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1881)

**TESTING**

 - Removed sct_register_graymatter (obsolete old code) from sct_testing functions [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1904)
 - Implemented multiprocessing and argparse in sct_testing, and other improvements related to Sentry [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1872)


## 3.2.1 (2018-06-12)
This release fixes a few bugs, notably one related to template registration when using disc-based alignment. It also features an improved version of sct_deepseg_sc with the introduction of 3D kernel models, as well as a more accurate segmentation on T1-weighted scans. The main documentation now includes a link to a new collection of repositories: sct-pipeline, which gathers examples of personalized analysis pipelines for processing spinal cord MRI data with SCT. [View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.2.0...3.2.1)

**BUG**

 - Skip URL if filename isn't provided by HTTP server; catch anything in URL try loop [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1843)
 - Fixed registration issue caused by labels far from cord centerline [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1828)
 - Fixed wrong disc labeling and other minor improvements [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1814)
 - Added test to make sure not to crop outside of slice range [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1813)
 - Forcing output type to be float32 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1800)
 - Fixed z_centerline_voxel not defined if -no-angle is set to 1 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1797)

**ENHANCEMENT**

 - Adding threshold (or not) option for the sct_deepseg_gm [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1846)
 - Manual centerline is now output when using viewer [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1840)
 - Added CNN for centerline detection, brain detection and added possibility for 3d CNN kernel [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1822)
 - Fixed verbose in QC, integrated coveralls [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1808)
 - Now possible to specify a vertebral labeling file when using -vert [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1798)

**DOCUMENTATION**

 - Added link to github.com/sct-pipeline [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1796)

**INSTALLATION**

 - Adapted final verbose if user decided to not modify the .bashrc [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1832)

**TESTING**

 - Coveralls added to Travis to prevent build failure [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1847)



## 3.2.0 (2018-05-29)
This release includes: a new example dataset (now includes T2*-w and fMRI data) with an updated batch_processing.sh, a new function to compute MT-saturation effect (sct_compute_mtsat), an improved straightening that can account for inter-vertebral disc positions to be used alongside sct_register_to_template for more accurate registration, and few improvements on sct_pipeline and quality control (QC) report generation. [View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.1.1...3.2.0)

**BUG**

 - Fixed sct_pipeline if more than two -p flags are used [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1760)
 - Fixed re-use of the same figure during QC generation [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1711)
 - sct_deepseg_sc - Issue when input is .nii instead of .nii.gz [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1706)
 - Fslview no more called at the end of process if it it deprecated [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1648)
 - Fixing the TensorFlow installation for some old platforms. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1646)
 - Re-ordering of 4th dimension when apply transformation on 4D scans [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1638)
 - Fix "-split" option issues on sct_image [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1627)

**ENHANCEMENT**

 - Updated batch_processing and sct_example_data with new features [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1779)
 - Various fixes for sct_pipeline [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1771)
 - sct_pipeline: store metadata in Pickle report [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1761)
 - Adding volume-wise standardization normalization for the sct_deepseg_gm [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1757)
 - Make sct_get_centerline robust to intensities with range [0, 1] [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1746)
 - Improved doc and minor fixes with centerline fitting [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1736)
 - Make sct_process_segmentation compatible with the new ldisc convention [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1735)
 - Removed flirt dependency [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1729)
 - More pessimistic caching of outputs [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1719)
 - Slice counting fixed [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1687)
 - output of -display ordered per label value [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1686)
 - Improvements in straightening and registration to the template [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1669)
 - The QC report is now a standalone html file. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1668)
 - Adding a port option for the qc server [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1657)
 - Make QC generation opt-in [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1653)
 - Fixing cropping issue in sct_straighten_spinalcord [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1652)
 - Set MeanSquares the default metric for sct_fmri_moco [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1649)
 - Now possible to change data orientation on 4D data [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1637)
 - Use python  concurrent.futures instead of multiprocessing  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1587)

**FEATURE**

 - New function to create violin plots from sct_pipeline results [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1759)
 - Enable input file with label at a specific disc [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1698)
 - Control the brightness of the image in the GUI. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1684)
 - Implements MTsat function [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1672)
 - Improvements in straightening and registration to the template [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1669)
 - Integration of SCT into fsleyes UI [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1629)
 - Add Sentry error reporting [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1544)

## 3.1.1 (2018-02-16)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.1.0...3.1.1)

**BUG**

 - Fix TensorFlow installation on Debian [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1601)
 - BUG: Fixed a small bug on None condition [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1594)
 - Fixed missing output [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1581)
 - Bug fix and various improvements [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1571)
 - Now working for 2d data [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1565)
 - Fix Timer with progress  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1554)
 - BUG: concat_transfo: fixed wrong catch of dimension [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1534)
 - reinstall only current numpy version [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1520)
 - Enable the calculation of spinal cord shape at the edge of the image [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1517)
 - Disabling rotation in register to template [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1516)

**ENHANCEMENT**

 - Adding minimal Dockerfile for SCT. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1600)
 - Bug fix and various improvements [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1571)
 - Find mirror servers in case OSF is not accessible [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1567)
 - Now supports fsleyes when displaying viewer syntax at the end of a process [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1533)

**FEATURE**

 - sct_deepseg_sc implementation [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1586)
 - sct_deepseg_gm implementation [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1564)

**TESTING**

 - Fixed minor verbose issues during testing [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1582)

## v3.1.0 (2017-10-27)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.0.8...v3.1)

**BUG**

 - Fix errors in create_atlas.m [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1499)
 - Fix a regression bug. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1490)
 - Used the absolute path to create the temporary label file in propseg [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1487)
 - Fixed: Optic is used by default if -init-mask is used with external file provided [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1485)
 - Fixed global dependency in sct_process_segmentation call [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1481)
 - Fixed z-regularization for slicereg [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1465)

**ENHANCEMENT**

 - Fixed: Raise in sct.run in bad order. Also added specific sct errors [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1503)
 - More improvements to the viewer [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1496)
 - Refactored WM atlas creation pipeline and improved documentation  [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1492)
 - Option to install SCT in development mode [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1491)
 - Added key bindings to the undo, save and help actions. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1480)
 - Introduced the zoom functionality to the anatomical canvas [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1477)
 - Improvements on centerline for template generation [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1473)
 - Major refactoring of testing framework [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1453)

**FEATURE**

 - Improvement of sct_analyze_lesions: compute percentage of a given tract occupied by lesions [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1500)
 - sct_get_centerline: new manual feature [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1467)
 - sct_detect_pmj: new_feature [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1466)

**TESTING**

 - Major refactoring of testing framework [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1453)

## v3.0.8 (2017-09-13)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.0.7...v3.0.8)

**BUG**

 - Added try/except for QC report [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1457)
 - Conversion issue for float32 images with large dynamic [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1450)
 - (Partly-)Fixed bug related to memory issue with diagonalization [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1446)
 - DEV: fixed bug on centerline when referencing to the PMJ [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1436)

**ENHANCEMENT**

 - Now possible to input single label at disc (instead of mid-body) [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1451)
 - Now using N-1 instead of N as denominator for computing the STD. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1441)

**FEATURE**

 - Function to analyze lesions #1351 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1439)

## v3.0.7 (2017-08-02)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.0.6...v3.0.7)

**BUG**

 - The params attributes are initialized to the type integer [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1431)
 - Fixing stdout issue on sct_testing [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1430)
 - Changed destination image for concatenation of inverse warping field [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1413)
 - crashes if apply transfo on 4d images [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1408)
 - Allow the -winv parameter to write a file to disk [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1404)
 - Change import path of resample [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1401)
 - Precision error while calculating Dice coefficient #1098 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1398)

**ENHANCEMENT**

 - Enables to set Gaussian weighting of mutual information for finding C2-C3 disk [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1422)
 - Adapt concat and apply transfo to work on 2d images [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1420)
 - Fixed small issues in pipeline [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1417)
 - Use custom template for sct_register_graymatter [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1407)
 - compute_ernst_angle: set the parameter t1 default value to optional value of 850ms [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1400)
 - Improvements on centerline and template generation [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1378)

**FEATURE**

 - NEW: dmri_display_bvecs: new function to display bvecs [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1394)
 - Function to extract texture features #1350 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1376)

**TESTING**

 - Various fixes to pipeline and testing [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1424)
 - New test for sct_label_utils compatible with sct_pipeline [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1402)

**DOCUMENTATION**

 - Changed default values and clarified doc [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1405)


## v3.0.6 (2017-07-04)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.0.5...v3.0.6)

**BUG**

 - Catch the OSError exception thrown when the git command is missing [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1396)
 - BUG: register_multimodal: fixed typo when calling isct_antsRegistration [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1392)
 - BUG: fix bug when slice is empty [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1390)
 - Ignore using user packages when install with conda and pip [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1384)
 - Fix referential for JIM centerline [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1377)
 - image/pad: now copy input data type (fixes issue 1362) [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1363)
 - Use a pythonic way to compare a variable as  None [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1361)
 - The init-mask accepts "viewer" as a value [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1357)
 - Fixed unassigned variable in case -z or -vert is not used [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1354)

**ENHANCEMENT**

 - Restrict deformation for ANTs algo [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1388)
 - Made error message more explicit if crash occurs [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1387)
 - Insert previous and next buttons in the qc reports page [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1358)
 - integrate new class for multiple stdout inside sct_check_dependencies [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1342)

**DOCUMENTATION**

 - Update README.md [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1346)

**INSTALLATION**

 - Ignore using user packages when install with conda and pip [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1384)
 - Update sct testing data [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1370)
 - Added the dependency psutil in the conda requirements [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1359)
 - Added egg files in the list of gitignore [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1355)

## v3.0.5 (2017-06-09)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.0.4...v3.0.5)

**BUG**

 - Force numpy 1.12.1 on osx and linux [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1352)
 - Use a different function to identify if a file exists [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1341)
 - Fixing an issue introduced with the sct_get_centerline. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1338)

**ENHANCEMENT**

 - Binarize GM seg after warping back result to original space [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1347)
 - Generation of centerline as ROI [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1344)

**FEATURE**

 - Introduce a pipeline to use the HPC architecture [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1260)

## v3.0.4 (2017-05-19)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.0.3...v3.0.4)

**BUG**

 - Normalize the init value to between 0 and 1 for propseg [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1324)
 - Moved the QC assets into the spinalcordtoolbox package [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1311)
 - Improved the formatting of the changelog generator [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1303)
 - Show remaining time status for downloads [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1299)

**ENHANCEMENT**

 - Added the command parameter `-noqc` [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1313)
 - Add dimension sanity checking for input file padding op [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1304)

**FEATURE**

 - Introducing spinal cord shape symmetry [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1332)

**TESTING**

 - Validate the function name in sct_testing [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1305)
 - Fix regression bug in sct_testing [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1310)

## v3.0.3 (2017-04-26)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.0.2...v3.0.3)

**BUG**

 - Fixes case if data image, segmentation and labels are not in the same space [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1292)
 - Fix the handling of the path of the QC report. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1285)
 - Change the format of the SCT version. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1284)
 - changed the DISPLAY variable due to conflicts with FSLView in batch_processing.sh [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1281)

**INSTALLATION**

 - Added course_hawaii17 into the list of available dataset from sct_download_data [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1291)
 - Incorrect variable when installing SCT in a different directory [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1289)

**DOCUMENTATION**

 - Added description with examples in the register_to_template command (#1262) [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1287)
 - Fixed typo in register_multimodal command [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1276)

## v3.0.2 (2017-04-20)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.0.1...v3.0.2)

**BUG**

 - Force the SCT environment to use only the python modules installed by SCT [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1266)
 - Fixing disabling options on straightening [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1255)
 - Fixed tSNR computation of the mean and std of the input image [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1245)
 - Increased the data type size from the default int16 to int32 to avoid overflow issues in sct_process_segmentation [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1224)
 - Fixed data type issue in sct_process_segmentation [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1223)

**ENHANCEMENT**

 - Improvements to denoising on sct_segment_graymatter [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1265)
 - Extend the functionality of sct_viewer [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1254)
 - Add OptiC for improved spinal cord detection [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1253)
 - Introduction spinalcordtoolbox python setup file [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1243)

**FEATURE**

 - Add option -rms to perform root mean square (instead of mean) in sct_maths [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1256)
 - Introduce QC report generation [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1251)
 - Introduce the QC html viewer [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1250)

**TESTING**

 - Introduce the QC html viewer [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1250)
 - Introduce python package configuration file (setup.cfg) [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1242)

## v3.0.1 (2017-03-22)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.0.0...v3.0.1)
### FEATURE
 - Merge multiple source images onto destination space. [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1220)

## v3.0.0 (2017-03-15)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.0_beta32...v3.0.0)
### BUG
 - Modifying the type of coordinates for vertebral matching [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1206)
 - Removing discontinuities at edges on segmentation [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1196)
 - BUG: computing centreline using physical coordinates instead of voxel… [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1189)
 - Fix issue #1172: -vertfile as an optional parameter [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1173)
 - Improvements to the viewer of sct_propseg [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1169)
 - Removed confusion with command variables when using PropSeg viewer [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1161)
 - Patch sct_register_to_template with -ref subject [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1158)
 - zero voxels no more included when computing MI + new flag to compute normalized MI [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1136)
### ENHANCEMENT
 - Changed default threshold_distance from 2.5 to 10 to avoid edge effect [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1183)
 - Adapt sct_create_mask and sct_label_utils to 2D data [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1180)
 - Improvements to the viewer of sct_propseg [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1169)
### TESTING
 - OPT: display mean and std instead of mean twice [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1187)

## 3.0_beta32 (2017-02-10)
[View detailed changelog](https://github.com/spinalcordtoolbox/spinalcordtoolbox/compare/v3.0_beta31...v3.0_beta32)
### BUG
 - BUG: install_sct: fixed PATH issue (#1153): closed at 2017-02-08 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1154)
 - BUG: compute_snr: fixed variable name: closed at 2017-02-03 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1148)
 - Changed the algorithm to fetch the download filename: closed at 2017-02-03 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1146)
 - Copy header of input file to ensure qform is unchanged: closed at 2017-01-31 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1137)
 - zero voxels no more included when computing MI + new flag to compute normalized MI: closed at 2017-02-01 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1136)
 - Downloading the binaries using the python module instead of CURL: closed at 2017-01-30 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1134)
 - [sct_segment_graymatter] correct background value: closed at 2017-01-31 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1133)
 - Fixing indexes issue on Travis OSX: closed at 2017-01-17 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1115)
 - REF: display spinal cord length when required (full spinal cord): closed at 2017-01-17 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1112)
 - Adding rules for in-segmentation errors: closed at 2017-01-17 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1110)
### ENHANCEMENT
 - Generate a changelog from GitHub: closed at 2017-02-10 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1152)
 - OPT: maths: visu only produced if verbose=2: closed at 2017-02-02 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1147)
### TESTING
 - Add message to user when spinal cord is not detected and verbose improvement for testing: closed at 2017-02-01 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1145)
 - Display results of isct_test_function: closed at 2017-01-20 [View pull request](https://github.com/spinalcordtoolbox/spinalcordtoolbox/pull/1117)

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
