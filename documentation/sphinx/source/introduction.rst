Introduction
############

SCT tools process MRI data (NIFTI files) and can do tasks such as:

- Automatic identification (segmentation) of the spinal cord
- Automatic vertebral labeling
- Segmentation of spinal cord white matter and gray matter
- Registration to template, and deformation (eg. straightening)
- Detection of anatomical highlights (eg. PMJ, spinal cord centerline)
- Correction (motion compensation, eddy currents correction)
- Measurements (for quantitative MRI)
- Help out with manual labeling and segmentation with a GUI

It also has low-level tools:

- Warping field creation and application
- NIFTI volume manipulation tools for common operations


Notes
#####


Segmentation of GM/WM
*********************




Quality Control
***************

Some SCT tools can generate Quality Control (QC) reports.
These reports consist in “appendable” HTML files, containing a table
of entries and allowing to show, for each entry, animated images
(background with overlay on and off).

To generate a QC report, add the `-qc` command-line argument,
with the location (folder, to be created by the SCT tool),
where the QC files should be generated.

