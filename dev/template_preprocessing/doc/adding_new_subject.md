# Adding a subject to the template data set

This is a step-by-step procedure for adding a subject to the template data set. The creation of the template is an automatic procedure which, for each subject, takes as input the DICOM PATHs and returns the data completely processed (i.e., straightened, registed to the template space and vertebrae aligned). However, it is inevitable to add for each subject small adjustements (see below).

## Summary of steps

### Files to be created
- ``crop.txt``: ASCII file to indicate where to crop the data
- ``centerline_propseg_RPI.nii.gz``: labeled NIFTI image (binary) to help propseg generating the segmentation of the spinal cord
- ``labels_vertebral.nii.gz``: labeled NIFTI image (not binary) to indicate fiducial markers corresponding to the brainstem and the vertebral bodies (from C2-C3 to T12-L1)

N.B.: Those files need to be generated for both contrasts T1 and T2 (i.e., 6 files in total).

### Files to modify
- ``dev/template_preprocessing/pipeline_template.py``: the batch used to create the template for both T1 and T2 data.
  - you need to add your subject to the variable SUBJECT_LIST_TOTAL
- ``dev/template_preprocessing/preprocess_data_template.py``: batch that creates all the files that you will have bravely generated yourself (see above)

## Detailed pocedure

All subjects need to have both a full T1 and a full T2 image.

Step-by-step procedure:

* Convert the DICOM to NIFTI (e.g., using dcm2nii, output file name: data.nii.gz)
* Change orientation to RPI
  * ``sct_orientation –i data.nii.gz –s RPI -o data_RPI.nii.gz``
* Open data_RPI.nii.gz in fslview and create a mask (cmd+c), which indicates the following anatomical landmarks:
  * Landmark value 1: labels_vertebral_1.png (rostral pons)
  * Landmark value 2: labels_vertebral_2.png (ponto-medullary junction)
  * Landmark value 3-20: labels_vertebral_3-20.png (vertebral levels from C2-C3 to T12-L1)
    * value 3: C2-C3,
    * value 4: C3_C4, ...
    * value 8: C7-T1
    * value 9: T1-T2, ...
    * value 20: T12-L1
* Save the mask under: ``labels_vertebral.nii.gz`` (cmd+s).
* Crop **data_RPI.nii.gz** slightly above the brainstem and slightly below L2-L3.
  * ``sct_crop_image –i data_RPI.nii.gz –dim 2 XXX -o data_RPI_crop.nii.gz``
* Report where you are cropping the image in the file **crop.txt**:
  * Format: zmin_anatomic,zmax_anatomic  (e.g.: 15,623 if you are cropping between slices 15 and 623). If there is a need to crop along y axis (as for some data from marseille that present artefacts) please specify as follow: zmin_anatomic,zmax_anatomic,ymin_anatomic, ymax_anatomic (e.g.: 15,623,30,200 if you are adding a crop along y axis between slices 30 and 200).

