# Adding a subject to the template data set

This is a step-by-step procedure for adding a subject to the template data set. The creation of the template is an automatic procedure which, for each subject, takes as input the DICOM PATHs and returns the data completely processed (i.e., straightened, registed to the template space and vertebrae aligned). However, it is inevitable to add for each subject small adjustements (see below).

## Summary of steps

### Files to be created
- ‘crop.txt’: ASCII file to indicate where to crop the data
- ‘centerline_propseg_RPI.nii.gz’: labeled NIFTI image (binary) to help propseg generating the segmentation of the spinal cord
- ‘labels_vertebral.nii.gz’: labeled NIFTI image (not binary) to indicate fiducial markers corresponding to the brainstem and the vertebral bodies (from C2-C3 to T12-L1)

N.B.: Those files need to be generated for both contrasts T1 and T2 (i.e., 6 files in total).

### Files to modify
- 'dev/template_preprocessing/pipeline_template.py': the batch used to create the template
  - you need to add your subject to the variable SUBJECT_LIST_TOTAL
- 'dev/template_preprocessing/preprocess_data_template.py': batch that creates all the files that you will have bravely generated yourself (see above)


## Detailed pocedure
