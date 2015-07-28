# TEMPLATE PREPROCESSING

- [Step-by-step procedure](https://github.com/neuropoly/spinalcordtoolbox/blob/template/dev/template_preprocessing/README.md#step-by-step-procedure)
- [Data](https://github.com/neuropoly/spinalcordtoolbox/blob/template/dev/template_preprocessing/README.md#Data)
- [Adding a new subject to the pipeline](https://github.com/neuropoly/spinalcordtoolbox/tree/template/dev/template_preprocessing#adding-a-new-subject-to-the-pipeline)

## Step-by-step procedure

The following functions are used to preprocess T1 and T2 data for generating a template of the spinal cord. Steps to follow:
- Open: **preprocess_data_template.py**
  - edit variable ``path_results`` and specify output results. E.g.: /Users/julien/data/template_results
- Run **preprocess_data_template.py**:
  - ``python ~/code/spinalcordtoolbox/dev/template_preprocessing/preprocess_data_template.py``
- Open: **pipeline_template.py**
  - Edit variable: ``PATH_INFO``: corresponds to the variable ``path_results`` in file **preprocess_data_template.py**
  - Edit variable: ``PATH_OUTPUT``: results of fully-preprocessed T1 and T2 to be used for generating the template.
- Run **pipeline_template.py**:
  - ``python ~/code/spinalcordtoolbox/dev/template_preprocessing/pipeline_template.py``
- Use output data for generating the template.
- Once you have generated the T1 and T2 template, you need to co-register them.
  - ``TODO``

## Data

All data (inputs, outputs and info files) are located in (NeuroPoly lab):
~~~
/Volumes/Usagers/Etudiants/tamag/data/data_template
~~~

The following data were not selected:
- T020b: only has one contrast (T1 or T2)
- errsm_26: only has one contrast (T1 or T2)
- FL: T1 data of poor quality
- MD: T1 data of poor quality
- TR: T1 data of poor quality (bad stitching)
- AP: T2 data of poor quality
- TT: T2 data of poor quality

## Adding a new subject to the pipeline

This is a step-by-step procedure for adding a subject to the template data set. The creation of the template is an automatic procedure which, for each subject, takes as input the DICOM PATHs and returns the data completely processed (i.e., straightened, registed to the template space and vertebrae aligned). However, it is inevitable to add for each subject small adjustements (see below).

### Summary of the steps

#### Files to be created
- **crop.txt**: ASCII file to indicate where to crop the data
- **centerline_propseg_RPI.nii.gz**: labeled NIFTI image (binary) to help propseg generating the segmentation of the spinal cord
- **labels_vertebral.nii.gz**: labeled NIFTI image (not binary) to indicate fiducial markers corresponding to the brainstem and the vertebral bodies (from C2-C3 to T12-L1)

N.B.: Those files need to be generated for both contrasts T1 and T2 (i.e., 6 files in total).

#### Data structure
~~~~
PATH_INFO
    |--- T1
    |     |---<subject>
    |            |--- crop.txt 
    |            |--- centerline_propseg_RPI.nii.gz 
    |            |--- labels_vertebral.nii.gz
    |            |--- (labels_updown.nii.gz) (this file is not required anymore)
    |
    |--- T2
    |     |---<subject>
    |            |--- crop.txt 
    |            |--- centerline_propseg_RPI.nii.gz 
    |            |--- labels_vertebral.nii.gz
    |            |--- (labels_updown.nii.gz) (this file is not required anymore)
~~~~

#### Files to modify
- **preprocess_data_template.py**: batch that creates all the files that you will have bravely generated yourself (see above)
- **pipeline_template.py**: batch used to create the template for both T1 and T2 data.
  - you need to add your subject to the variable SUBJECT_LIST

### Detailed pocedure

All subjects need to have both a full T1 and a full T2 image.

Step-by-step procedure (to do for each contrast):

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
* Report where you are cropping the image in the file **crop.txt** using this format:
  * zmin_anatomic,zmax_anatomic  (e.g.: 15,623 if you are cropping between slices 15 and 623).
    * If there is a need to crop along y axis (as for some data from marseille that present artefacts) please specify as follow: 
      * zmin_anatomic,zmax_anatomic,ymin_anatomic, ymax_anatomic (e.g.: 15,623,30,200 if you are adding a crop along y axis between slices 30 and 200).
* From the cropped image ``data_RPI_crop.nii.gz``, create a label file ``centerline_propseg_RPI.nii.gz`` that will be used to initiate the segmentation of propseg. 
  * Open ``data_RPI_crop.nii.gz`` with flsview and create a mask (cmd+c).
  * Put labels of value 1 at the center of the cord all along the spinal cord, approximately every 30 slices. Note that you need to put a label at the first slice (z=0) and at the last slice (z=nz) as this file will be used for the straightening of the image.
* Generate the segmentation using propseg
  * ``sct_propseg -i data_RPI_crop.nii.gz -t XXX -init-centerline centerline_propseg_RPI.nii.gz`` (here, XXX is t1 or t2 depending on the contrast)
  * Check if the segmentation is correct. Since propseg often diverges at edges, you need to crop the segmentation and report the crop values in the file ``crop.txt`` that was previously created. Use this format:
    * zmin_anatomic,zmax_anatomic,zmin_seg,zmax_seg (or: zmin_anatomic,zmax_anatomic,ymin_anatomic,ymax_anatomic,zmin_seg,zmax_seg if you cropped along y at the previous step).
      * N.B.: If you only want to crop the segmentation at the bottom, you can write **max** instead of zmax_seg (e.g.: 15,max  if you are cropping at slice 15).
* You have now generated all the necessary files for the pipeline to work. Test the pipeline’s **do_preprocessing** in file ``pipeline_template.py``. To do so:
  * Open ``pipeline_template.py``
    * Comment variable: ``SUBJECTS_LIST`` and create a temporary variable with only your subject to test.
    * Under ``def main():``, comment all processes, except ``do_preprocessing('T1')`` (or ``do_preprocessing('T2')``)
  * Run ``pipeline_template.py`` ,  step for this subject and make sure results are good. Notably: 
    * Checking the resulting image: ``data_RPI_crop_normalized_straight_crop.nii.gz``
    * Checking that no vertebral label has disappeared in the process (i.e. that labels_vertebral_dilated_reg_2point_crop.nii.gz still contains 20 labels).
      * ``sct_label_utils -i labels_vertebral_dilated_reg_2point_crop.nii.gz -t display-voxel``
