TEMPLATE PREPROCESSING
=============



    pipeline_template.py
    =======

        - Process is done for each subject. The process is done independently for T2 and T1 which leads to the creation of two templates: one for T2 and one for T1. Then registering this two templates onto one another can be done.

        - the list of commands is included in the file: pipeline_template.py


        Two folders must be precised for the pipeline to work:
        - PATH_INFO: which will contain all the information necessary for each subject (it can be generated from scratch with the scripts preprocess_data_template.py)
        - PATH_OUTPUT: which will gather all the results from the template creation process.

        Data organisation for PATH_INFO is as follow:
        PATH_INFO/
        ........./T1
        ............/subject
        ..................../crop.txt
        ..................../centerline_propseg_RPI.nii.gz
        ..................../labels_vertebral.nii.gz
        ..................../labels_updown.nii.gz (optional now and not advised as it can be incorporated into centerline_propseg_RPI.nii.gz)
        ........./T2
        ............/subject
        ..................../crop.txt
        ..................../centerline_propseg_RPI.nii.gz
        ..................../labels_vertebral.nii.gz
        ..................../labels_updown.nii.gz (optional now and not advised as it can be incorporated into centerline_propseg_RPI.nii.gz)



        PATH_INFO/subject must contains those elements:
        - crop.txt: ASCII txt file that indicates zmin and zmax for cropping the anatomic image and the segmentation . Format: zmin_anatomic,zmax_anatomic,zmin_seg,zmax_seg  If there is a need to crop along y axis the RPI image, please specify as follow: zmin_anatomic,zmax_anatomic,zmin_seg,zmax_seg,ymin_anatomic,ymax_anatomic
            -> cropping the anatomic image must be done a little above the brainstem and at L2-L3 (if the size of the data allows it)
        - centerline_propseg_RPI.nii.gz: a label file to help propseg initiation.
        - labels_vertebral.nii.gz: a label file to localise the vertebral levels.


        List of steps of the pipeline is below. The name of the pipeline is pipeline_template.py. Functions involved are in brackets ():


        1. Import dicom files and convert to NIFTI format (dcm2nii) (output: data_RPI.nii.gz).
        2. Change orientation to RPI (sct_orientation).
        NB: You must create a label file from data_RPI.nii.gz to localise each vertebral level.
        These are placed on the left side of the vertebral body, at the edge of the cartilage separating two vertebra. The value of the label corresponds to the level. There are 20 labels: [name of point at top] + PMJ + 18 labels of vertebral level going until the frontier T12/L1 I.e., Brainstem [name first label]=1, (PMJ)=2, C2/C3=3, C3/C4=4, C4/C5=5, C5/C6=6, T1/T2=7, T2/T3=8, T3/T4=9 ... T11/T12=19, T12/L1=20. Screenshots can be seen at: /dev/template_preprocessing.
        (output: labels_vertebral.nii.gz, to be saved in PATH_INFO/[contrast]/subject)
        3. Crop image a little above the brainstem and a little under L2/L3 vertebral disk (sct_crop_image)(output: data_RPI_crop.nii.gz).
        N.B.: you should note at which slice you are cropping in the crop.txt file in the PATH_INFO folder.
        N.B.: You must create a label file from data_RPI_crop.nii.gz to help propseg following the spinal cord (output: centerline_propseg_RPI.nii.gz). First label and last label must be respectively placed at z=0 and z=max (important for what follows). Distance between labels depends on the image quality but 30 slices (i.e. 3cm) is usually a good setting for the settings of propseg.
        4. Process segmentation of the spinal cord (sct_propseg -i data_RPI_crop.nii.gz -init-centerline centerline_propseg_RI.nii.gz)(output: data_RPI_crop_seg.nii.gz)
        5. Erase three bottom and top slices from the segmentation to avoid edge effects from propseg (output: data_RPI_crop_seg_mod.nii.gz)
        6. Check segmentation results and crop if needed (sct_crop_image)(output: data_RPI_crop_seg_mod_crop.nii.gz)
        N.B.: you should note at which slice you are cropping in the crop.txt file in the PATH_INFO folder. If no need to crop, just put 0 and max for zmin_seg and zmax_seg.
        7. Concatenation of segmentation and original label file centerline_propseg_RPI.nii.gz (fslmaths -add)(output: seg_and_labels.nii.gz).
        8. Extraction of the centerline for normalizing intensity along the spinalcord before straightening (sct_get_centerline_from_labels)(output: generated_centerline.nii.gz)
        8. Normalize intensity along z (sct_normalize -c generated_centerline.nii.gz)(output: data_RPI_crop_normalized.nii.gz)
        9. Straighten volume using this concatenation (sct_straighten_spinalcord -c seg_and_labels.nii.gz -a nurbs)(output: data_RPI_crop_normalized_straight.nii.gz).
        10. Apply those transformation to labels_vertebral.nii.gz:
            -crop with zmin_anatomic and zmax_anatomic (sct_crop_image)(output: labels_vertebral_crop.nii.gz)
            -dilate labels before applying warping fields to avoid the disapearance of a label (fslmaths -dilF)(output: labels_vertebral_crop_dilated.nii.gz)
            -apply warping field curve2straight (sct_apply_transfo -x nn) (output: labels_vertebral_crop_dialeted_reg.nii.gz)
            -select center of mass of labels volume due to past dilatation (sct_label_utils -t cubic-to-point)(output: labels_vertebral_crop_dilated_reg_2point.nii.gz)
        11. Apply transfo to seg_and_labels.nii.gz (sct_apply_transfo)(output: seg_and_labels_reg.nii.gz).
        12. Crop volumes one more time to erase the blank spaces due to the straightening (sct_crop_image ). To do this, the pipeline uses your straight centerline as input and returns the slices number of the upper and lower nonzero points. It then crops your volume (sct_crop_image)(outputs: data_RPI_crop_normalized_straight_crop.nii.gz, labels_vertebral_crop_dilated_reg_crop.nii.gz).
        13. For each subject of your list, the pipeline creates a cross of 5 mm at the top label from labels_vertebral_crop_dilated_reg_crop.nii.gz in the center of the plan xOy and a point at the bottom label from labels_vertebral.nii.gz in the center of the plan xOy (sct_create_cross)(output:landmark_native.nii.gz).
        14. Calculate mean position of top and bottom labels from your list of subjects to create cross on a template shape file (sct_create_cross)
        15. Push the straightened volumes into the template space. The template space has crosses in it for registration. (sct_push_into_template_space)(outputs: data_RPI_crop_straight_normalized_crop_2temp.nii.gz, labels_vertebral_crop_dilated_reg_crop_2temp.nii.gz)
        16. Apply cubic to point to the label file as it now presents cubic group of labels instead of discrete labels (sct_label_utils -t cubic-to-point) (output: labels_vertebral_dilated_reg_2point_crop_2temp.nii.gz)
        16. Use sct_average_levels to calculate the mean landmarks for vertebral levels in the template space. This scripts take the folder containing all the masks created in previous step and for a given landmark it averages values across all subjects and put a landmark at this averaged value. You only have to do this once for a given preprocessing process. If you change the preprocessing or if you add subjects you have 2 choices : assume that it will not change the average too much and use the previous mask, or generate a new one. (sct_average_levels) (output: template_landmarks.nii.gz)
        17. Use sct_align_vertebrae -t SyN (transformation) -w spline (interpolation) to align the vertebrae using transformation along Z (sct_align_vertebrae -t SyN -w sline -R template_landmarks.nii.gz)(output: <subject>_aligned_normalized.nii.gz)


        All data are located in:
        ~~~
        tatum: /Users/tamag/data/data_template/Results_template
        ~~~



    DATA
    =======

        Date: 1st July 2015
        Number of subject: 45

        List of subject contained in the template's data set:
        [['errsm_02', '/Volumes/data_shared/montreal_criugm/errsm_02/22-SPINE_T1', '/Volumes/data_shared/montreal_criugm/errsm_02/28-SPINE_T2'],['errsm_04', '/Volumes/data_shared/montreal_criugm/errsm_04/16-SPINE_memprage/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_04/18-SPINE_space'],\
        ['errsm_05', '/Volumes/data_shared/montreal_criugm/errsm_05/23-SPINE_MEMPRAGE/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_05/24-SPINE_SPACE'],['errsm_09', '/Volumes/data_shared/montreal_criugm/errsm_09/34-SPINE_MEMPRAGE2/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_09/33-SPINE_SPACE'],\
        ['errsm_10', '/Volumes/data_shared/montreal_criugm/errsm_10/13-SPINE_MEMPRAGE/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_10/20-SPINE_SPACE'], ['errsm_11', '/Volumes/data_shared/montreal_criugm/errsm_11/24-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_11/09-SPINE_T2'],\
        ['errsm_12', '/Volumes/data_shared/montreal_criugm/errsm_12/19-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_12/18-SPINE_T2'],['errsm_13', '/Volumes/data_shared/montreal_criugm/errsm_13/33-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_13/34-SPINE_T2'],\
        ['errsm_14', '/Volumes/data_shared/montreal_criugm/errsm_14/5002-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_14/5003-SPINE_T2'], ['errsm_16', '/Volumes/data_shared/montreal_criugm/errsm_16/23-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_16/39-SPINE_T2'],\
        ['errsm_17', '/Volumes/data_shared/montreal_criugm/errsm_17/41-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_17/42-SPINE_T2'], ['errsm_18', '/Volumes/data_shared/montreal_criugm/errsm_18/36-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_18/33-SPINE_T2'],\
        ['errsm_21', '/Volumes/data_shared/montreal_criugm/errsm_21/27-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_21/30-SPINE_T2'],['errsm_22', '/Volumes/data_shared/montreal_criugm/errsm_22/29-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_22/25-SPINE_T2'],\
        ['errsm_23', '/Volumes/data_shared/montreal_criugm/errsm_23/29-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_23/28-SPINE_T2'],['errsm_24', '/Volumes/data_shared/montreal_criugm/errsm_24/20-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_24/24-SPINE_T2'],\
        ['errsm_25', '/Volumes/data_shared/montreal_criugm/errsm_25/25-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_25/26-SPINE_T2'],['errsm_30', '/Volumes/data_shared/montreal_criugm/errsm_30/51-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_30/50-SPINE_T2'],\
        ['errsm_31', '/Volumes/data_shared/montreal_criugm/errsm_31/31-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_31/32-SPINE_T2'],['errsm_32', '/Volumes/data_shared/montreal_criugm/errsm_32/16-SPINE_T1/echo_2.09 ', '/Volumes/data_shared/montreal_criugm/errsm_32/19-SPINE_T2'],\
        ['errsm_33', '/Volumes/data_shared/montreal_criugm/errsm_33/30-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_33/31-SPINE_T2'],['1','/Volumes/data_shared/marseille/ED/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-101','/Volumes/data_shared/marseille/ED/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-65'],\
        ['ALT','/Volumes/data_shared/marseille/ALT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15','/Volumes/data_shared/marseille/ALT/01_0100_space-composing'],['JD','/Volumes/data_shared/marseille/JD/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/JD/01_0100_compo-space'],\
        ['JW','/Volumes/data_shared/marseille/JW/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/JW/01_0100_compo-space'],['MLL','/Volumes/data_shared/marseille/MLL_1016/01_0008_sc-mprage-1mm-2palliers-fov384-comp-sp-7','/Volumes/data_shared/marseille/MLL_1016/01_0100_t2-compo'],\
        ['MT','/Volumes/data_shared/marseille/MT/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/MT/01_0100_t2composing'],['TR', '/Volumes/data_shared/marseille/TR_T076/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/TR_T076/01_0016_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-19'],\
        ['T047','/Volumes/data_shared/marseille/T047/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5','/Volumes/data_shared/marseille/T047/01_0100_t2-3d-composing'],['VC','/Volumes/data_shared/marseille/VC/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-23','/Volumes/data_shared/marseille/VC/01_0008_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-113'],\
        ['VG','/Volumes/data_shared/marseille/VG/T1/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-15','/Volumes/data_shared/marseille/VG/T2/01_0024_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-11'],['VP','/Volumes/data_shared/marseille/VP/01_0011_sc-mprage-1mm-2palliers-fov384-comp-sp-25','/Volumes/data_shared/marseille/VP/01_0100_space-compo'],\
        ['pain_pilot_1','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/24-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot1/25-SPINE'],['pain_pilot_2','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/13-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot2/30-SPINE_T2'],\
        ['pain_pilot_4','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/33-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot4/32-SPINE_T2'],['TM', '/Volumes/data_shared/marseille/TM_T057c/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/TM_T057c/01_0105_t2-composing'],\
        ['errsm_20', '/Volumes/data_shared/montreal_criugm/errsm_20/12-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_20/34-SPINE_T2'],['pain_pilot_3','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/16-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot3/31-SPINE_T2'],\
        ['errsm_34','/Volumes/data_shared/montreal_criugm/errsm_34/41-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/errsm_34/40-SPINE_T2'],['errsm_35','/Volumes/data_shared/montreal_criugm/errsm_35/37-SPINE_T1/echo_2.09','/Volumes/data_shared/montreal_criugm/errsm_35/38-SPINE_T2'],\
        ['pain_pilot_7', '/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot7/32-SPINE_T1/echo_2.09', '/Volumes/data_shared/montreal_criugm/d_sp_pain_pilot7/33-SPINE_T2'], ['errsm_03', '/Volumes/data_shared/montreal_criugm/errsm_03/32-SPINE_all/echo_2.09', '/Volumes/data_shared/montreal_criugm/errsm_03/38-SPINE_all_space'],\
        ['FR', '/Volumes/data_shared/marseille/FR_T080/01_0039_sc-mprage-1mm-3palliers-fov384-comp-sp-13', '/Volumes/data_shared/marseille/FR_T080/01_0104_spine2'],['GB', '/Volumes/data_shared/marseille/GB_T083/01_0029_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/GB_T083/01_0033_sc-tse-spc-1mm-3palliers-fov256-nopat-comp-sp-7'],\
        ['T045', '/Volumes/data_shared/marseille/T045/01_0007_sc-mprage-1mm-2palliers-fov384-comp-sp-5', '/Volumes/data_shared/marseille/T045/01_0101_t2-3d-composing']]




    preprocess_data_template.py
    ========

        This file was made to ease the generation of the template. It enables one to generate all the necessary files that are subject specific from scratch.

         N.B.: IF SUBJECTS ARE ADDED TO THE TEMPLATE GENERATION, THE FILE preprocess_data_template SHOULD BE MODIFIED