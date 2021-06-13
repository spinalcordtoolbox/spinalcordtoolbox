Registering dMRI data to the PAM50 template
###########################################

Before the PAM50 template can be used to extract values from specific regions within the DTI images, it must first be registered to the space of the dMRI data.

Segmenting the spinal cord
--------------------------

As a prerequisite step, we segment the spinal cord from the 3D mean image generated during motion correction. Providing input segmentations to the registration command is optional, but doing so will help to improve the accuracy of the registration.

.. code::

   sct_deepseg_sc -i dmri_crop_moco_dwi_mean.nii.gz -c dwi -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : The input image.
   - ``-c`` : The contrast of the input image.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``dmri_crop_moco_dwi_mean_seg.nii.gz`` : An mask image containing the segmented spinal cord.

Registering the template to the DTI space
-----------------------------------------

.. code:: sh

   sct_register_multimodal -i "${SCT_DIR}/data/PAM50/template/PAM50_t1.nii.gz" \
                           -iseg "${SCT_DIR}/data/PAM50/template/PAM50_cord.nii.gz" \
                           -d dmri_crop_moco_dwi_mean.nii.gz \
                           -dseg dmri_crop_moco_dwi_mean_seg.nii.gz \
                           -initwarp ../t2s/warp_template2t2s.nii.gz \
                           -initwarpinv ../t2s/warp_t2s2template.nii.gz \
                           -param step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3 \
                           -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Source image. Here, we select the T1 version of the PAM50 template, because the T1 contrast is the closest visual match to our DTI data.
   - ``-iseg`` : Segmented spinal cord for the source image. Here, we use the PAM50 segmented spinal cord volume.
   - ``-d`` : Destination image.
   - ``-dseg`` : Segmented spinal cord for the destination image.
   - ``-initwarp`` : TODO: Is this necessary? I understand that it's meant to improve the registration with the GM/WM-informed warping field, but I worry that it makes this tutorial unnecessarily complex for people who will read it as a standalone tutorial. So, I'm wondering if the T2* warping fields should be kept to the GM/WM tutorials only.
   - ``-initwarpinv`` : TODO: Is this necessary? (See above.)
   - ``-param`` : Here, we will tweak the default registration parameters to specify a different nonrigid deformation. The important change is ``algo=centermass``. Because the template object is already "preregistered" from the previous tutorial (using ``-initwarp``), the benefits of the default ``algo=centermassrot`` have already been realized. So, we specify a different algorithm in step 1 to exclude the unnecessary rotation. (TODO: Change this if ``-initwarp`` is also changed.)
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``PAM50_t1_reg.nii.gz`` : The PAM50 template image, registered to the space of dMRI data.
   - ``dmri_crop_moco_dwi_mean_reg.nii.gz`` : The mean dMRI image, registered to the space of the PAM50 template.
   - ``warp_PAM50_t12dmri_crop_moco_dwi_mean.nii.gz`` : The warping field to transform the PAM50 template to the dMRI space.
   - ``warp_dmri_crop_moco_dwi_mean2PAM50_t1.nii.gz`` : The warping field to transform the dMRI data to the PAM50 template space.

Renaming the output files
-------------------------

Finally, we rename the warping fields for convenience, since the automatically generated filenames are a little verbose.

.. code::

   mv warp_PAM50_t12dmri_crop_moco_dwi_mean.nii.gz warp_template2dmri.nii.gz
   mv warp_dmri_crop_moco_dwi_mean2PAM50_t1.nii.gz warp_dmri2template.nii.gz

