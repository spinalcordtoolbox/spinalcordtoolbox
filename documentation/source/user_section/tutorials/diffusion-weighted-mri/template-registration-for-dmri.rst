Registering dMRI data to the PAM50 template
###########################################

Before the PAM50 template can be used to extract values from specific regions within the DTI images, it must first be registered to the space of the dMRI data.

Segmenting the spinal cord
--------------------------

As a prerequisite step, we run :ref:`sct_deepseg` once more, this time to segment the motion corrected 3D mean image. Note that providing an input segmentation to the registration command is optional, but doing so will help to improve the accuracy of the registration.

.. code::

   sct_deepseg spinalcord -i dmri_moco_dwi_mean.nii.gz -qc ~/qc_singleSubj

:Input arguments:
   - ``spinalcord``: Task to perform. Here, we are using ``spinalcord`` to segment the spinal cord. This task is contrast-agnostic, meaning it can be used on any type of image (T1, T2, T2*, etc.)
   - ``-i`` : The input image.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``dmri_moco_dwi_mean_seg.nii.gz`` : An mask image containing the segmented spinal cord.


Registering the template to the DTI space
-----------------------------------------

Usually, template registration would be performed using the :ref:`sct_register_to_template` command. That command is important because it matches the vertebral levels of the data to that of the PAM50 template. Unfortunately, though, because dMRI scans are typically acquired axially with thick slices, it is much more difficult to use :ref:`sct_label_vertebrae` to acquire the vertebral labels needed for the vertebral matching step.

To get around this limitation, we recommend that you first perform :ref:`vertebral labeling <vertebral-labeling>` and :ref:`template registration <template-registration>` using a different contrast for the same subject (e.g. T2 or T2* anatomical data, where vertebral levels are much more apparent). This will provide you with warping fields between the template and that contrast's data, which you can then re-use to initialize the dMRI registration via the ``-initwarp`` and ``-initwarpinv`` flags. Doing so provides all of the benefits of vertebral matching, without having to label the dMRI data directly.

Since we are starting the dMRI registration with the vertebral-matching transformation already applied, all that remains is fine-tuning for the dMRI data. So, here we use a different command: :ref:`sct_register_multimodal`. This command is designed to register any two images together, so it can be seen as the generalized counterpart to :ref:`sct_register_to_template`.

.. code:: sh

   sct_register_multimodal -i "${SCT_DIR}/data/PAM50/template/PAM50_t1.nii.gz" \
                           -iseg "${SCT_DIR}/data/PAM50/template/PAM50_cord.nii.gz" \
                           -d dmri_moco_dwi_mean.nii.gz \
                           -dseg dmri_moco_dwi_mean_seg.nii.gz \
                           -initwarp ../t2/warp_template2anat.nii.gz \
                           -initwarpinv ../t2/warp_anat2template.nii.gz \
                           -owarp warp_template2dmri.nii.gz \
                           -owarpinv warp_dmri2template.nii.gz \
                           -param step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3 \
                           -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Source image. Here, we select the T1 version of the PAM50 template, because the T1 contrast is the closest visual match to our DTI data.
   - ``-iseg`` : Segmented spinal cord for the source image. Here, we use the PAM50 segmented spinal cord volume.
   - ``-d`` : Destination image. Here, we select the motion-corrected mean DWI image for our dMRI data.
   - ``-dseg`` : Segmented spinal cord for the destination image.
   - ``-initwarp`` : A "source->destination" warping field (here, "template->data"), used to initialize the registration process. Here, we supply a previous T2 warping field generated by :ref:`sct_register_to_template`, because we want to start out with a transformation that includes vertebral level matching (plus the T2 contrast best defines the cord shape). However, any previous warping fields for the same subject could be used here, as long as vertebral levels were used during the template registration.
   - ``-initwarpinv`` : A "destination->source" warping field (here, "data->template"), used to initialize the registration process. Here, we supply a previous T2 warping field generated by :ref:`sct_register_to_template`, because we want to start out with a transformation that includes vertebral level matching (plus the T2 contrast best defines the cord shape). However, any previous warping fields for the same subject could be used here, as long as vertebral levels were used during the template registration.
   - ``-owarp``: The name of the output warping field. This is optional, and is only specified here to make the output filename a little clearer. By default, the filename would be automatically generated from the filenames ``-i`` and ``-d``, which in this case would be the (less clear) ``warp_PAM50_t12dmri_moco_dwi_mean.nii.gz``.
   - ``-owarpinv`` : The name of the output inverse warping field. This is specified for the same reasons as ``-owarp``.
   - ``-param`` : Here, we will tweak the default registration parameters to specify a different nonrigid deformation. The important change is ``algo=centermass``. Because the template object is already "preregistered" from the previous tutorial (using ``-initwarp``), the benefits of the default ``algo=centermassrot`` have already been realized. So, we specify a different algorithm in step 1 to exclude the unnecessary rotation.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``PAM50_t1_reg.nii.gz`` : The PAM50 template image, registered to the space of dMRI data.
   - ``dmri_moco_dwi_mean_reg.nii.gz`` : The mean dMRI image, registered to the space of the PAM50 template.
   - ``warp_template2dmri.nii.gz`` : The warping field to transform the PAM50 template to the dMRI space.
   - ``warp_dmri2template.nii.gz`` : The warping field to transform the dMRI data to the PAM50 template space.


Preparing the PAM50 template for metric extraction
--------------------------------------------------

Finally, we use the resulting warping field to transform the full template to the space of the dMRI data. This will allow us to use the PAM50 template and atlas to extract metrics from specific regions of the image.

.. code::

   sct_warp_template -d dmri_moco_dwi_mean.nii.gz -w warp_template2dmri.nii.gz -qc ~/qc_singleSubj

:Input arguments:
   - ``-d`` : Destination image the template will be warped to.
   - ``-w`` : Warping field (template space to anatomical space).
   - ``-a`` : Because ``-a 1`` is specified, the white and gray matter atlas will also be warped.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``label/template/`` : This directory contains the entirety of the PAM50 template, transformed into the DT space.
   - ``label/atlas/`` : This direct contains 36 NIFTI volumes for WM/GM tracts, transformed into the DT space.
