Lesion segmentation in multiple sclerosis (MS)
##############################################

SCT provides several deep learning-based algorithms to segment lesions in multiple sclerosis (MS), namely:

* ``sct_deepseg_lesion`` - trained on T2w and T2star images. Details: `NeuroImage, C., et al. NeuroImage (2019) <https://doi.org/10.1016/j.neuroimage.2018.09.081>`_.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/sct_deepseg_sc_steps.png
   :align: center
   :figwidth: 60%

* ``sct_deepseg -task seg_sc_ms_lesion_stir_psir`` - trained on sagittal STIR and PSIR images. It is a region-based model, outputting a single segmentation image containing 2 classes representing the spinal cord and MS lesions. Details: https://github.com/ivadomed/canproco.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/lesion-analysis/seg_sc_ms_lesion_stir_psir.gif
   :align: center
   :figwidth: 60%

* ``sct_deepseg -task seg_ms_lesion_mp2rage`` - trained on cropped MP2RAGE-UNIT1 images. Details: https://github.com/ivadomed/model_seg_ms_mp2rage.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/lesion-analysis/model_seg_ms_mp2rage.png
   :align: center
   :figwidth: 60%

.. note::

   The ``sct_deepseg_lesion`` and ``sct_deepseg -task seg_ms_lesion_mp2rage`` algorithms segment only the MS lesion(s).
   You can use the contrast-agnostic model (``sct_deepseg -task seg_sc_contrast_agnostic``) or ``sct_deepseg_sc`` to segment the spinal cord.

You can try ``sct_deepseg_lesion`` on your own T2w or T2star image using the following command:

.. code:: sh

   sct_deepseg_lesion -i t2.nii.gz -c t2

:Input arguments:
   - ``-i`` : Input T2w image
   - ``-c`` : Contrast of the input image

:Output files/folders:
   - ``t2_lesionseg.nii.gz`` : 3D binary mask of the segmented lesion

You can try ``seg_sc_ms_lesion_stir_psir`` on your own STIR or PSIR image using the following command:

.. code:: sh

   sct_deepseg -i psir.nii.gz -task seg_sc_ms_lesion_stir_psir -qc ./qc

:Input arguments:
   - ``-i`` : Input PSIR (or STIR) image
   - ``-task`` : Task to perform. In this case, we use the ``seg_sc_ms_lesion_stir_psir`` model
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the segmentation slice-by-slice

You can try ``seg_ms_lesion_mp2rage`` on your own MP2RAGE UNIT1 image using the following commands.
As the model was trained on cropped images, we recommend cropping the input image before running the segmentation.

.. code:: sh

   sct_deepseg -i IMAGE_UNIT1 -task seg_sc_contrast_agnostic -o IMAGE_seg
   sct_crop_image -i IMAGE_UNIT1 -m IMAGE_seg -dilate 30x30x5
   sct_deepseg -i IMAGE_UNIT1 -task seg_ms_lesion_mp2rage -qc ./qc

:Input arguments:
    - ``-i`` : Input MP2RAGE UNIT1 image
    - ``-task`` : Task to perform. In this case, we use the ``seg_ms_lesion_mp2rage`` model
    - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the segmentation slice-by-slice