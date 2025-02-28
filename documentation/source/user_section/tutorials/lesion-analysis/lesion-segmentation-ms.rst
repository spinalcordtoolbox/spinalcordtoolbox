Lesion segmentation in multiple sclerosis (MS)
##############################################

SCT provides several deep learning-based algorithms to segment lesions in multiple sclerosis (MS). Depending on the
image contrast, you can use the following algorithms:

Contrast-agnostic
*****************

As described in the :ref:`contrasts` section, SCT has moved towards developing contrast-agnostic segmentation tools. The ``seg_ms_lesion`` model is SCT's effort to create a contrast-agnostic lesion segmentation tool that can be used on any type of image (T1, T2, T2*, etc.), in order to ensure consistent lesion segmentation results between contrasts.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_lesion.png
   :align: center
   :figwidth: 60%

You can try the ``seg_ms_lesion`` on the sample T2w image using the following command:

.. code:: sh

   sct_deepseg -i t2.nii.gz -task seg_ms_lesion -qc ./qc

:Input arguments:
    - ``-i`` : Input T2w image with fake lesion
    - ``-task`` : Task to perform. In this case, we use the ``seg_ms_lesion`` model
    - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the segmentation slice-by-slice

----

MP2RAGE-UNIT1
*************

The algorithm ``seg_ms_lesion_mp2rage`` was trained on cropped MP2RAGE-UNIT1 images. Details: `Cohen-Adad, J., et al. Zenodo release (2023) <https://zenodo.org/doi/10.5281/zenodo.8376753>`_.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/lesion-analysis/model_seg_ms_mp2rage.png
   :align: center
   :figwidth: 60%

You can try ``seg_ms_lesion_mp2rage`` on your *own* MP2RAGE UNIT1 image using the following commands.
As the model was trained on cropped images, we recommend cropping the input image before running the segmentation.

.. code:: sh

   sct_deepseg -i IMAGE_UNIT1 -task seg_sc_contrast_agnostic -o IMAGE_seg
   sct_crop_image -i IMAGE_UNIT1 -m IMAGE_seg -dilate 30x30x5
   sct_deepseg -i IMAGE_UNIT1 -task seg_ms_lesion_mp2rage -qc ./qc

:Input arguments:
    - ``-i`` : Input MP2RAGE UNIT1 image
    - ``-task`` : Task to perform. In this case, we use the ``seg_ms_lesion_mp2rage`` model
    - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the segmentation slice-by-slice

----

T2w and T2star
**************

The legacy CLI tool ``sct_deepseg_lesion`` was trained on T2w and T2star images. Details: `Gros, C., et al. NeuroImage (2019) <https://doi.org/10.1016/j.neuroimage.2018.09.081>`_.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/sct_deepseg_sc_steps.png
   :align: center
   :figwidth: 80%

Run the following command to segment the lesion using ``sct_deepseg_lesion`` from the input image:

.. code:: sh

   sct_deepseg_lesion -i t2.nii.gz -c t2

:Input arguments:
   - ``-i`` : Input T2w image with fake lesion
   - ``-c`` : Contrast of the input image

:Output files/folders:
   - ``t2_lesionseg.nii.gz`` : 3D binary mask of the segmented lesion
   - ``t2_res_RPI_seg.nii.gz`` : intermediate segmentation file -- you can ignore this file
   - ``t2_RPI_seg.nii.gz`` : intermediate segmentation file -- you can ignore this file