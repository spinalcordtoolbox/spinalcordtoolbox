SCIseg - lesion segmentation in spinal cord injury (SCI): ``sct_deepseg -task seg_sc_lesion_t2w_sci``
#####################################################################################################

As its name suggests, ``SCIseg`` is based on deep learning and is available in SCT v6.2 and higher via ``sct_deepseg -task seg_sc_lesion_t2w_sci``. In SCT v6.4, the model was updated to ``SCIsegV2``, the command remains the same.

The model was trained on raw T2-weighted images of SCI patients comprising traumatic (acute preoperative, intermediate, chronic) and non-traumatic (ischemic SCI and degenerative cervical myelopathy, DCM) SCI lesions.

The data included images with heterogeneous resolutions (axial/sagittal/isotropic) and scanner strengths (1T/1.5T/3T).

Given an input image, the model segments **both** the lesion and the spinal cord.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/lesion-analysis/sciseg.png
  :align: center
  :figwidth: 60%

You can try ``SCIseg`` on your own T2w image using the following command:

.. code:: sh

   sct_deepseg -i t2.nii.gz -task seg_sc_lesion_t2w_sci -qc ./qc

:Input arguments:
   - ``-i`` : Input T2w image
   - ``-task`` : Task to perform. In our case, we use the ``SCIseg`` model via the ``seg_sc_lesion_t2w_sci`` task
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the segmentation slice-by-slice

:Output files/folders:
   - ``t2_sc_seg.nii.gz`` : 3D binary mask of the segmented spinal cord
   - ``t2_lesion_seg.nii.gz`` : 3D binary mask of the segmented lesion
   - ``t2_lesion_seg.json`` : JSON file containing details about the segmentation model


Details:

* **SCIsegV1:** `Enamundram, N.K.*, Valošek, J.*, et al. medRxiv (2024) <https://doi.org/10.1101/2024.01.03.24300794>`_
* **SCIsegV2:** `Enamundram, N.K.*, Valošek, J.*, et al. arXiv (2024) <https://doi.org/10.48550/arXiv.2407.17265>`_
