Spinal nerve rootlets segmentation
##################################

SCT provides a deep learning model for the segmentation of spinal nerve rootlets from T2-weighted and MP2RAGE images.
The model is available in SCT v7.0 and higher via ``sct_deepseg rootlets``.
In the previous SCT versions (SCT v6.2 and higher), the model segmented only T2-weighted and was available via via ``sct_deepseg -task seg_spinal_rootlets_t2w``.

This model was trained on 3D T2-weighted and MP2RAGE images (UNIT1, INV1, INV2) and provides level-specific semantic segmentation (i.e., 2: C2 rootlet, 3: C3 rootlet, etc.) of the dorsal and ventral spinal nerve rootlets C2-T1.

Run the following command to segment the spinal nerve rootlets from the input image:

.. code:: sh

   sct_deepseg rootlets -i t2.nii.gz -qc ~/qc_singleSubj

:Input arguments:
    - ``rootlets`` : Task to perform. In our case, we use the ``rootlets`` task.
    - ``-i`` : Input T2w image
    - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the segmentation slice-by-slice

:Output files/folders:
    - ``t2_seg.nii.gz`` : 3D level-specific segmentation (i.e., 2: C2 rootlet, 3: C3 rootlet, etc.) of the dorsal and ventral spinal nerve rootlets
    - ``t2_seg.json`` : JSON file containing details about the segmentation model


Details:

* `Valošek, J., et al. Imaging Neuroscience (2024) <https://doi.org/10.1162/imag_a_00218>`_