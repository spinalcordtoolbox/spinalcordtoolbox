Applying the labeling algorithm
###############################

.. code:: sh

   sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2 -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Input image
   - ``-s`` : Segmented spinal cord corresponding to the input image
   - ``-c`` : Contrast of the input image
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``straight_ref.nii.gz`` : The straightened input image produced by the intermediate straightening step. Can be re-used by other SCT functions that need a straight reference space.
   - ``warp_curve2straight.nii.gz`` : The 4D warping field that defines the transform from the original curved anatomical image to the straightened image.
   - ``warp_straight2curve.nii.gz`` : The 4D warping field that defines the inverse transform from the straightened anatomical image back to the original curved image.
   - ``straightening.cache`` : SCT functions that require straightening will check for this file. If it is present in the working directory, ``straight_ref.nii.gz`` and the two warping fields will be re-used, saving processing time.
   - ``t2_seg_labeled.nii.gz`` : Image containing the labeled spinal cord. Each voxel of the segmented spinal cord is labeled with a vertebral level as though the vertebrae were projected onto the spinal cord. The convention for label values is C3-->3, C4-->4, etc.
   - ``t2_seg_labeled_discs.nii.gz`` : Image containing single-voxel intervertebral disc labels (without the segmented spinal cord). Each label is centered within the disc. The convention for label values is C2/C3-->3, C3/C4-->4, etc. This file also contains additional labels (such as the pontomedullary junction and groove), but these are not yet used.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-sct_label_vertebrae.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_label_vertebrae``.