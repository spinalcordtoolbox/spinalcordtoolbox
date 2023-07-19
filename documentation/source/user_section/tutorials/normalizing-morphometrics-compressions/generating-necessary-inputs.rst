.. _generating-necessary-inputs:

Generate the 3 necessary input files from this subject file for sct_compute_compression
#######################################################################################

To use sct_compute_compression, we need the 3 following input files:


1. Spinal cord segmentation
2. Vertebral labels
3. Compression labels (t2_compressed_compression_labels.nii.gz)


## 1. Spinal cord segmentation
------------------------------
.. code:: sh

   # 1. Generate spinal cord segmentation
      sct_deepseg_sc -i t2_compressed.nii.gz -c t2

:Input arguments:
   - ``-i`` : Input image.
   - ``-c`` : Contrast of the input image.


:Output files/folders:
   - ``t2_compressed_seg.nii.gz`` : Spinal cord segmentation.


## 2. Label the spinal cord vertebral levels
--------------------------------------------
.. code:: sh

   # 2. Generate spinal cord vertebral labeling
      sct_label_vertebrae -i t2_compressed.nii.gz -s t2_compressed_seg.nii.gz -c t2

:Input arguments:
   - ``-i`` : The input image file.
   - ``-s`` : Spinal cord segmentation.
   - ``-c`` : Contrast of the input image.


:Output files/folders:
   - ``sub-twh018_T2w_seg.nii.gz`` : Spinal cord segmentation.

## 3. Generate spinal cord compression labels
---------------------------------------------
1. Open the image in FSLeyes. If the image is not 3D, open both axial and sagittal images in FSLeyes.

.. code:: sh

   fsleyes t2_compressed.nii.gz &

2. Look at the clinical data provided with compression information to know at which level the compressions are located.
3. Locate the compression in the sagittal view.
4. In the axial view, toggle to the maximum compressed slice around the compression.
5. Click on alt+E (Windows) or option+E (macOS) to open edit mode. Select size 1 and click on the pencil.
6. Create an empty mask alt+N (Windows) or option+N (macOS).
7. Place the label at the center of the spinal cord of the axial image.
8. Repeat steps 3 to 7 for the number of compressions.
9. Save and quit.


You can also look at the example using an automatic script when multiples images require compression labels: 
https://github.com/spinalcordtoolbox/manual-correction/wiki#manual-labeling-of-spinal-cord-compression
