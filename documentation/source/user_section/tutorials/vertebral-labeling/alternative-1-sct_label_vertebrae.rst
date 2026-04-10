.. _vert-labeling-section:

Alternative #1: ``sct_label_vertebrae``
#######################################

This section describes the legacy labeling workflow using :ref:`sct_label_vertebrae`.

This approach remains useful when you already have a cord segmentation and want to label that segmentation with detected discs automatically. For most new workflows, we recommend using :ref:`sct_deepseg_spine` first, then using the generated disc labels directly.

Algorithm steps
---------------

#. **Straightening**: The spinal cord is straightened to make it easier to use a moving window-based approach in a subsequent step.
#. **C2-C3 disc detection:** The C2-C3 disc is used as a starting point because it is a distinct disc that is easy to detect (compared to, say, the T7-T9 discs, which are indistinct compared to one another).
#. **Labeling of neighbouring discs**: The neighbouring discs are found using a similarity measure with the PAM50 template at each specific level.
#. **Un-straightening**: Finally, the spinal cord and the labeled segmentation are both un-straightened, and the labels are saved to image files.

Applying the algorithm
----------------------

To apply :ref:`sct_label_vertebrae` to our T2 data, the following command is used:

.. code:: sh

   sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2

:Input arguments:
   - ``-i`` : Input anatomical image.
   - ``-s`` : Spinal cord segmentation corresponding to the input image.
   - ``-c`` : Image contrast (for example, ``t2``).

:Output files/folders:
   - ``t2_seg_labeled.nii.gz`` : Labeled spinal cord segmentation.

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using :ref:`Quality Control (QC) <qc>` reports. Optionally, If you have :ref:`fsleyes-instructions` installed, a ``fsleyes`` command will printed as well.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/vertebral-labeling/io-sct_label_vertebrae.png
   :align: center

   Input/output images for :ref:`sct_label_vertebrae`