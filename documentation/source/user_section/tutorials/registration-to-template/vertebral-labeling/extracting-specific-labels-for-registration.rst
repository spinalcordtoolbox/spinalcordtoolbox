.. _extracting-specific-labels:

Extracting specific labels for registration
###########################################

Following the recommendations from the :ref:`previous page<how-many-labels>`, we can use ``sct_label_utils`` to create a new label image containing just 2 of the vertebral levels.

.. code:: sh

   sct_label_utils -i t2_seg_labeled.nii.gz -vert-body 3,9 -o t2_labels_vert.nii.gz

:Input arguments:
   - ``-i`` : Input image containing a spinal cord labeled with vertebral levels
   - ``-vert-body`` : The top and bottom vertebral levels to create new single-voxel labels for. The labels will be centered within the vertebral body. Choose labels based on your region of interest. Here, we have chosen ``3,9``, indicating C3 to T1 as our region of interest.
   - ``-o`` : Output filename

:Output files/folders:
   - ``t2_labels_vert.nii.gz`` : Image containing 2 single-voxel vertebral labels, centered within the C3 and T1 vertebral bodies.

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using :ref:`fsleyes-instructions`. However, it is worth noting that because the output labels are single-voxel, you may need to use zoom to see them clearly in FSLeyes.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/vertebral-labeling/io-sct_label_utils.png
   :align: center

   Input/output images for ``sct_label_utils``