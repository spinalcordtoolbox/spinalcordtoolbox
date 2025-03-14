.. _extracting-specific-labels:

..
    comment:: In the 2024 SCT Course, this section was moved to the start of
              the registration section to improve the flow of the course.
              However, it is kept in the "labeling" section in the web
              tutorials because each tutorial section is intended to be
              self-contained. Having a discrepancy here might not be the best,
              so we should reflect on whether we want to keep things this way.

Extracting specific labels for registration
###########################################

Following the recommendations from the :ref:`previous page<how-many-labels>`, we can use :ref:`sct_label_utils` to create a new label image containing just 2 of the vertebral levels.

.. code:: sh

   sct_label_utils -i t2_seg_labeled.nii.gz -vert-body 3,9 -o t2_labels_vert.nii.gz

:Input arguments:
   - ``-i`` : Input image containing a spinal cord labeled with vertebral levels
   - ``-vert-body`` : The top and bottom vertebral levels to create new single-voxel labels for. The labels will be centered within the vertebral body. Choose labels based on your region of interest. Here, we have chosen ``3,9``, indicating C3 to T1 as our region of interest.
   - ``-o`` : Output filename

:Output files/folders:
   - ``t2_labels_vert.nii.gz`` : Image containing 2 single-voxel vertebral labels, centered within the C3 and T1 vertebral bodies.

Once the command has finished, if you have ``fsleyes`` installed, at the bottom of your terminal there will be instructions for inspecting the results using :ref:`fsleyes-instructions`. However, because the output labels are single-voxel, you may need to use zoom to see them clearly in FSLeyes.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/vertebral-labeling/io-sct_label_utils.png
   :align: center

   Input/output images for :ref:`sct_label_utils`