.. _choosing-labels:

Extracting specific labels for registration
###########################################

Following the recommendations in :ref:`how-many-labels`, we can use ``sct_label_utils`` to create a new label image containing just 2 of the vertebral levels.

.. code:: sh

   sct_label_utils -i t2_seg_labeled.nii.gz -vert-body 3,9 -o t2_labels_vert.nii.gz

:Input arguments:
   - ``-i`` : Input image containing a spinal cord labeled with vertebral levels
   - ``-vert-body`` : The top and bottom vertebral levels to create new point labels for. Choose labels based on your region of interest. Here, we have chosen ``3,9`` (C3 to T1).
   - ``-o`` : Output filename

:Output files/folders:
   - ``t2_labels_vert.nii.gz`` : Image containing the 2 single-voxel vertebral labels

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-sct_label_utils.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_label_utils``.