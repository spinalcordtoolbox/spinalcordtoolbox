Alternative #1: Manually labeling the C2-C3 disc
################################################

If the fully automated approach fails, you can instead use a semi-automated approach by manually labeling the C2-C3 disc:

.. code:: sh

   sct_label_utils -i t2.nii.gz -create-viewer 3 -o label_c2c3.nii.gz \
                   -msg "Click at the posterior tip of C2/C3 inter-vertebral disc"

:Input arguments:
   * ``-i`` : The input anatomical image.
   * ``-create-viewer`` : This argument will open up an interactive GUI coordinate picker that will prompt you to label the disc corresponding to the ID `3`. (C2-C3 disc)
   * ``-msg`` : The text that will appear at the top of the window.
   * ``-o`` : The name of the output file.

:Output files/folders:
   * ``label_c2c3.nii.gz`` : An image containing the single-voxel label as selected in the GUI coordinate picker.

You can then pass this resulting file to the ``sct_label_vertebrae`` function via the ``-initlabel`` argument. This replaces the C2-C3 detection part of the algorithm, but preserves the remaining automated labeling steps.

.. code:: sh

   sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2 -initlabel label_c2c3.nii.gz -qc ~/qc_singleSubj

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/vertebral-labeling/vertebral-labeling-manual-c2c3.png
   :align: center

   Input/output images for ``sct_label_vertebrae --initlabel label_c2c3.nii.gz``
