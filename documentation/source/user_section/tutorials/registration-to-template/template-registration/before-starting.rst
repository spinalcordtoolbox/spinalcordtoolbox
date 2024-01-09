Before starting this tutorial
#############################

This tutorial is intended to be run following the previous :ref:`spinalcord-segmentation` and :ref:`vertebral-labeling` tutorials, as registration to the PAM50 template relies on having a spinal cord segmentation mask as well as a point-label file representing vertebral level landmarks.

You can either run those tutorials first, or download the necessary files below:

#. Read through the following pages to familiarize yourself with key SCT concepts:

   * :ref:`pam50`: An overview of the PAM50 template's features, as well as context for why the template is used.
   * :ref:`warping-fields`: Background information on the image transformation format used by the registration process.

#. Make sure that you have the following files in your working directory:

   * ``single_subject/data/t2/t2.nii.gz``: An anatomical spinal cord scan in the T2 contrast.
   * ``single_subject/data/t2/t2_seg.nii.gz``: A 3D binary mask for the spinal cord.
   * ``single_subject/data/t2/t2_labels_vert.nii.gz`` : Image containing the 2 single-voxel vertebral labels

   You can get these files by downloading :sct_tutorial_data:`data_template-registration.zip`.


#. Open a terminal and navigate to the ``single_subject/data/t2/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/t2/