Before starting this tutorial
#############################

1. Read through the following pages to familiarize yourself with key SCT concepts:

    * :ref:`pam50`: An overview of the PAM50 template's features, as well as context for why the template is used.
    * :ref:`warping-fields`: Background information on the image transformation format used by the registration process.
    * :ref:`qc`: Primer for SCT's Quality Control interface. After each step of this tutorial, you will be able to open a QC report that lets you easily evaluate the results of each command.

2. Make sure that you have the following files in your working directory:

   * ``single_subject/data/t2/t2.nii.gz``: An anatomical spinal cord scan in the T1 contrast.
   * ``single_subject/data/t2/t2_seg.nii.gz``: An 3D binary mask for the spinal cord.
   * ``t2_labels_vert.nii.gz`` : Image containing the 2 single-voxel vertebral labels

   You can get these files by downloading :sct_tutorial_data:`data_template-registration.zip`.

.. note:: If you are :ref:`completing all of SCT's tutorials in sequence <completing-the-tutorials-in-sequence>`, your working directory should already contain the files needed for this tutorial.

3. Open a terminal and navigate to the ``/single_subject/data/t2/`` directory.