Before starting this tutorial
#############################

1. Read through the following pages to familiarize yourself with key SCT concepts:

    * :ref:`pam50`: An overview of the PAM50 template's features, as well as context for why the template is used.
    * :ref:`warping-fields`: Background information on the image transformation format used by the registration process.
    * :ref:`qc`: Primer for SCT's Quality Control interface. After each step of this tutorial, you will be able to open a QC report that lets you easily evaluate the results of each command.

2. Make sure that you have the following files in your working directory:

   * ``single_subject/data/t2/t2.nii.gz``: An anatomical spinal cord scan in the T1 contrast.
   * ``single_subject/data/t2/t2_seg.nii.gz``: An 3D binary mask for the spinal cord.
   * ``single_subject/data/t2/t2_labels_vert.nii.gz`` : Image containing the 2 single-voxel vertebral labels

   You can get these files by downloading :sct_tutorial_data:`data_template-registration.zip`.


3. Open a terminal and navigate to the ``single_subject/data/t2/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/t2/