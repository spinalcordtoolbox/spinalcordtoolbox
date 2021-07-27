Before starting this tutorial
#############################

1. Read through the following page to familiarize yourself with key SCT concepts:

    * :ref:`inspecting-your-results`: After some steps in this tutorial, instructions are provided to open the output images using :ref:`Quality Control (QC) <qc>` reports and :ref:`fsleyes-instructions`.

2. Make sure that you have the following files in your working directory:

 * ``single_subject/data/mt/mt0.nii.gz``: The image without the magnetization transfer pulse.
 * ``single_subject/data/mt/mt1.nii.gz``: The image with the magnetization transfer pulse.

.. TODO: Replace these file requirements with seg/mask generation steps

 * ``single_subject/data/mt/mt1_seg.nii.gz`` : The segmented spinal cord for the MT1 image (used for registering MT0 on MT1).
 * ``single_subject/data/mt/mask_mt1.nii.gz`` : The mask surrounding the spinal cord region of interest (used for registering MT0 on MT1).

   You can get these files by downloading :sct_tutorial_data:`data_mtr-computation.zip`.

.. note:: If you are :ref:`completing all of SCT's tutorials in sequence <completing-the-tutorials-in-sequence>`, your working directory should already contain the files needed for this tutorial.

3. Open a terminal and navigate to the ``single_subject/data/mt/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/mt/