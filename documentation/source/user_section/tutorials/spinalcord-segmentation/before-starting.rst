Before starting this tutorial
#############################

1. Read through the following page to familiarize yourself with key SCT concepts:

    * :ref:`inspecting-your-results`: After some steps in this tutorial, instructions are provided to open the output images using :ref:`Quality Control (QC) <qc>` reports and :ref:`fsleyes-instructions`.

2. Make sure that you have the following files in your working directory:

   * ``single_subject/data/t1/t1.nii.gz``: An anatomical spinal cord scan in the T1 contrast.
   * ``single_subject/data/t2/t2.nii.gz``: An anatomical spinal cord scan in the T2 contrast.

   You can get these files by downloading :sct_tutorial_data:`data_spinalcord-segmentation.zip`.

3. Open a terminal and navigate to the ``single_subject/data/t2/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/t2/