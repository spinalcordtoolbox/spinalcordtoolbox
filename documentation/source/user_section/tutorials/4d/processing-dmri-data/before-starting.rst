Before starting this tutorial
#############################

1. Read through the following page to familiarize yourself with key SCT concepts:

   * :ref:`inspecting-your-results`: After some steps in this tutorial, instructions are provided to open the output images using :ref:`Quality Control (QC) <qc>` reports and :ref:`fsleyes-instructions`.

2. Make sure that you have the following files in your working directory:

   * ``single_subject/data/dmri/dmri.nii.gz`` : A 4D dMRI image comprised of 35 3D volumes.
   * ``single_subject/data/dmri/bvals.txt`` : A text file containing a b-value for each volume in the dMRI image, indicating the diffusion weightings for each of the volumes in the dMRI image.
   * ``single_subject/data/dmri/bvecs.txt`` : A text file with three lines, each containing a value for each volume in the dMRI image. Together, the the three sets of values represent the ``(x, y, z)`` coordinates of the b-vectors, which indicate the direction of the diffusion encoding for each volume of the dMRI image.
   * ``single_subject/data/t2s/warp_template2t2s.nii.gz`` : A "template->data" warping field from a previous registration. We will use this to initialize the dMRI registration.
   * ``single_subject/data/t2s/warp_t2s2template.nii.gz`` : A "data->template" warping field from a previous registration. We will use this to initialize the dMRI registration.

   You can get these files by downloading :sct_tutorial_data:`data_processing-dmri-data.zip`.

.. note:: If you are :ref:`completing all of SCT's tutorials in sequence <completing-the-tutorials-in-sequence>`, your working directory should already contain the files needed for this tutorial.

3. Open a terminal and navigate to the ``single_subject/data/dmri/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/dmri/