.. _before-starting-lumbar-registration:

Before starting this tutorial
#############################

#. Make sure that you are working with a copy of SCT 6.1 or higher, and have an up-to-date copy of the PAM50 template:

   .. code:: sh

      sct_download_data -d PAM50

   For older versions of SCT, you can download the `latest release <https://github.com/spinalcordtoolbox/PAM50/releases>`_ of the PAM50 template manually, then copy the files into the ``$SCT_DIR/data/PAM50`` folder. (We recommend making a backup of the PAM50 folder if you are performing ongoing work with a stable release of SCT, that way you can go back to the older copy when resuming your work.)

#. Make sure that you have the following files in your working directory:

   * ``single_subject/data/t2_lumbar/t2.nii.gz`` : T2w anatomical scan of the lumbar spinal cord.

   You can get this file by downloading :sct_tutorial_data:`data_lumbar-registration.zip`.

#. Open a terminal and navigate to the ``single_subject/data/t2_lumbar/`` directory:

   .. code:: sh

      cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/t2_lumbar/
