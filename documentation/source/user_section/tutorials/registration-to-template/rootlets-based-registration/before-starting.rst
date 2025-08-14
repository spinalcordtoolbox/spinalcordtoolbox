.. _before-starting-rootlets-based-registration:

Before starting this tutorial
#############################

This rootlets-based registration tutorial is separate from the previous registration tutorials, meaning it can be run standalone.

#. Make sure that you are working with a copy of SCT 7.0 or higher, and have an up-to-date copy of the PAM50 template:

   .. code:: sh

      sct_download_data -d PAM50

   For older versions of SCT, you can download the `latest release <https://github.com/spinalcordtoolbox/PAM50/releases>`_ of the PAM50 template manually, then copy the files into the ``$SCT_DIR/data/PAM50`` folder. (We recommend making a backup of the PAM50 folder if you are performing ongoing work with a stable release of SCT, that way you can go back to the older copy when resuming your work.)

#. Make sure that you have the following files in your working directory:

   * ``single_subject/data/t2/t2.nii.gz`` : T2w anatomical scan of the spinal cord.
   * ``single_subject/data/t2/t2_seg.nii.gz`` : Segmentation of the spinal cord in the T2w image.
   * ``single_subject/data/t2/t2_rootlets.nii.gz`` : Segmentation of the spinal nerve rootlets in the T2w image.

   You can get this file by downloading :sct_tutorial_data:`data_lumbar-registration.zip`.

#. Open a terminal and navigate to the ``single_subject/data/t2/`` directory:

   .. code:: sh

      cd {PATH_TO_DOWNLOADED_DATA}/single_subject/data/t2/
