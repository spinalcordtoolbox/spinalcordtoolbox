Before starting this tutorial
#############################

1. Because this is a follow-on tutorial for :ref:`registering-additional-contrasts`, that tutorial must be completed beforehand, as several files are reused here.

 * ``mt1_seg.nii.gz`` : The segmented spinal cord for the MT1 image (used for registering MT0 on MT1).
 * ``mask_mt1.nii.gz`` : The mask surrounding the spinal cord region of interest (used for registering MT0 on MT1).
 * ``label/template`` : The warped PAM50 template objects (used to compute MTR for specific regions).

2. Open a terminal and navigate to the ``sct_course_london20/single_subject/data/mt/`` directory.