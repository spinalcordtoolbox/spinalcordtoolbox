.. _rootlets-based-registration:

Registering images to the PAM50 template based on spinal nerve rootlets
#######################################################################

This tutorial demonstrates how to `register <http://jpeelle.net/mri/image_processing/registration.html>`_ an anatomical MRI scan to the PAM50 template based on the spinal nerve rootlets instead of the vertebral levels. While T2w images are used for this tutorial, the registration is applicable also to other contrasts; see `sct_deepseg -rootlets` for the list of supported contrasts for automatic rootlet segmentation.

.. toctree::
   :maxdepth: 1

   rootlets-based-registration/before-starting
   rootlets-based-registration/rootlets-segmentation
   rootlets-based-registration/sct_register_to_template
   rootlets-based-registration/applying-the-registration-algorithm