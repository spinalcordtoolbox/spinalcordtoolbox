.. _rootlets-based-registration:

Registering images to the PAM50 template based on spinal nerve rootlets
#######################################################################

This tutorial demonstrates how to `register <http://jpeelle.net/mri/image_processing/registration.html>`_ an anatomical MRI scan to the PAM50 template based on the spinal nerve rootlets instead of the vertebral levels. T2w images are used for this tutorial, as the automatic rootlets segmentation is only available for T2w and MP2RAGE.

.. toctree::
   :maxdepth: 1

   rootlets-based-registration/before-starting
   rootlets-based-registration/sct_register_to_template
   rootlets-based-registration/applying-the-registration-algorithm