.. _template-registration:

Registering anatomical images to the PAM50 template
###################################################

This tutorial demonstrates a multi-step pipeline to `register <http://jpeelle.net/mri/image_processing/registration.html>`_ an anatomical MRI scan to the PAM50 Template. While T2 images are used for this tutorial, each step is applicable to multiple contrast types (T1, T2, T2*).

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/registration-pipeline.png
   :align: center
   :height: 300px

.. toctree::
   :caption: Table of Contents
   :maxdepth: 1

   template-registration/before-starting
   template-registration/sct_register_to_template
   template-registration/applying-the-registration-algorithm
   template-registration/customizing-the-algorithm
   template-registration/applying-the-warping-fields