.. _template-registration:

Registering labeled anatomical images to the PAM50 template
###########################################################

This tutorial demonstrates how to `register <http://jpeelle.net/mri/image_processing/registration.html>`_ a segmented, labeled anatomical MRI scan to the PAM50 Template. While T2 images are used for this tutorial, each step is applicable to multiple contrast types (T1, T2, T2*).

.. toctree::
   :maxdepth: 1

   template-registration/before-starting
   template-registration/sct_register_to_template
   template-registration/applying-the-registration-algorithm
   template-registration/customizing-registration
   template-registration/applying-the-warping-fields

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/template-registration/registration-pipeline.png
   :align: center