.. _template-registration:

Registering labeled anatomical images to the PAM50 template
###########################################################

This tutorial demonstrates how to `register <http://jpeelle.net/mri/image_processing/registration.html>`__ a segmented, labeled anatomical MRI scan to the PAM50 Template. While T2 images are used for this tutorial, each step is applicable to multiple contrast types (T1, T2, T2*).

.. toctree::
   :maxdepth: 1

   rootlets-based-registration/before-starting
   rootlets-based-registration/sct_register_to_template
   rootlets-based-registration/applying-the-registration-algorithm
   rootlets-based-registration/customizing-registration
   rootlets-based-registration/applying-the-warping-fields

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/template-registration/registration-pipeline.png
   :align: center

..
   comment:: In the SCT Course slides and ``batch_single_subject.sh`` script,
             we include some additional registration commands to demonstrate
             other potential uses of ``sct_register_to_template`` (``-param``,
             ``-ldisc``, compressed cord). I think we are missing these
             commands from this tutorial. We should consider adding them here.
             There is a "Customizing Registration" page, however. But, if we
             follow what was proposed in issue #3095, those tips could be added
             to ``sct_register_to_template``'s dedicated page, freeing up space
             in the tutorial for an "other sample commands" page that then
             links to ``sct_register_to_template``.