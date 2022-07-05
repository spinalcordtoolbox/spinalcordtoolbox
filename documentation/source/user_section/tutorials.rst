.. _tutorials:

Tutorials
#########

.. note::

   The following tutorials make use of :ref:`Quality Control (QC) <qc>` reports for inspecting processed image data. Optionally, we also recommend trying out :ref:`fsleyes-instructions` to visualize the data from these tutorials, too.

Written tutorials
*****************

We provide the following hands-on tutorials for SCT's command-line tools.

#. :doc:`Segmentation <tutorials/segmentation>`
#. :doc:`Registration to template <tutorials/registration-to-template>`

   * :doc:`tutorials/registration-to-template/vertebral-labeling`
   * :doc:`tutorials/registration-to-template/template-registration`
   * :doc:`tutorials/registration-to-template/shape-metric-computation`
   * :doc:`tutorials/registration-to-template/registering-additional-contrasts`

#. :doc:`Non-template registration <tutorials/non-template-registration>`

   * :doc:`tutorials/non-template-registration/mtr-computation`

#. :doc:`Gray matter segmentation <tutorials/gray-matter-segmentation>`

   * :doc:`tutorials/gray-matter-segmentation/gm-wm-segmentation`
   * :doc:`tutorials/gray-matter-segmentation/gm-wm-metric-computation`
   * :doc:`tutorials/gray-matter-segmentation/improving-registration-with-gm-seg`

#. :doc:`Atlas-based analysis <tutorials/atlas-based-analysis>`
#. :doc:`Diffusion-weighted MRI (Motion correction, DTI computation) <tutorials/diffusion-weighted-mri>`
#. :doc:`Other features <tutorials/other-features>`

   * :doc:`Functional MRI (Motion correction, Spinal level labeling) <tutorials/other-features/processing-fmri-data>`
   * :doc:`Spinal cord smoothing <tutorials/other-features/spinalcord-smoothing>`
   * :doc:`Visualizing misaligned cords <tutorials/other-features/visualizing-misaligned-cords>`

#. :doc:`Analysis pipelines with SCT <tutorials/analysis-pipelines-with-sct>`

Video tutorials
***************

SCT have a `YouTube channel`_ which contains additional tutorials.

.. _Youtube channel: https://www.youtube.com/playlist?list=PLJ5-Fnq9XpaVgCZfY-GOGJaT0fmZN4vji

SCT Course
**********

If you would prefer to learn how to use SCT in a guided workshop setting, we provide an in-person SCT course each year. You can learn more about past and future courses in the :ref:`course section<courses>`.


.. Note: The TOC below is hidden because neither ":maxdepth: 2" nor ":maxdepth: 1" looks correct. Instead, we manually
         create our own table of contents using lists, and this will produce a good-looking hybrid of both options.

.. toctree::
   :hidden:
   :maxdepth: 2

   Segmentation <tutorials/segmentation>
   tutorials/registration-to-template
   tutorials/non-template-registration
   tutorials/gray-matter-segmentation
   tutorials/atlas-based-analysis
   tutorials/diffusion-weighted-mri
   tutorials/other-features
   Analysis pipelines with SCT <tutorials/analysis-pipelines-with-sct>
