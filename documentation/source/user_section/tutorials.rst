.. _tutorials:

Tutorials
#########

.. note::

   The following tutorials make use of :ref:`Quality Control (QC) <qc>` reports for inspecting processed image data. Optionally, we also recommend trying out :ref:`fsleyes-instructions` to visualize the data from these tutorials, too.

Written tutorials
*****************

We provide the following hands-on tutorials for SCT's command-line tools.

#. :doc:`Spinal cord segmentation <tutorials/segmentation>`
#. :doc:`Vertebral labeling <tutorials/vertebral-labeling>`
#. :doc:`Shape analysis <tutorials/shape-analysis>`

   * :doc:`tutorials/shape-analysis/compute-csa-and-other-shape-metrics`
   * :doc:`tutorials/shape-analysis/normalize-morphometrics-compression`

#. :doc:`Lesion analysis <tutorials/lesion-analysis>`
#. :doc:`Spinal nerve rootlets segmentation <tutorials/spinal-nerve-rootlets-segmentation>`
#. :doc:`Registration to template <tutorials/registration-to-template>`

   * :doc:`tutorials/registration-to-template/template-registration`
   * :doc:`tutorials/registration-to-template/rootlets-based-registration`
   * :doc:`tutorials/registration-to-template/registering-additional-contrasts`
   * :doc:`tutorials/registration-to-template/lumbar-registration`

#. :doc:`Multimodal registration <tutorials/multimodal-registration>`

   * :doc:`tutorials/multimodal-registration/mtr-computation`
   * :doc:`tutorials/multimodal-registration/contrast-agnostic-registration`

#. :doc:`Gray matter segmentation <tutorials/gray-matter-segmentation>`

   * :doc:`tutorials/gray-matter-segmentation/gm-wm-segmentation`
   * :doc:`tutorials/gray-matter-segmentation/gm-wm-metric-computation`
   * :doc:`tutorials/gray-matter-segmentation/improving-registration-with-gm-seg`

#. :doc:`Atlas-based analysis <tutorials/atlas-based-analysis>`
#. :doc:`Diffusion-weighted MRI (Motion correction, DTI computation) <tutorials/diffusion-weighted-mri>`
#. :doc:`Functional MRI (Motion correction, Spinal level labeling) <tutorials/processing-fmri-data>`
#. :doc:`Other features <tutorials/other-features>`

   * :doc:`Spinal cord smoothing <tutorials/other-features/spinalcord-smoothing>`
   * :doc:`Visualizing misaligned cords <tutorials/other-features/visualizing-misaligned-cords>`

#. :doc:`Processing batches of subjects using pipeline scripts <tutorials/analysis-pipelines-with-sct>`

Video tutorials
***************

SCT have a `YouTube channel`_ which contains additional tutorials.

.. _Youtube channel: https://www.youtube.com/playlist?list=PLJ5-Fnq9XpaVgCZfY-GOGJaT0fmZN4vji

SCT Course
**********

If you would prefer to learn how to use SCT in a guided workshop setting, we provide an in-person SCT course each year. You can learn more about past and future courses in the :ref:`course section<courses>`.


.. Note: The toctree below is required by Sphinx for the sidebar. However, the automatically generated sidebar isn't ideal, because ":maxdepth: 2" shows too many sections, but ":maxdepth: 1" doesn't show enough. To get around this, we set the toctree as `:hidden:`, then manually create a secondary TOC using bullet point lists (see above). This manual method produces e a good-looking hybrid of both of the 'max-depth' options.

.. Note 2: Both the hidden toctree (below) and the manual TOC (above) should be updated together. Make sure to use short titles in each section's page (since these will automatically be shown in the sidebar). But, feel free to use longer titles in the manual TOC, where there is more space.

.. toctree::
   :hidden:
   :maxdepth: 2

   tutorials/segmentation
   tutorials/vertebral-labeling
   tutorials/shape-analysis
   tutorials/lesion-analysis
   tutorials/spinal-nerve-rootlets-segmentation
   tutorials/registration-to-template
   tutorials/multimodal-registration
   tutorials/gray-matter-segmentation
   tutorials/atlas-based-analysis
   tutorials/diffusion-weighted-mri
   tutorials/processing-fmri-data
   tutorials/other-features
   tutorials/analysis-pipelines-with-sct
