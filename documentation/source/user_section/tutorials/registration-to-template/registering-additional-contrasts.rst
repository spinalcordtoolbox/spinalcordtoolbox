.. _registering-additional-contrasts:

Coregistering additional data (MT, DT) to the PAM50 template
############################################################

This tutorial demonstrates how to register an additional contrasts to the PAM50 template, alongside other data files that were acquired in the same session. Specifically, we will be registering MT data to the PAM50 template alongside the T2 data that was registered in the :ref:`previous tutorial <template-registration>`.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registering-additional-contrasts/mt-registration-pipeline.png
   :align: center

This tutorial is designed for `magnetization transfer <https://radiopaedia.org/articles/magnetisation-transfer-1>`_ (MT) images or `diffusion tensor <https://radiopaedia.org/articles/diffusion-tensor-imaging-and-fibre-tractography?lang=us>`_ (DT) images. However, these steps are also applicable to any images that are similar in appearance to T1, T2, or T2* contrasts.

.. toctree::
   :maxdepth: 1

   registering-additional-contrasts/before-starting
   registering-additional-contrasts/spinalcord-segmentation
   registering-additional-contrasts/creating-a-mask
   registering-additional-contrasts/registration-1-reusing-warping-fields
   registering-additional-contrasts/registration-2-direct-registration
   registering-additional-contrasts/applying-the-warping-fields