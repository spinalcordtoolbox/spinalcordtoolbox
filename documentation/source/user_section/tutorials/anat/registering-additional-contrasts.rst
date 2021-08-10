.. _registering-additional-contrasts:

Coregistering additional data (MT, DT) to the PAM50 template
############################################################

This tutorial demonstrates how to register images of other types, such as `magnetization transfer <https://radiopaedia.org/articles/magnetisation-transfer-1>`_ (MT) images and `diffusion tensor <https://radiopaedia.org/articles/diffusion-tensor-imaging-and-fibre-tractography?lang=us>`_ (DT) images. A common use case is to coregister multiple contrasts acquired in the same session. In this tutorial, we will work with MT data, but these steps should apply to any images that are similar in appearance to T1, T2, or T2* contrasts.


.. toctree::
   :caption: Table of Contents
   :maxdepth: 1

   registering-additional-contrasts/before-starting
   registering-additional-contrasts/spinalcord-segmentation
   registering-additional-contrasts/creating-a-mask
   registering-additional-contrasts/registration-1-reusing-warping-fields
   registering-additional-contrasts/registration-2-direct-registration
   registering-additional-contrasts/applying-the-warping-fields