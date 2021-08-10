Improving registration results using white and gray matter segmentations
########################################################################

This tutorial is a follow-on for both the :ref:`gm-wm-segmentation` tutorial and the :ref:`registering-additional-contrasts` tutorial. It demonstrates how to use the previously-acquired T2* white and gray matter segmentations to improve the registration results for MT data acquired in the same session.

.. note::

   Most of the time, the improvement of using GM registration is small. In some cases it can even make it worse (because the result will largely depend on the quality of the GM segmentation), so in general we don’t recommend going through this 2-step registration.

   .. TODO: This warning was present in the presenter notes of one of the slides, but is a little discouraging. It begs the questions: "What is the point of this tutorial? If we don't recommend this method (in general), then in what specific cases would we recommend going through this?"

.. toctree::
   :caption: Table of Contents
   :maxdepth: 1

   improving-registration-with-gm-seg/before-starting
   improving-registration-with-gm-seg/gm-informed-t2s-template-registration
   improving-registration-with-gm-seg/gm-informed-mt-template-registration
