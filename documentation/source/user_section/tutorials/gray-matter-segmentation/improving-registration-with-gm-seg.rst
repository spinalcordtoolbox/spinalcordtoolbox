.. _improving-registration-with-gm-seg:

Improving registration results using white and gray matter segmentations
########################################################################

This tutorial is a follow-on for both the :ref:`gm-wm-segmentation` tutorial and the :ref:`registering-additional-contrasts` tutorial. It demonstrates how to use the previously-acquired T2* white and gray matter segmentations to improve the registration results for MT data acquired in the same session.

.. note::

   Most of the time, the improvement of using GM registration is small. In some cases it can even make it worse (because the result will largely depend on the quality of the GM segmentation). So, we recommend that you start with the standard template registration technique first.

.. toctree::
   :caption: Table of Contents
   :maxdepth: 1

   improving-registration-with-gm-seg/before-starting
   improving-registration-with-gm-seg/gm-informed-t2s-template-registration
   improving-registration-with-gm-seg/gm-informed-mt-template-registration
