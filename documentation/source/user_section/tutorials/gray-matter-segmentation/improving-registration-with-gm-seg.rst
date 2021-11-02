.. _improving-registration-with-gm-seg:

Improving registration results using white and gray matter segmentations
########################################################################

This tutorial demonstrates how to use the previously-acquired T2* :ref:`white and gray matter segmentations <gm-wm-segmentation>` to improve the registration results for MT data acquired in the same session as the T2* data.

.. toctree::
   :maxdepth: 1

   improving-registration-with-gm-seg/before-starting
   improving-registration-with-gm-seg/gm-informed-t2s-template-registration
   improving-registration-with-gm-seg/gm-informed-mt-template-registration

.. note::

   Most of the time, the improvement of using GM registration is small, and in some cases can make the registration results worse (because the result will largely depend on the quality of the GM segmentation). So, we recommend that you start with the :ref:`standard template registration technique <registration-to-template>` first before adding in white and gray matter segmentations.