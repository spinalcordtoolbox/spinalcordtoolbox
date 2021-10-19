.. _gm-wm-segmentation:

Segmenting the gray and white matter for T2* data
#################################################

This tutorial demonstrates how to segment the gray and white matter using the ``sct_deepseg_gm`` tool.

Gray matter segmentation can be used to compute metrics such as cross-sectional area (CSA) of the gray matter. It can also be used to refine the template registration by accounting for the spinal cord gray matter shape.


.. toctree::
   :maxdepth: 1

   gm-wm-segmentation/before-starting
   gm-wm-segmentation/sct_deepseg_gm
   gm-wm-segmentation/applying-the-gm-segmentation-algorithm
   gm-wm-segmentation/computing-the-wm-segmentation