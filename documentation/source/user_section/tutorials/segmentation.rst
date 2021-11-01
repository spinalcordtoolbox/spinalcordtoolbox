.. _spinalcord-segmentation:

Spinal cord segmentation for anatomical images
##############################################

This tutorial demonstrates how to use SCT's command-line scripts to perform spinal cord segmentation. In image processing, `segmentation <https://en.wikipedia.org/wiki/Image_segmentation>`_ is the process of partitioning an image into different segments (sets of pixels/voxels). In the context of SCT, we generate a 3D mask (solid region overlaying segment of interest) that identifies the spinal cord within anatomical images of the spine. This tutorial compares two different algorithms provided by SCT, and is meant to give you a feel for common usage of these tools on real-world data.

.. toctree::
   :maxdepth: 1

   segmentation/before-starting
   segmentation/contrasts
   segmentation/sct_propseg
   segmentation/sct_propseg-example-t2
   segmentation/sct_propseg-example-t1
   segmentation/fixing-failed-sct_propseg-segmentations
   segmentation/sct_deepseg_sc
   segmentation/sct_deepseg_sc-example-t1
   segmentation/choosing-an-algorithm

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/spinalcord-segmentation.png
   :align: center
   :height: 300px
