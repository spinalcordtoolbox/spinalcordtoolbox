.. _spinalcord-segmentation:

Spinal cord segmentation for anatomical images
##############################################

This tutorial demonstrates how to use SCT's command-line scripts to perform spinal cord segmentation. Segmentation in this case means to generate a 3D mask that identifies the spinal cord within anatomical images of the spine. This tutorial compares two different algorithms provided by SCT, and is meant to give you a feel for common usage of these tools on real-world data.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/spinalcord-segmentation.png
   :align: center
   :height: 300px

.. toctree::
   :caption: Table of Contents
   :maxdepth: 1

   spinalcord-segmentation/before-starting
   spinalcord-segmentation/contrasts
   spinalcord-segmentation/sct_propseg
   spinalcord-segmentation/sct_propseg-example-t2
   spinalcord-segmentation/sct_propseg-example-t1
   spinalcord-segmentation/fixing-failed-propseg-segmentations
   spinalcord-segmentation/sct_deepseg_sc
   spinalcord-segmentation/sct_deepseg_sc-example-t1
   spinalcord-segmentation/choosing-an-algorithm
