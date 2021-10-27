Algorithm #2: ``sct_deepseg_sc``
################################

As its name suggests, ``sct_deepseg_sc`` is based on deep learning. It is a newer algorithm, having been introduced to SCT in 2018. The steps of the algorithm are as follows:

:1. Spinal cord detection:
   First, a convolutional neural network is used to generate a probablistic heatmap for the location of the spinal cord.

:2. Centerline detection:
   The heatmap is then fed into the `OptiC <https://archivesic.ccsd.cnrs.fr/PRIMES/hal-01713965v1>`_ algorithm to detect the spinal cord centerline.

:3. Patch extraction:
   The spinal cord centerline is used to extract a patch from the image. (This is done to exclude regions that we are certain do not contain the spinal cord.)

:4. Segmentation:
   Lastly, a second convolutional neural network is applied to the extracted patch to segment the spinal cord.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/sct_deepseg_sc_steps.png
   :align: center
   :figwidth: 65%