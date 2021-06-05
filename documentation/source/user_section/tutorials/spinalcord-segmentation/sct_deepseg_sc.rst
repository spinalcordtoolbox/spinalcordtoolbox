Algorithm #2: ``sct_deepsec_sc``
################################

As its name suggests, ``sct_deepseg_sc`` is based on deep learning. It is a newer algorithm, having been introduced to SCT in 2018. The steps of the algorithm are as follows:

#. A convolutional neural network is used to generate a probablistic heatmap for the location of the spinal cord.
#. The heatmap is fed into the `OptiC <https://archivesic.ccsd.cnrs.fr/PRIMES/hal-01713965v1>`_ algorithm to detect the spinal cord centerline.
#. The spinal cord centerline is used to extract a patch from the image.

   - We extract a patch to help combat class imbalance. If the full image were to be used instead, the spinal cord region would be small in proportion to the non-spinal cord regions of the image, and thus harder to detect.

#. Lastly, a second convolutional neural network is applied to the extracted patch to segment the spinal cord.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/sct_deepseg_sc_steps.png
   :align: center
   :figwidth: 65%

   The steps for ``sct_deepseg_sc``