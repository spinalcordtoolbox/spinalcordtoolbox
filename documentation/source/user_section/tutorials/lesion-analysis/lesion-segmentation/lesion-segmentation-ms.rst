Lesion segmentation in multiple sclerosis: ``sct_deepseg_lesion``
#####################################################################################################

As its name suggests, ``sct_deepseg_lesion`` is based on deep learning.

TODO: consider adding dome details what data was used to train the model, similarly to the SCIseg model (lesion-segmentation-sci.rst)

The steps of the algorithm are as follows:

:1. Spinal cord detection:
   First, a convolutional neural network is used to generate a probablistic heatmap for the location of the spinal cord.

:2. Centerline detection:
   The heatmap is then fed into the `OptiC <https://archivesic.ccsd.cnrs.fr/PRIMES/hal-01713965v1>`_ algorithm to detect the spinal cord centerline.

:3. Patch extraction:
   The spinal cord centerline is used to extract a patch from the image. (This is done to exclude regions that we are certain do not contain the spinal cord.)

:4. Segmentation:
   Lastly, a second convolutional neural network is applied to the extracted patch to segment the lesion.

.. note::

   The ``sct_deepseg_lesion`` algorithm segments only the MS lesion(s). You can use ``sct_deepseg_sc`` to segment the spinal cord. TODO: probably mention also contrast-agnostic model here

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/sct_deepseg_sc_steps.png
   :align: center
   :figwidth: 60%

You can try ``sct_deepseg_lesion`` on your own T2w or T2star image using the following command:

.. code:: sh

   sct_deepseg_lesion -i t2.nii.gz -c t2

:Input arguments:
   - ``-i`` : Input T2w image
   - ``-c`` : Contrast of the input image

:Output files/folders:
   - ``t2_lesionseg.nii.gz`` : 3D binary mask of the segmented lesion

Details:

* **Algorithm:** `NeuroImage, C., et al. NeuroImage (2019) <https://doi.org/10.1016/j.neuroimage.2018.09.081>`_
