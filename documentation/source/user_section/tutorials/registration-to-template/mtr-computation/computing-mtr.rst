Computing MTR using coregistered MT data
########################################

Now that we have aligned the MT0 and MT1 images using co-registration, we can compute the magnetization transfer ratio (MTR) for each voxel.

.. code:: sh

   sct_compute_mtr -mt0 mt0_reg.nii.gz -mt1 mt1.nii.gz

:Input arguments:
   - ``-mt0`` : The input image without the magnetization transfer pulse.
   - ``-mt1`` : The input image with the magnetization transfer pulse.

:Output files/folders:
   - ``mtr.nii.gz`` : An image containing the voxel-wise magnetization transfer ratio.

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using :ref:`fsleyes-instructions`.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/mtr-computation/io-sct_compute_mtr.png
   :align: center

   Input/output images for ``sct_compute_mtr``