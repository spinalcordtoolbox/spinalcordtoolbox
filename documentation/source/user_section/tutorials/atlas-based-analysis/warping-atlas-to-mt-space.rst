Transforming the GM/WM atlas to the MT space using warping fields
#################################################################

Now that the necessary theory has been covered, we can demonstrate how to use the PAM50 atlas to extract MTR for specific white matter tracts.

First, the white/gray matter atlas must be transformed from the unbiased PAM50 coordinate space to the space of the MT data we wish to compute metrics for. To do this, we apply a warping field (``warp_template2mt.nii.gz``) to the template.

.. code:: sh

   sct_warp_template -d mt1.nii.gz -w warp_template2mt.nii.gz -a 1 -qc ~/qc_singleSubj

:Input arguments:
   - ``-d`` : Destination image the template will be warped to.
   - ``-w`` : Warping field (template space to anatomical space).
   - ``-a`` : Because ``-a 1`` is specified, the white and gray matter atlas will also be warped.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``label/template/`` : This directory contains the entirety of the PAM50 template, transformed into the MT space.
   - ``label/atlas/`` : This direct contains 36 NIFTI volumes for WM/GM tracts, transformed into the MT space.

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using either :ref:`Quality Control (QC) <qc>` reports or :ref:`fsleyes-instructions`.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registering-additional-contrasts/io-sct_warp_template.png
   :align: center

   Input/output images for ``sct_warp_template``

Now that the atlas has been warped to the MT space, it can be used to extract MTR for specific white matter regions.