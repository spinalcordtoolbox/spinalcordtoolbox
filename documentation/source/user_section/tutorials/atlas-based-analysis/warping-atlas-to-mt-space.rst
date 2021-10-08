Transforming the GM/WM atlas to the MT space using warping fields
#################################################################

For the next example, we will demonstrate how to extract MTR for specific white matter tracts. Since we are now working with MT data, we will switch folders using ``cd``:

.. code:: sh

   cd ../mt

In order to extract and aggregate metrics from specific white matter tracts, the white/gray matter atlas must first be transformed from the unbiased PAM50 coordinate space to the space of the data you wish to compute metrics for. To do this, we will re-use the ``warp_template2mt.nii.gz`` file generated in previous MT tutorials:

* :ref:`registering-additional-contrasts`
* :ref:`gm-informed-mt-registration`

The command below assumes that the warping field is already present in your working directory.

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