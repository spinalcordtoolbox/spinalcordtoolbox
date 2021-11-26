.. _transforming-template-section:

Transforming the template using warping fields
##############################################

Once the transformations are estimated, we can apply the resulting warping field to the template to bring it into to the subject’s native space.

.. code:: sh

   sct_warp_template -d t2.nii.gz -w warp_template2anat.nii.gz -a 0 -qc ~/qc_singleSubj

:Input arguments:
   - ``-d`` : Destination image the template will be warped to.
   - ``-w`` : Warping field (template space to anatomical space).
   - ``-a`` : Whether or not to also warp the white matter atlas. (If ``-a 1`` is specified, ``label/atlas/`` will also be warped.)
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``label/template/`` : This directory contains the 15 PAM50 template objects that have been transformed into the subject space (i.e. the t2.nii.gz anatomical image). These files can be used to compute metrics for different regions of the spinal cord. For further details on the template itself, visit the :ref:`pam50` page.

Once the command has finished, at the bottom of your terminal there will be instructions for inspecting the results using :ref:`Quality Control (QC) <qc>` reports. Optionally, If you have :ref:`fsleyes-instructions` installed, a ``fsleyes`` command will printed as well.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/template-registration/io-sct_warp_template.png
   :align: center

   Input/output images for ``sct_warp_template``