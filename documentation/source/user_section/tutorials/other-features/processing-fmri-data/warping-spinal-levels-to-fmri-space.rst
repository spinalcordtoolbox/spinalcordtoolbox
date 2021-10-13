Warping the spinal levels to the fMRI space
###########################################

Now that we've registered the PAM50 template with the motion-corrected fMRI data, we can use the resulting warping field to transform the full template to the space of the dMRI data.

.. code::

   sct_warp_template -d fmri_moco_mean.nii.gz -w warp_template2fmri.nii.gz -s 1 -a 0 -qc ~/qc_singleSubj

:Input arguments:
   - ``-d`` : Destination image the template will be warped to.
   - ``-w`` : Warping field (template space to anatomical space).
   - ``-s`` : Because ``-s 1`` is specified, spinal levels will be warped.
   - ``-a`` : Because ``-a 0`` is specified, the white and gray matter atlas will not be warped.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``label/template/`` : This directory contains the entirety of the PAM50 template, transformed into the fMRI space.
   - ``label/spinal_levels/`` : This direct contains 20 label images corresponding to different spinal cord levels, spanning both C1:C8 and T1:T12, transformed into the fMRI space. In each NIfTI file, the value of each voxel is the probability for this voxel to belong to the spinal level.

Visualizing the spinal levels
-----------------------------

If you have ``fsleyes`` installed, you can visualize the warped spinal levels together using the following syntax:

.. code::

   fsleyes --scene lightbox --hideCursor fmri_moco_mean.nii.gz -cm greyscale -dr 0 1000 \
                                         label/spinal_levels/spinal_level_03 -cm red \
                                         label/spinal_levels/spinal_level_04 -cm blue \
                                         label/spinal_levels/spinal_level_05 -cm green \
                                         label/spinal_levels/spinal_level_06 -cm yellow

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/processing-fmri-data/spinal-levels.png
   :align: center