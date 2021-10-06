Extracting DTI from specific spinal cord regions
################################################

Preparing the PAM50 template for metric extraction
--------------------------------------------------

Now that we've registered the PAM50 template with the motion-corrected dMRI data, we can use the resulting warping field to transform the full template to the space of the dMRI data. This will allow us to use the PAM50 template and atlas to extract metrics from specific regions of the image.

.. code::

   sct_warp_template -d dmri_moco_dwi_mean.nii.gz -w warp_template2dmri.nii.gz -qc ~/qc_singleSubj

:Input arguments:
   - ``-d`` : Destination image the template will be warped to.
   - ``-w`` : Warping field (template space to anatomical space).
   - ``-a`` : Because ``-a 1`` is specified, the white and gray matter atlas will also be warped.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``label/template/`` : This directory contains the entirety of the PAM50 template, transformed into the DT space.
   - ``label/atlas/`` : This direct contains 36 NIFTI volumes for WM/GM tracts, transformed into the DT space.

Extracting DTI values from white matter across individual verebral levels
-------------------------------------------------------------------------

In this example, we will extract and aggregate values from the white matter region of the fractional anisotropy (FA) diffusion tensor image.

.. code:: sh

   sct_extract_metric -i dti_FA.nii.gz -f label/atlas \
                      -l 51 -method map \
                      -vert 2:5 -vertfile label/template/PAM50_levels.nii.gz -perlevel 1 \
                      -o fa_in_wm.csv

:Input arguments:
   - ``-i`` : Image to extract intensity values from.
   - ``-f`` : File or folder used to pick out specific regions from the input image. In this case, we supply the folder of the white matter atlas that was transformed to the space of the DTI data.
   - ``-l`` : The IDs of the label (or group of labels) to compute metrics for. In this case, label 51 is a combined label that represents all of the WM tracts together. You can see the full list of labels to choose from by running ``sct_extract_metric -list-labels``.
   - ``-method`` : We specify ``map`` to choose the Maximum a Posteriori method, which helps to account for the partial volume effect. This method is most suitable for small regions. This method should only be used with the PAM50 white/gray matter atlas (or with any custom atlas) as long as the sum across all overlapping labels equals 1 in each voxel part of the atlas.
   - ``-vert``: This argument further restricts the metric extraction to specific vertebral levels. In this case, 4 levels in total (C2-C5) will be used.
   - ``-vertfile``: This argument specifies the vertebral label file to use when extracting metrics.
   - ``-perlevel``: Providing this argument will extract the metrics for each vertebral level, rather than averaging across the entire region.
   - ``-o`` : The name of the output file.

.. csv-table:: ``fa_in_wm.csv``: FA values in white matter across vertebral levels
   :file: fa_in_wm.csv
   :header-rows: 1