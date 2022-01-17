Using binary masks to extract intensity values for gray and white matter
########################################################################

SCT also offers another tool called ``sct_extract_metric`` that allows you to extract and aggregate voxel values from specific regions within images.

Here, we can use this tool (alongside the gray and white matter masks) to extract intensity values from the gray and white matter regions of the T2* anatomical image.

.. code::

   sct_extract_metric -i t2s.nii.gz -f t2s_wmseg.nii.gz -method bin -z 2:12 -o t2s_value.csv

.. code::

   sct_extract_metric -i t2s.nii.gz -f t2s_gmseg.nii.gz -method bin -z 2:12 -o t2s_value.csv -append 1

:Input arguments:
   - ``-i`` : Image to extract intensity values from.
   - ``-f`` : File or folder used to pick out specific regions from the input image. In this case, we provide masks to extract the image intensity for the white and gray matter specifically.
   - ``-method`` : We specify ``bin`` to binarize the mask, then use the binary mask to select the voxels for the metric extraction.
   - ``-z`` : The axial slice region to extract metrics for. By default, the intensity will be summed for each slice, then averaged across the region. You can also specify the argument ``-perslice 1`` to extract metrics for each slice individually.
   - ``-o`` : The name of the output file.
   - ``-append`` : Whether or not to append the results to the end of an existing file, rather than overwriting it.

.. csv-table:: ``t2s_value.csv``: Intensity values in GM and WM
   :file: t2s_value.csv
   :header-rows: 1

.. note:: The ``-f`` option can be used with any binary mask, and not just the white and gray matter segmentations.

For further reading, this technique for signal intensity quantification was applied to the monitoring for myelopathic progression with multiparametric quantitative MRI in `[Martin et al. PLoS One 2018] <https://doi.org/10.1371/journal.pone.0204082>`_.