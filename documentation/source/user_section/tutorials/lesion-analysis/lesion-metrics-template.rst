Template/Atlas-based lesion analysis
####################################

The ``sct_analyze_lesion`` function allows you to provide a folder containing an atlas/template registered to an anatomical image.
If specified, the function computes:

* a. for each lesion, the proportion of that lesion within each vertebral level and each region of the template (e.g. GM, WM, WM tracts). Each cell in the output XLS file contains a percentage value representing how much of the lesion volume exists within the region indicated by the row/column (rows represent vertebral levels, columns represent ROIs). The percentage values are summed to totals in both the bottom row and the right column, and the sum of all cells is 100 (i.e. 100 percent of the lesion), found in the bottom-right.
* b. the proportions of each ROI (e.g. vertebral level, GM, WM) occupied by lesions.

.. note::

   TODO: as we need to provide a template/atlas, which is covered by the next tutorial (see below), we can consider moving this tutorial after the template/atlas-based lesion segmentation tutorial.

   You can register the template and warp the atlas to the anatomical image using the ``sct_register_to_template`` and ``sct_warp_template`` functions, respectively.
   See :ref:`template-registration <template-registration>` for more information.

Running ``sct_analyze_lesion``
------------------------------

``sct_analyze_lesion`` can be applied to lesion and spinal cord segmentation masks using the following command

.. code:: sh

   sct_analyze_lesion -m t2_lesion_seg.nii.gz -s t2_sc_seg.nii.gz -f label_T2w -qc ~/qc_singleSubj

:Input arguments:
   - ``-m`` : 3D binary mask of the segmented lesion
   - ``-s`` : 3D binary mask of the segmented spinal cord
   - ``-f`` : Folder containing the atlas/template registered to the anatomical image
   - ``-qc`` : Directory for Quality Control reporting. QC report contains a figure for the tissue bridges