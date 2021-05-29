Hands-on Example: T1
####################

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/t1_deepseg_before_after.png
   :align: right
   :figwidth: 20%

   No leakage with ``sct_deepseg_sc``

Since we aim to improve the T1 segmentation, ensure that you are still in the T1 directory (``sct_course_london20/single_subject/data/t1``). Once there, run this command:

.. code:: sh

   sct_deepseg_sc -i t1.nii.gz -c t1 -qc ~/qc_singleSubj -ofolder deepseg

:Input arguments:
   - ``-i`` : Input image
   - ``-c`` : Contrast of the input image
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the segmentation slice-by-slice
   -  ``-ofolder`` : The folder to output files to. We specify this here so that we don't overwrite the ``t2_seg.nii.gz`` file output by ``sct_propseg``.

:Output files/folders:
   - ``t2_seg.nii.gz`` : 3D binary mask of the segmented spinal cord

Much like ``sct_propseg``, we use the same values for ``-i``, ``-c``, and ``-qc``. In this case, however, we have added an additional ``-ofolder`` command. This is so that we do not overwrite the results generated in the previous steps, which allows us to compare the output of both algorithms. ``-ofolder`` is not strictly necessary, however.

Inspecting the T1w results using QC
***********************************

Once again, you may either execute the command given by the script, or simply refresh the QC webpage from the previous examples.

In this case, ``sct_deepseg_sc`` has managed to improve upon the results of ``sct_propseg``.