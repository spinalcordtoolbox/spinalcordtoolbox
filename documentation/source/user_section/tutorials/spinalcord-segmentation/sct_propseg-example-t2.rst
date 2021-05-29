Hands-on Example: T2
####################

Run the following command to process the image:

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/t2_qc.png
  :align: right
  :figwidth: 40%

  The QC report for the segmented image

.. code:: sh

   sct_propseg -i t2.nii.gz -c t2 -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Input image
   - ``-c`` : Contrast of the input image
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the segmentation slice-by-slice

:Output files/folders:
   - ``t2_seg.nii.gz`` : 3D binary mask of the segmented spinal cord

During execution, the script will provide status updates as it progress through its various stages.


Inspecting the results using QC Reports
***************************************

When complete, the script will output a command to inspect the results. (**Note:** The exact filepath will vary depending on your filesystem.)

.. code:: sh

   Use the following command to see the results in a browser:
   xdg-open "sct_course_london20/single_subject/data/t2/qc_singleSubj/index.html"

Running this command in your Terminal window will open up a page in your default browser. On this page, the spinal cord is displayed slice by slice. It has also been cropped from the overall anatomical image to provide a quick overview. The segmentation is displayed using a red overlay that can be toggled by repeatedly pressing the right arrow key. More information about QC reporting can be found on the <link to QC reporting> page.


Using FSLeyes
*************

If you have `FSLeyes <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes>`_ installed, the script will also output a second command to inspect the results. This will be true for all commands run in this tutorial. (**Note:** The exact filepath will vary depending on your filesystem.)

.. code:: sh

   Done! To view results, type:
   fsleyes sct_course_london20/single_subject/data/t2/t2.nii.gz -cm greyscale sct_course_london20/single_subject/data/t2/t2_seg.nii.gz -cm red -a 100.0 &

As with the Quality Control page, the spinal cord segmentation is displayed in red on top of the anatomical image. Further guidance on the usage of FSLeyes can be found in the `FSL Course <https://fsl.fmrib.ox.ac.uk/fslcourse/lectures/practicals/intro1/index.html>`_.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/t2_fsleyes.png
  :align: center
  :figwidth: 75%

  The segmented image opened in FSLeyes


