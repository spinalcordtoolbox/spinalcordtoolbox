.. _spinalcord-segmentation:

Spinal cord segmentation
########################

This tutorial demonstrates how to use SCT's command-line scripts to segment spinal cords from anatomical MRI images of the spine. It is intended to be completed from start to finish, as it compares two different algorithms provided by SCT. It is meant to give you a feel for common usage of these tools on real-world data.

.. warning::

   This tutorial uses sample MRI images that must be retrieved beforehand. Please download and unzip `sct_course_london20.zip <https://osf.io/bze7v/?action=download>`_ , then open up the unzipped folder in your terminal and verify its contents using ``ls``.

   .. code:: sh

      ls
      # Output:
      # multi_subject single_subject

   We will be using images from the ``single_subject/data`` directory, so navigate there and verify that it contains subdirectories for various MRI image contrasts using ``ls``.

   .. code:: sh

      cd single_subject/data
      ls
      # Output:
      # dmri  fmri  LICENSE.txt  mt  README.txt  t1  t2  t2s

.. note::

   Both ``sct_propseg`` and ``sct_deepseg_sc`` are designed to work with four common MRI image contrasts: T1 weighted, T2 weighted, T2* weighted, and DWI. If your image data uses a contrast not listed here, select the closest visual match to the available options. For example, FMRI images have bright cerebrospinal fluid (CSF) regions and dark spinal cord regions, so the T2 contrast option would be an appropriate choice.

   .. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/image_contrasts.png
      :align: center
      :figwidth: 75%

      Image contrasts and their corresponding ``-c`` settings

Algorithm #1: ``sct_propseg``
*****************************

SCT provides two command-line scripts for segmenting the spinal cord. The first of these is called ``sct_propseg``. This tutorial will explain the how the script works from a high-level theoretical perspective, and then it will provide two usage examples.

Theory
------

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/optic_steps.png
   :align: right
   :figwidth: 20%

   Centerline detection using OptiC

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/mesh_propagation.png
   :align: right
   :figwidth: 20%

   3D mesh propagation using PropSeg

``sct_propseg`` itself is a single command, but internally it uses three processing steps to segment the spinal cord.

#. Detect the approximate center of the spinal cord automatically using a machine learning-based method (OptiC). This is an initialization step for the core algorithm, PropSeg.
#. Create a coarse 3D mesh by propagating along the spinal cord (PropSeg).
#. Refine the surface of the mesh using small adjustments.

   .. note::

      The centerline detection step is also provided in a standalone script called ``sct_get_centerline``.

Example: T2w image
------------------

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/t2_image.png
  :align: right
  :figwidth: 10%

  t2.nii.gz

From the ``single_subject/data`` folder, navigate to the ``t2`` directory, and verify that it contains a single T2-weighted image:

.. code:: sh

   cd t2
   ls
   # Output:
   # t2.nii.gz

Running the script
^^^^^^^^^^^^^^^^^^

Next, run the following command to process the image:

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/t2_qc.png
  :align: right
  :figwidth: 40%

  The QC report for the segmented image

.. code:: sh

   sct_propseg -i t2.nii.gz -c t2 -qc ~/qc_singleSubj

Note that we have provided three arguments:

- ``-i``, which indicates the input image (``t2.nii.gz``).
- ``-c``, which indicates the contrast of the image (``t2``).
- ``-qc``, the directory for Quality Control reporting (``~/qc_singleSubj``). QC reports will allow us to evaluate the segmentation slice-by-slice.

During execution, the script will provide status updates as it progress through its various stages.

Inspecting the results using QC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When complete, the script will output a command to inspect the results. (**Note:** The exact filepath will vary depending on your filesystem.)

.. code:: sh

   Use the following command to see the results in a browser:
   xdg-open "sct_course_london20/single_subject/data/t2/qc_singleSubj/index.html"

Running this command in your Terminal window will open up a page in your default browser. On this page, the spinal cord is displayed slice by slice. It has also been cropped from the overall anatomical image to provide a quick overview. The segmentation is displayed using a red overlay that can be toggled by repeatedly pressing the right arrow key. More information about QC reporting can be found on the <link to QC reporting> page.

Inspecting the results using FSLeyes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have FSLeyes installed, the script will also output a second command to inspect the results. This will be true for all commands run in this tutorial. (**Note:** The exact filepath will vary depending on your filesystem.)

.. code:: sh

   Done! To view results, type:
   fsleyes sct_course_london20/single_subject/data/t2/t2.nii.gz -cm greyscale sct_course_london20/single_subject/data/t2/t2_seg.nii.gz -cm red -a 100.0 &

As with the Quality Control page, the spinal cord segmentation is displayed in red on top of the anatomical image. Further guidance on the usage of FSLeyes can be found in the `FSL Course <https://fsl.fmrib.ox.ac.uk/fslcourse/lectures/practicals/intro1/index.html>`_.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/t2_fsleyes.png
  :align: center
  :figwidth: 75%

  The segmented image opened in FSLeyes

Example: T1w image
------------------

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/t1_image.png
  :align: right
  :figwidth: 8%

  t1.nii.gz

Next, we will navigate to the T1 directory and verify that it contains a single T1-weighted image. If you are still in the T2 directory from the previous section, this can be done as follows:

.. code:: sh

   cd ../t1
   ls
   # Output
   # t1.nii.gz

Running the script
^^^^^^^^^^^^^^^^^^

Once here, we can run the ``sct_propseg`` command to process the image:

.. code:: sh

   sct_propseg -i t1.nii.gz -c t1 -qc ~/qc_singleSubj

This command is identical to the previous step, apart from the ``-c`` argument to indicate a different contrast.

Inspecting the results using QC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/t1_propseg_before_after.png
  :align: right
  :figwidth: 20%

  Segmentation leakage with ``sct_propseg``

As before, a Quality Control report command will be output when the script is complete. You may also simply refresh the webpage generated in the T2 section to see the new T1 results.

This time, however, there is an issue. The spinal cord segmentation has leaked outside of the expected area. This is caused by a bright outer region that is too close to the spinal cord. ``sct_propseg`` relies on contrast between the CSF and the spinal cord; without sufficient contrast, the segmentation may fail (as it has here).

Fixing a failed segmentation
----------------------------

To combat segmentation issues like this, there are several approaches that you can take:

- Manually correct the segmentation.
- Modify the input parameters for ``sct_propseg``.

  - You can generate a list of available parameters using the command ``sct_propseg -h``.
  - **Note:** This usage is more advanced, so instructions are provided in a separate tutorial, :ref:`correcting_sct_propseg`.

- Use the second segmentation algorithm that SCT provides, called ``sct_deepseg_sc``.

Algorithm #2: ``sct_deepseg_sc``
********************************

Theory
------

As its name suggests, ``sct_deepseg_sc`` is based on deep learning. It is a newer algorithm, having been introduced to SCT in 2018. The steps of the algorithm are as follows:

#. A convolutional neural network is used to generate a probablistic heatmap for the location of the spinal cord.
#. The heatmap is fed into the OptiC algorithm to detect the spinal cord centerline.
#. The spinal cord centerline is used to extract a patch from the image.

   - We extract a patch to help combat class imbalance. If the full image were to be used instead, the spinal cord region would be small in proportion to the non-spinal cord regions of the image, and thus harder to detect.

#. Lastly, a second convolutional neural network is applied to the extracted patch to segment the spinal cord.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/sct_deepseg_sc_steps.png
   :align: center
   :figwidth: 65%

   The steps for ``sct_deepseg_sc``

Example: T1w image
------------------

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/t1_deepseg_before_after.png
   :align: right
   :figwidth: 20%

   No leakage with ``sct_deepseg_sc``

Running the script
^^^^^^^^^^^^^^^^^^

Since we aim to improve the T1 segmentation, ensure that you are still in the T1 directory (``sct_course_london20/single_subject/data/t1``). Once there, run this command:

.. code:: sh

   sct_deepseg_sc -i t1.nii.gz -c t1 -qc ~/qc_singleSubj -ofolder deepseg

Much like ``sct_propseg``, we use the same values for ``-i``, ``-c``, and ``-qc``. In this case, however, we have added an additional ``-ofolder`` command. This is so that we do not overwrite the results generated in the previous steps, which allows us to compare the output of both algorithms. ``-ofolder`` is not strictly necessary, however.

Inspecting the results using QC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once again, you may either execute the command given by the script, or simply refresh the QC webpage from the previous examples.

In this case, ``sct_deepseg_sc`` has managed to improve upon the results of ``sct_propseg``.

Choosing between ``sct_propseg`` and ``sct_deepseg_sc``
*******************************************************

Although ``sct_deepseg_sc`` was introduced as a follow-up to the original ``sct_propseg``, choosing between the two is not as straightfoward as it may seem. Neither algorithm is strictly superior in all cases; whether one works better than the other is data-dependent. Given the variation in imaging data (imaging centers, sizes, ages, coil strengths, contrasts, scanner vendors, etc.) SCT recommends to try both algorithms with your pilot scans to evaluate the merit of each on your specific dataset, then stick with a single method throughout your study.

Development of these approaches is an iterative process, and the data used to develop these approaches evolves over time. If you have input regarding what has worked (or hasn't worked) for you, we would be happy to hear your thoughts in the `Spinal Cord MRI forums <https://forum.spinalcordmri.org/>`_, as it could help to improve the toolbox for future users.
