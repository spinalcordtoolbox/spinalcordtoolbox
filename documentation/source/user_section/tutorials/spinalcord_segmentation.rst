.. _spinalcord-segmentation:

Spinal cord segmentation
########################

This tutorial demonstrates how to use SCT's command-line scripts to segment spinal cords from anatomical MRI images of the spine. It is intended to be completed from start to finish, as it compares two different algorithms provided by SCT. It is meant to give you a feel for common usage of these tools on real-world data.

.. warning::

   This tutorial uses sample MRI images that must be retrieved beforehand. Please download and unzip `sct_course_london20.zip <https://osf.io/bze7v/?action=download>`_ , then open up the unzipped folder in your terminal of choice before continuing.

Algorithm #1: ``sct_propseg``
*****************************

SCT provides two command-line scripts for segmenting the the spinal cord. The first of these is called ``sct_propseg``. This tutorial will explain the how the script works from a high-level theoretical perspective, and then it will provide two usage examples.

Theory
------

``sct_propseg`` itself is a single command, but internally it uses three processing steps to segment the spinal cord.

#. Detect the approximate center of the spinal cord automatically using a machine learning-based method (OptiC). This is an initialization step for the core algorithm, PropSeg.

   - **Note:** This spinal cord detection step is also provided in a standalone script. If you want just the centerline detection and not the full spinal cord segmentation, please refer to ``sct_get_centerline``.

#. Create a coarse 3D mesh by propagating along the spinal cord (PropSeg).
#. Refine the surface of the mesh using small adjustments.

A note regarding image contrasts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sct_propseg`` is designed to work with four common MRI image contrasts: T1 weighted, T2 weighted, T2* weighted, and DWI. The contrast of choice must be indicated using a command-line argument when the script is called.

**Note**: If your image data uses a contrast not listed here, please select the closest visual match to the available options. For example, FMRI images have dark cerebrospinal fluid (CSF) regions and bright spinal cord regions, so the T2 contrast option would be an appropriate choice.

Example: T2w image
------------------

Running the script
^^^^^^^^^^^^^^^^^^

To begin, please open your terminal and navigate to the ``sct_course_london20`` directory that was created at the start of this tutorial. You should see the following when you list the contents using the ``ls`` command:

.. code:: sh

   $ ls
   multi_subject single_subject

Next, navigate to the ``T2`` directory containing the T2-weighted image as follows:

.. code:: sh

   $ cd single_subject/data/t2
   $ ls
   t2.nii.gz

As you can see, there is a single T2-weighted image within this directory. Once here, we can run the ``sct_propseg`` command to process the image:

.. code:: sh

   $ sct_propseg -i t2.nii.gz -c t2 -qc ~/qc_singleSubj

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
   xdg-open "/home/joshua/Desktop/sct_course_london20/single_subject/data/t2/qc_singleSubj/index.html"

Running this command in your Terminal window will open up a page in your default browser. On this page, the spinal cord is displayed slice by slice. It has also been cropped from the overall anatomical image to provide a quick overview. The segmentation is displayed using a red overlay that can be toggled by repeatedly pressing the right arrow key. More information about QC reporting can be found on the <link to QC reporting> page.

Inspecting the results using FSLEyes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have fsleyes installed, the script will also output a second command to inspect the results. This will be true for all commands run in this tutorial. (**Note:** The exact filepath will vary depending on your filesystem.)

.. code:: sh

   Done! To view results, type:
   fsleyes /home/joshua/Desktop/sct_course_london20/single_subject/data/t2/t2.nii.gz -cm greyscale /home/joshua/Desktop/sct_course_london20/single_subject/data/t2/t2_seg.nii.gz -cm red -a 100.0 &

As with the Quality Control page, the spinal cord segmentation is displayed in red on top of the anatomical image. Further guidance on the usage of FSLEyes can be found in the `FSL Course <https://fsl.fmrib.ox.ac.uk/fslcourse/lectures/practicals/intro1/index.html>`_.


Example: T1w image
------------------

Running the script
^^^^^^^^^^^^^^^^^^

To repeat the process on a T1-weighted image, navigate to the T1 directory. If you are still in the T2 directory from the previous section, this can be done as follows:

.. code:: sh

   $ cd ../t1
   $ ls
   t1.nii.gz

As you can see, there is a single T1-weighted image within this directory. Once here, we can run the ``sct_propseg`` command to process the image:

.. code:: sh

   $ sct_propseg -i t1.nii.gz -c t1 -qc ~/qc_singleSubj

This command is identical to the previous step, apart from the ``-c`` argument to indicate a different contrast.

Inspecting the results using QC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As before, a Quality Control report command will be output when the script is complete. You may also simply refresh the webpage generated in the T2 section to see the new T1 results.

This time, however, there is an issue. The spinal cord segmentation has leaked outside of the expected spinal cord region. This is caused by a thin CSF region, which poorly separates the spinal cord from the outer vertebrae. ``sct_propseg`` relies on contrast between the CSF and the spinal cord; without contrasting regions, the segmentation may fail (as it has here).

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

Step 2 is shared between ``sct_deepseg_sc`` and ``sct_propseg``, but steps 1, 3, and 4 are entirely new.

Example: T1w image
------------------

Running the script
^^^^^^^^^^^^^^^^^^

Since we aim to improve the T1 segmentation, ensure that you are still the T1 directory (``sct_course_london20/single_subject/data/t1``). Once there, run this command:

.. code:: sh

   $ sct_deepseg_sc -i t1.nii.gz -c t1 -qc ~/qc_singleSubj -ofolder deepseg

Much like ``sct_propseg``, we use the same values for ``-i``, ``-c``, and ``-qc``. In this case, however, we have added an additional ``-ofolder`` command. This is so that we do not overwrite the results generated in the previous steps, which allows us to compare the output of both algorithms. ``-ofolder`` is not strictly necessary, however.

Inspecting the results using QC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once again, you may either execute the command given by the script, or simply refresh the QC webpage from the previous examples.

In this case, ``sct_deepseg_sc`` has managed to improve upon the results of ``sct_propseg``.

Choosing between ``sct_propseg`` and ``sct_deepseg_sc``
*******************************************************

Although ``sct_deepseg_sc`` was introduced as a follow-up to the original ``sct_propseg``, choosing between the two is not as straightfoward as it may seem. Neither algorithm is strictly superior in all cases; whether one works better than the other is data-dependent. Given the variation in imaging data (imaging centers, sizes, ages, coil strengths, contrasts, scanner vendors, etc.) SCT recommends to try both algorithms in your pilot studies to evaluate the merit of each on your specific dataset.

Development of these approaches is an iterative process, and the data used to develop these approaches evolves over time. Your experience with these approaches is valuable to us. If you have input regarding what has worked (or hasn't worked) for you, we would be happy to hear your thoughts in the `Spinal Cord MRI forums <https://forum.spinalcordmri.org/>`_, as it could help to improve the toolbox for future users.
