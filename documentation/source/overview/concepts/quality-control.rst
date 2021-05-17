.. _qc:

Quality Control
***************

Some SCT tools can generate Quality Control (QC) reports. These reports consist of "appendable" HTML files, containing a table of entries and allowing to show, for each entry, animated images (background with overlay on and off).

To generate a QC report, add the ``-qc`` command-line argument, with the location (folder, to be created by the SCT tool), where the QC files should be generated. For example, take the following command:

.. code-block::

   sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2 -qc qc_folder


Because the ``-qc`` argument was provided, a Quality Control report will be generated, and instructions to open the report will be printed at the end of the terminal output.

.. code-block::

   Successfully generated the QC results in qc_folder/_json/qc_2021_05_13_151239.874958.json
   Use the following command to see the results in a browser:
   xdg-open "qc_folder/index.html"

Running the command will open up a page in your browser that looks like this:

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/quality-control/qc.png
  :align: center
  :figwidth: 75%

From this page, you can perform the following actions:

  * **Navigate between images** using the arrow keys or mouse.
  * **Toggle overlay** using the right arrow key.
  * **Evaluate and mark images** using the ``f`` key. Pressing once marks the highlighted image as "Pass", twice marks it as "Fail", and three times marks it as "Artifact".
  * **Export marked images** in a list format using the buttons at the bottom of the table.