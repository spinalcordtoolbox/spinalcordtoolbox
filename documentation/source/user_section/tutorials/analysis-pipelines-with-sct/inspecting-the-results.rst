Inspecting the results of processing
####################################

After running the entire pipeline, you should have all results under the ``output/results/`` folder.

CSA for T2 data
---------------

Here for example, we show the mean CSA averaged between C2-C3 levels computed from the T2 data. Each line represents a subject.

.. csv-table:: ``CSA.csv``: CSA values in T2 data across 3 subjects
   :file: CSA.csv
   :header-rows: 1

The variability is mainly due to the inherent variability of CSA across subjects.

MTR in white matter
-------------------

Here are the results of MTR quantification in the dorsal column of each subject between C2 and C5. Notice the remarkable inter-subject consistency.

.. csv-table:: ``MTR_in_DC.csv``: MTR values values in white matter (dorsal columns) across 3 subjects
   :file: MTR_in_DC.csv
   :header-rows: 1

.. _batch-qc_report:

Quality Control report
----------------------

Because the ``-qc`` flag was used for each of the SCT commands within the ``process_data.sh`` batch script, a QC report is generated under the ``qc/`` folder. It can be viewed by running ``open qc/index.html``, or by double-clicking on the ``index.html`` file:

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/batch-processing-of-subjects/qc.png
   :align: center

When processing multiple subjects (e.g., using :ref:`sct_run_batch`), QC reports are especially useful as they have several features to help quickly assess multiple images all at once:

- The columns of the QC report can be sorted. For example, you can sort by "Function" to review the :ref:`sct_deepseg` outputs for all of the subjects together, then all of the :ref:`sct_label_vertebrae` outputs, and so on.
- You can also use keyboard shortcuts to quickly skim through subjects. The arrow keys can be used to switch subjects and toggle the overlay, while the 'F' key can be used to mark subjects as "Fail", "Artifact", or "Pass".
- You can then use the "Export Fails" button to export failing subjects as a ``qc_fail.yml`` file, to be used alongside SCT's manual correction scripts.

On the following page, we will see how to make use of this ``qc_fail.yml`` file for manual correction.

.. note::

   For a deeper look at QC report features, you can also refer to the :ref:`qc` section of the :ref:`inspecting-your-results` page.
