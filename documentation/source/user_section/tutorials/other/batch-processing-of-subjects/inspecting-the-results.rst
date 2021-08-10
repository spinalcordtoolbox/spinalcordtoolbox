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

Quality Control report
----------------------

A QC report is generated under ``qc/``. As shown before, the QC report is useful to quickly assess the quality of the analysis pipeline.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/batch-processing-of-subjects/qc.png
   :align: center