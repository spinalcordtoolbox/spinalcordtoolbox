What if things go wrong?
########################

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/batch-processing-of-subjects/segmentation-issue.png
   :align: center

If you spot any issues during QC, e.g., error during segmentation, you can correct the issue (manually or by adjusting SCT parameters), and then re-run the processing.

``process_data.sh`` contains a number of convenince functions that will select any file with suffix ``_manual``, if it exists, then use that file (instead of automatically computing, e.g., a segmentation). That way, you can manually process outlier subjects without interrupting the automated processing for the dataset as a whole.
