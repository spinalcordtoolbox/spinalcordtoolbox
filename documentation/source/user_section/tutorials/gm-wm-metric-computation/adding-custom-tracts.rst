Modifying ``info_label.txt`` to add custom tracts to your analysis
##################################################################

The file ``info_label.txt`` lists the NIfTI files corresponding ot each white matter tract (ID 0—>36), as well as the “combined labels” which could be used for quantifying metrics within a combination of tracts.

If you want to add more tract combinations for all of your studies, you can edit this file under your SCT installation folder (``$SCT_DIR/data/PAM50/atlas/info_label.txt``). Each time you will run ``sct_warp_template``, the modified file will be copied to each new subject.

In the example below, we have added two ensembles of tracts corresponding to the right and left hemi-cord:

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/gm-wm-metric-computation/custom-tracts.png
   :align: center