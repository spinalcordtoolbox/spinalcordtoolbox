Modifying ``info_label.txt`` to add custom tracts to your analysis
##################################################################

The file ``info_label.txt`` lists the NIfTI files corresponding ot each white matter tract (ID 0—>36), as well as the “combined labels” which could be used for quantifying metrics within a combination of tracts. This file is present inside each copy of the PAM50 template.

If you want to add more tract combinations for all of the subjects in your study, you can edit the original ``info_label.txt`` file under your SCT installation folder (``$SCT_DIR/data/PAM50/atlas/info_label.txt``). That way, each time you run ``sct_warp_template``, the modified file will be copied (alongside the warped template and atlas) for each new subject.

In the example below, we have added two ensembles of tracts corresponding to the right and left hemi-cord:

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/gm-wm-metric-computation/custom-tracts.png
   :align: center