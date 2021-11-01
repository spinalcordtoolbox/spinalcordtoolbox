Modifying ``info_label.txt`` to add custom tracts to your analysis
##################################################################

Inside the ``atlas`` folder containing the warped white-matter atlas, the file ``info_label.txt`` lists the NIfTI files corresponding to each white matter tract (ID 0—>36), as well as the “combined labels” which could be used for quantifying metrics within a combination of tracts. This file is present inside each copy of the PAM50 template.

If you would like to add additional combinations of tracts, you can edit the original ``info_label.txt`` file under your SCT installation folder (``$SCT_DIR/data/PAM50/atlas/info_label.txt``). That way, each time you run ``sct_warp_template``, the modified file will be copied for each new subject (alongside the warped template and atlas).

In the example below, we have added two ensembles of tracts corresponding to the right and left hemi-cord:

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/gm-wm-metric-computation/custom-tracts.png
   :align: center