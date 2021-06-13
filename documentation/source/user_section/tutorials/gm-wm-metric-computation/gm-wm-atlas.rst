Atlas-based analysis
####################

While the white and gray matter segmentations are useful in their own right, there is great value in  extracting metrics from even more specific regions of the white and gray matter (e.g. individual tracts)  using an atlas.

In SCT, a digitalized version of the Gray’s Anatomy spinal cord white matter atlas [Standring, Gray's Anatomy 2008] was merged with the PAM50 template. For more information on the procedure, see `[Lévy et al., Neuroimage 2015] <https://pubmed.ncbi.nlm.nih.gov/26099457/>`_.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/gm-wm-metric-computation/white-matter-atlas.png
   :align: center

The atlas is composed of 30 WM tracts, 6 GM regions and the surrounding CSF. The reason for including the CSF is to be able to account for partial volume effect during metric estimation, as will be explained further in the following pages.