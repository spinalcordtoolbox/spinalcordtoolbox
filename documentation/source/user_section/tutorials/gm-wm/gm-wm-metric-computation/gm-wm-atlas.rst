Atlas-based analysis
####################

While it can be helpful to extract metrics using the white and gray matter segmentations alone, there is additional value in extracting metrics from more specific regions of the white and gray matter (e.g. individual tracts). This can be done using an atlas.

In SCT, a digitalized version of the Gray’s Anatomy spinal cord white matter atlas [Standring, Gray's Anatomy 2008] was merged with the :ref:`pam50`. For more information on the procedure, see `[Lévy et al., Neuroimage 2015] <https://pubmed.ncbi.nlm.nih.gov/26099457/>`_.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/gm-wm-metric-computation/white-matter-atlas.png
   :align: center

The atlas is composed of 30 WM tracts, 6 GM regions and the surrounding CSF. The reason for including the CSF is to be able to account for partial volume effect during metric estimation, as will be explained further in the following pages.