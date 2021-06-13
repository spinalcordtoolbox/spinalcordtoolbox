The partial volume effect
#########################

Let’s imagine that the metric value you would like to quantify is 50 in the WM and 0 in the CSF. Because of the coarse resolution of MRI, the apparent value within the voxel will be a mixture between the WM and CSF compartment, yielding the value 25. This phenomenon is called “partial volume effect”.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/gm-wm-metric-computation/partial-volume-effect.png
   :align: center

SCT provides several methods to account for this effect when extracting metrics from images.