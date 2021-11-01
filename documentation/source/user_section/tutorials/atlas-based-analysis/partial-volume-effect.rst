.. _partial-volume-effect:

The partial volume effect
#########################

While the atlas itself is detailed enough to capture the intricate structure of the white matter tracts, your imaging data likely does not have the resolution to capture this same detail. Instead, it is common for your image to be represented by coarse voxels, with each voxel spanning multiple tracts.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/gm-wm-metric-computation/partial-volume-effect.png
   :align: center

This means that if the *actual* metric value you would like to quantify is, say, 50 in the WM and 0 in the CSF, then the *apparent* value within the voxel will be a mixture between the WM and CSF compartment, yielding the value 25. This phenomenon is called “partial volume effect”.

SCT provides several methods to account for this effect when extracting metrics from images.