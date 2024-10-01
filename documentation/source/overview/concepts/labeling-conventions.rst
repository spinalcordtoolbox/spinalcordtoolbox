Point labeling conventions
##########################

SCT accepts two different conventions for single-voxel (point) vertebral labels:

1. **Vertebral levels**: For this type of label file, labels should be placed as though the vertebrae were projected onto the spinal cord, with the label centered in the middle of each vertebral level.
2. **Intervertebral discs**: For this type of label file, labels should placed on the posterio  r tip of each disc.

For image registration, you can provide either vertebral body labels or disc labels, as the decision does not significantly impact the performance of the registration.

.. note:: For disc labeling, SCT's tools can generate additional labels beyond discs, including the pontomedullary groove (label 49), pontomedullary junction (label 50), and conus medullaris (label 60), as shown below.

   The PMJ can be used as part of a :ref:`PMJ-based method <csa-pmj>` for computing the CSA of the spinal cord, while the conus medullaris can be used during :ref:`lumbar registration <lumbar-registration>`.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/vertebral-labeling/vertebral-labeling-conventions.png
   :align: center
   :figwidth: 600px

   Conventions for vertebral and disc labels.