Labeling conventions
####################

Next, the segmented spinal cord must be labeled to provide reference markers for matching the PAM50 template to subject's MRI. SCT accepts two different conventions for vertebral labels:

1. **Vertebral levels**: For this type of label file, labels should be placed as though the vertebrae were projected onto the spinal cord, with the label centered in the middle of each vertebral level.
2. **Intervertebral discs**: For this type of label file, labels should placed on the posterior tip of each disc.

   * **Note:** For disc labeling, SCT's tools can generate additional labels beyond discs, including the pontomedullary groove (label 49) and pontomedullary junction (label 50). The latter can be used as part of a :ref:`PMJ-based method <csa-pmj>` for computing the CSA of the spinal cord.

For image registration, you can provide either vertebral body labels or disc labels, as the decision does not significantly impact the performance of the registration.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/vertebral-labeling/vertebral-labeling-conventions.png
   :align: center
   :figwidth: 600px

   Conventions for vertebral and disc labels.