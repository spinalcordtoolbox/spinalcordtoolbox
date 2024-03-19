Types of vertebral labels
#########################

Before we can label the spinal cord vertebral levels, we must first describe the types of labels we can create. SCT's uses two types of vertebral labels:

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/vertebral-labeling/vertebral-labeling-label-types.png
   :align: center
   :figwidth: 400px

   A visual comparison of the different label types used in SCT.

1. Full body labels (used for CSA calculation)
2. Point (single voxel) labels (used for landmark-matching during registration between two images)

This tutorial will demonstrate how to create both types of labels, and subsequent tutorials will demonstrate how to apply those labels to compute CSA and register images to the PAM50 template.