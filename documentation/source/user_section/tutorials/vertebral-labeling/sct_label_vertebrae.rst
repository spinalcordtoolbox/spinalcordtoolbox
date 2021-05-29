.. _vert-labeling-section:

Labeling algorithm: ``sct_label_vertebrae``
###########################################

SCT provides the ``sct_label_vertebrae`` command for vertebral labeling. Here are the steps for the algorithm within this command:

  #. **Straightening**: The spinal cord is straightened to make it easier to use a moving window-based approach in a subsequent step.
  #. **C2-C3 disc detection:** The C2-C3 disc is used as a starting point because it is a distinct disc that is easy to detect (compared to, say, the T7-T9 discs, which are indistinct compared to one another).
  #. **Labeling of neighbouring discs**: The neighbouring discs are found using a similarity measure with the PAM50 template at each specific level.
  #. **Un-straightening**: Finally, the spinal cord and the labeled segmentation are both un-straightened, and the labels are saved to image files.

The vertebral/disc labeling algorithm has the following features:

  - **Contrast-independent**: Can be used on images regardless of their contrast type.
  - **Produces both label types:** Labels are produced for both vertebral levels and intervertebral discs.
  - **Robust to missing discs:** The labeling algorithm uses several priors from the template, including the probabilistic distance between adjacent discs and the size of the vertebral discs. These priors allow it to be robust enough to handle cases where instrumentation results in missing discs or susceptibility artifacts. *(See the figure below.)*

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/instrumentation-missing-discs.png
   :align: center
   :figwidth: 400px

   ``sct_label_vertebrae`` is able to label vertebral levels despite missing discs due to instrumentation.