.. _how-many-labels:

How many vertebral labels should I use for registration?
########################################################

If you plan on using the labeled spinal cord to perform registration, you don't necessarily have to label every vertebrae. Instead, you can choose to work with a subset of labels:

:3+ labels:
   This is the most accurate method for matching the vertebral levels of the anatomical image to the levels of the template. However, there is a key downside to this approach: Because more than two points must be matched, the level-matching transformation cannot be affine. As a result, the output warping field will be undefined for regions above the top label and below the bottom label.

:2 labels:
   In practice, the difference in accuracy between using 3+ labels and 2 labels is often negligible. Using 2 labels also has the added benefit of allowing for an affine level-matching transformation, which means the template-to-image warping field will be defined for the entire image. For these reasons, we strongly recommend starting with 2 labels for your registration.

:1 label:
   If your image covers only 1 vertebrae, you can still provide a single label. Note that the transformation in this case will be limited to a Z-axis translation, as an affine transformation can't be determined for a single point.

As we recommend starting with 2 labels, the :ref:`next page<extracting-specific-labels>` of this tutorial will show you how to extract a subset of labels from a  fully-labeled spinal cord.
