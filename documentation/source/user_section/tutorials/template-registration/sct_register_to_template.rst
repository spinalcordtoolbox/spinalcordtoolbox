Registration algorithm: ``sct_register_to_template``
####################################################

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/thin-plate-straightening.png
   :align: right
   :figwidth: 300px

   Spinal cord straighting using thin-plate spline interpolation.

After spinal cord labeling, we can register the subject with the template. If we have the labeled spinal cord, we can register the anatomical image to the template. SCT provides the ``sct_register_to_template`` command for template registration. Here are the steps for the algorithm within this command:

1. **Straightening:** The straightening works by finding, for each point along the spinal cord, the mathematical transformation to go from a curved centerline to a straight centerline.

   * The straightening algorithm computes the orthogonal plane at each point along the centerline, then constructs a straight space in the output using thin-plate spline interpolation. This allows the inner geometry of the cord to be preserved.
   * The straightening algorithm outputs a forward and a backward warping field (ITK-compatible), which can be concatenated with subsequent transformations, as will be seen later.

2. **Vertebrae-matching transformation**: Once straightened, the next step involves a transformation to match the vertebral levels of the subject to that of the template. If 2 labels are provided, this transformation will be affine; if 3+ labels are provided, this transformation will be non-affine. (Note: This step focuses only on matching the coordinates of the labels, and does not consider the shape of the spinal cord, which is handled by the next step.)
3. **Shape-matching transformation**: A multi-step nonrigid deformation is estimated to match the subject’s cord shape to the template. By default, two steps are used: the first handles large deformations, while the second applies fine adjustments.

.. important::

   SCT provides many additional nonrigid deformation algorithms beyond the default configuration. You can visit the :ref:`customizing-registration-section` page to learn how to optimize the registration procedure for your particular contrast, resolution, and spinal cord geometry.