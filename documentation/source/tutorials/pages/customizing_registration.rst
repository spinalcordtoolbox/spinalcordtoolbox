.. _customizing-registration-section:

Customizing ``sct_register_to_template`` and ``sct_register_multimodal``
########################################################################

This page provides recommendations for how to adjust the registration commands if the default parameters are insufficient for your specific data and pipeline.

The ``-param`` argument
***********************

The ``-param`` argument defines the transformations for each step of the registration process. The easiest way to use ``-param`` is to start with the default values, and adjust one parameter at a time.

.. code-block::

   # Default values for '-param'. (Note: Normally, this would be written as one line, but a line break was added for readability.)
   -param step=1,type=imseg,algo=centermassrot,metric=MeanSquares,iter=10,smooth=0,gradStep=0.5,slicewise=0,smoothWarpXY=2,pca_eigenratio_th=1.6:
          step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,iter=3,smooth=1,gradStep=0.5,slicewise=0,smoothWarpXY=2,pca_eigenratio_th=1.6

Step 1 and step 2 can be modified, and additional steps can be added. In general, we recommend that you start with coarse-adjustment steps, then apply finer adjustments with each successive step.

.. note:: Because choosing the right configuration for your data can be overwhelming, feel free to visit the `SCT forum <https://forum.spinalcordmri.org/c/sct/>`_ where you can ask for clarification and guidance.

Common adjustments to ``-param``
================================

* ``type`` controls the type of data used for registration:

   * ``type=im``: Use the input anatomical image.
   * ``type=seg``: Use a separate segmentation image. Choose this if your image data contains distortions or artifacts, but you are confident in your segmentation. (For example, if you have manually corrected the segmentation.)
   * ``type=imseg``: Use both the input image and a segmentation. Only for use with ``algo=centermassrot``.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/sct_register_to_template-param-algo.png
  :align: right
  :figwidth: 40%

* ``algo`` determines the type of nonrigid deformation to apply to the spinal cord at each step:

   - ``algo=translation``: Axial (X-Y) translation.
   - ``algo=rigid``: Axial (X-Y) translation + rotation about the Z axis.
   - ``algo=affine``: Axial (X-Y) translation + rotation about the Z axis + axial (X-Y) scaling.
   - ``algo=syn``: Symmetric image normalization (`SyN <https://pubmed.ncbi.nlm.nih.gov/17659998/>`_) provided by `ANTs <https://stnava.github.io/ANTs/>`_.
   - ``algo=b-splinesyn``: B-Spline regularized form of SyN provided by `ANTs <https://stnava. github.io/ANTs/>`_.
   - ``algo=slicereg``: Slice-by-slide axial (X-Y) translation, regularized along the Z axis. This can be used as both an initial alignment of a segmented cord centerline, or to align the centers of two images that are already close.
   - ``algo=centermassrot``: An alignment of the center of mass of the segmented cord with the center of mass of the template. Similar to ``algo=slicereg``, but also includes rotation, to account for a turned cord due to e.g. compression on one side of the cord.
   - ``algo=columnwise``: This transformation involves a multi-step scaling operation with a much greater degree of freedom, so it is useful for highly compressed cords.

* ``metric``: Similarity metric etc.

   - ``metric=CC``
   - ``metric=MI``
   - ``metric=MeanSquares``:

* ``slicewise`` controls whether or not transformations should be computed on a slice-by-slice basis:

   * ``slicewise=1``: Apply deformations on a slice-by-slice basis.
   * ``slicewise=0``: Regularize transformations across the Z axis.

For information about other parameters, please view the help description of ``sct_register_multimodal -h``.

The ``-ref`` argument
*********************

The flag ``-ref`` lets you select the destination for registration: either the template (default) or the subject’s native space. The main difference is that when ``-ref template`` is selected,
the cord is straightened, whereas with ``-ref subject``, it is not.

When should you use ``-ref subject``? If your image is acquired axially with highly anisotropic resolution (e.g. 0.7x0.7x5mm), the straightening will produce through-plane interpolation errors. In that case, it is better to register the template to the subject space to avoid such inaccuracies.

The ``-ldisc`` argument
***********************

The approach described previously uses two labels at the mid-vertebral level to register the template, which is fine if you are only interested in a relatively small region (e.g. C2 —> C7). However, if your volume spans a large superior-inferior length (e.g., C2 —> L1), the linear scaling between your subject and the template might produce inaccurate vertebral level matching between C2 and L1. In that case, you might prefer to rely on all inter-vertebral discs for a more accurate registration.

Conversely, if you have a very small FOV (e.g., covering only C3/C4), you can create a unique label at disc C3/C4 (value=4) and use -ldisc for registration. In that case, a single translation (no scaling) will be performed between the template and the subject.

.. note::
   If more than 2 labels are provided, ``-ldisc`` is not compatible with ``-ref subject``. For more information, please see the help: sct_register_to_template -h