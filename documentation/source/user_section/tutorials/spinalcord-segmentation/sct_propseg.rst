Algorithm #1: ``sct_propseg``
#############################

SCT provides two command-line scripts for segmenting the spinal cord. The first of these is called ``sct_propseg``. This tutorial will explain the how the script works from a high-level theoretical perspective, and then it will provide two usage examples.

Theory
------

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/optic_steps.png
   :align: right
   :figwidth: 20%

   Centerline detection using OptiC

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/mesh_propagation.png
   :align: right
   :figwidth: 20%

   3D mesh propagation using PropSeg

``sct_propseg`` itself is a single command, but internally it uses three processing steps to segment the spinal cord.

#. Detect the approximate center of the spinal cord automatically using a machine learning-based method (`OptiC <https://archivesic.ccsd.cnrs.fr/PRIMES/hal-01713965v1>`_). This is an initialization step for the core algorithm, PropSeg.
#. Create a coarse 3D mesh by propagating along the spinal cord (PropSeg).
#. Refine the surface of the mesh using small adjustments.

   .. note::

      The centerline detection step is also provided in a standalone script called ``sct_get_centerline``.