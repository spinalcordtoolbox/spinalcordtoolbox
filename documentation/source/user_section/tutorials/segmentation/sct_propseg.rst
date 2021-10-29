Algorithm #1: ``sct_propseg``
#############################

The first spinal cord segmentation tool that SCT offers is called ``sct_propseg``, and it works as follows:

:1. Centerline detection:
   ``sct_propseg`` starts by using a machine learning-based method (`OptiC <https://archivesic.ccsd.cnrs.fr/PRIMES/hal-01713965v1>`_) to automatically detect the approximate center of the spinal cord.

   .. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/optic_steps.png
      :align: center
      :figwidth: 500px

   (The centerline detection step is also provided in a standalone script called ``sct_get_centerline``.)

:2. Mesh propagation:
   Next, a coarse 3D mesh is created by propagating along the spinal cord (PropSeg).

   .. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/mesh_propagation.png
      :align: center
      :figwidth: 500px

:3. Surface refinement:
   Finally, the surface of the mesh is refined using small adjustments.

