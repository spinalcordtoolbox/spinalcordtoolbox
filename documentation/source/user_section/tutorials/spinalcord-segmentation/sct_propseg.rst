Algorithm #1: ``sct_propseg``
#############################

The first spinal cord segmentation tool that SCT offers is called ``sct_propseg``, and it works as follows:

:1. Centerline detection:
   ``sct_propseg`` starts by usinga machine learning-based method (`OptiC <https://archivesic.ccsd.cnrs.fr/PRIMES/hal-01713965v1>`_) to automatically detect the approximate center of the spinal cord.

   (The centerline detection step is also provided in a standalone script called ``sct_get_centerline``.)

   .. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/optic_steps.png
      :align: left
      :figwidth: 500px

:2. Mesh propagation:
   Next, it creates a coarse 3D mesh by propagating along the spinal cord (PropSeg).

   .. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/mesh_propagation.png
      :align: left
      :figwidth: 500px

:3. Surface refinement:
   Finally, it refine the surface of the mesh using small adjustments.

