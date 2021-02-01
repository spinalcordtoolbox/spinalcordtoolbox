Voxels Space Orientation and Coordinate Conventions
***************************************************

Images
======

It is important to note that SCT Images, which are derived from NIFTI images, have their contents indexed in "Fortran order", meaning that in an image of shape :math:`(N_a, N_b, N_c)`, where we consider the `a` axis to be the first, two consecutive (in the sense of storage location) elements are in the `a` dimension, the first.
This is by opposition to the C ordering which is more widely used in most software, and where the fastest varying element is indexed last.

.. _reference_spaces:

Reference Spaces
================

As in many other tools, SCT follows a standard nomenclature for reference spaces in which the world or local coordinates are expressed.

The string is formed from character label among (relative to a human subject):

- `L` / `R`: left-right
- `P` / `A`: posterior-anterior
- `I` / `S`: inferior-superior

The character position corresponds to the axis index.

SCT uses the "from" convention, which for clarity we postfix by a dash.

The reference space for physical coordinates is LPI- (which is coming from nibabel and NIFTI).


An "image orientation" corresponds to the orientation of the surface/volume with regard to the reference orientation. It is encoded in the (NIFTI) file header.


For example, a `RAS` image orientation corresponds to a 3D image with:

- X axis oriented `L` towards `R`;
- Y axis oriented `P` towards `A`;
- Z axis oriented `I` towards `S`.


Notes:

- nibabel, BIDS
  are using the "towards" convention, ie. SCT's LPI(-) is their RAS(+).


.. _coordinates:

Coordinate Conventions
======================


Local/Voxel Coordinates
+++++++++++++++++++++++

When voxel coordinates are integers, coordinates are indices. Indices are expressed starting from 0 and up to N-1 where N is the number of voxels in the considered dimension.

When voxel coordinates are real numbers, we are using an *integer voxel center convention* (consistent with nibabel and NIFTI).

This means that a coordinate such as :code:`(i,j,k) == np.round((i,j,k))` expresses the center of a voxel.

NB: Voxel coordinates are called :math:`(i,j,k)` in the NIFTI documentation.



Global/Physical Coordinates
+++++++++++++++++++++++++++

Physical coordinates are always expressed as real numbers. They are defined from the relation expressed by the transform and unit system expressed in a image header.

Physical coordinates are expressed relative to the LPI- frame, considering the voxel dimensions, affine transform between voxel coordinates and world coordinates, and the physical dimension unit, all of which is encoded in the NIFTI file header.

NB: Voxel coordinates are called :math:`(x,y,z)` in the NIFTI documentation.


References
==========

- `An introduction to the NIFTI file format. <https://brainder.org/2012/09/23/the-nifti-file-format/>`_
  See *§ Orientation information* and around.

- `Official definition of the nifti1 header <https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h>`_
  See *§ 3D IMAGE (VOLUME) ORIENTATION AND LOCATION IN SPACE*

- `nipy/nibabel's documentation on coordinate systems
  <http://nipy.org/nibabel/coordinate_systems.html#naming-reference-spaces>`_

- ITK (`ANTs <https://sourceforge.net/p/advants/discussion/840261/thread/2a1e9307/#fb5a>`_,
  `Slicer <https://www.slicer.org/wiki/Coordinate_systems>`_) reference coordinate system is different (LPS-).

- `Matlab FieldTrip toolbox "How are the different head and MRI coordinate systems defined?"
  <http://www.fieldtriptoolbox.org/faq/how_are_the_different_head_and_mri_coordinate_systems_defined>`_
