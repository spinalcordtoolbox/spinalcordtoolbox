SCT Concepts
############


This section documents some SCT concepts and other useful things to know
when using it.

.. contents::
   :local:
..


Voxels Space Orientation and Coordinate Conventions
***************************************************


Images
======

It is important to note that SCT Images, which are derived from NIFTI
images, have their contents indexed in “Fortran order”, meaning that in
an image of shape :math:`(N_a, N_b, N_c)`, where we consider the `a`
axis to be the first, two consecutive (in the sense of storage location)
elements are in the `a` dimension, the first.
This is by opposition to the C ordering which is more widely used in
most software, and where the fastest varying element is indexed last.


Reference Spaces
================

As many other tools, SCT follows a standard nomenclature for reference
spaces in which the world or local coordinates are expressed.

The string is formed from character label among:

- `L` / `R`: left-right
- `P` / `A`: posterior-anterior
- `I` / `S`: inferior-superior

(relative to a human subject).

The character position corresponds to the axis index.


SCT uses the "from" convention, which for clarity we postfix by a
dash.

The reference space for physical coordinates is LPI- (which is coming
from nibabel and NIFTI).


An “image orientation” corresponds to the orientation of the
surface/volume with regard to the reference orientation.
It is encoded in the (NIFTI) file header.


For example, a `RAS` image orientation corresponds to a 3D image with:

- X axis oriented `L` towards `R`;
- Y axis oriented `P` towards `A`;
- Z axis oriented `I` towards `S`.


Notes:

- nibabel, BIDS
  are using the "towards" convention, ie. SCT's LPI(-) is their RAS(+).


Coordinate Conventions
======================


Local/Voxel Coordinates
+++++++++++++++++++++++

When voxel coordinates are integers, coordinates are indices.
Indices are expressed starting from 0 and up to N-1 where N is the
number of voxels in the considered dimension.

When voxel coordinates are real numbers, we are using an *integer
voxel center convention* (consistent with nibabel and NIFTI).

This means that a coordinate such as :code:`(i,j,k) == np.round((i,j,k))`
expresses the center of a voxel.

NB: Voxel coordinates are called :math:`(i,j,k)` in the NIFTI
documentation.



Global/Physical Coordinates
+++++++++++++++++++++++++++

Physical coordinates are always expressed as real numbers.
They are defined from the relation expressed by the transform and unit
system expressed in a image header.

Physical coordinates are expressed relative to the LPI- frame,
considering the voxel dimensions, affine transform between voxel
coordinates and world coordinates, and the physical dimension unit,
all of which is encoded in the NIFTI file header.

NB: Voxel coordinates are called :math:`(x,y,z)` in the NIFTI
documentation.


References
==========


- An introduction to the NIFTI file format.

  https://brainder.org/2012/09/23/the-nifti-file-format/

  See *§ Orientation information* and around.

- Official definition of the nifti1 header

  https://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h

  See *§ 3D IMAGE (VOLUME) ORIENTATION AND LOCATION IN SPACE*


- nipy/nibabel's documentation on coordinate systems

  http://nipy.org/nibabel/coordinate_systems.html#naming-reference-spaces


- ITK (ANTs, Slicer) reference coordinate system is different (LPS-)

  See https://www.slicer.org/wiki/Coordinate_systems,
  https://sourceforge.net/p/advants/discussion/840261/thread/2a1e9307/#fb5a

- Matlab FieldTrip toolbox “How are the different head and MRI coordinate systems defined?”

  http://www.fieldtriptoolbox.org/faq/how_are_the_different_head_and_mri_coordinate_systems_defined



Template / Atlas
****************

Background information on templates:

- *A Brief History of Advanced Normalization Tools (ANTs)*
  by Brian B. Avants (PENN) and Nicholas J. Tustison (UVA)

  https://stnava.github.io/ANTsTalk/#/


Templates:

- MNI-Poly-AMU - Template of the spinal cord including probabilistic
  white and gray matter.

  https://sourceforge.net/p/spinalcordtoolbox/wiki/MNI-Poly-AMU/

  .. TODO

- Spinal levels - Spinal levels of the spinal cord.

  https://sourceforge.net/p/spinalcordtoolbox/wiki/Spinal_levels/

  .. TODO

- White Matter atlas - Atlas of white matter spinal pathways.

  https://sourceforge.net/p/spinalcordtoolbox/wiki/White%20Matter%20atlas/

  .. TODO


Tips & Tricks
*************

- registration to a metric - Improve the registration of a template to
  a metric image by taking into account the spinal cord's internal
  structure

  https://sourceforge.net/p/spinalcordtoolbox/wiki/register_to_metric/

  .. TODO


- registration tricks - informations of the parameters available in the registration functions, and how to use them.

  https://sourceforge.net/p/spinalcordtoolbox/wiki/registration_tricks/

  .. TODO


Segmentation of the Spinal Cord
*******************************

SCT provides several tools to perform SC segmentation:

- :ref:`sct_propseg`
- :ref:`sct_deepseg_sc`

The latter one, using a deep learning model, is giving the best results on most
cases, but is not configurable.

The former one is the fallback tool. It has lots of options that can
be useful when segmenting tricky volumes.
You may use it if :ref:`sct_deepseg_sc` is performing worse results
than :ref:`sct_propseg` with default parameters.

.. TODO additional information, performance info, paper

Segmentation of GM/WM
*********************

SCT provides several tools to perform GM/WM segmentation:

- :ref:`sct_segment_graymatter`
- :ref:`sct_deepseg_gm`

The latter one, using a deep learning model, is giving the best results on most
cases.

The former one is the fallback tool.

.. TODO additional information, performance info, paper


Temporary Directories
*********************

Many SCT commands will create in temporary directories to operate,
and there is an option to avoid removing temporary directories, to be
used for troubleshooting purposes.

If you don't know where your temporary directory is located, you can
look at:
https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir


Matlab Integration on Mac
*************************

Matlab took the liberty of setting ``DYLD_LIBRARY_PATH`` and in order
for SCT to run, you have to run:

.. code:: matlab

   setenv('DYLD_LIBRARY_PATH', '');

Prior to running SCT commands. See
 https://github.com/neuropoly/spinalcordtoolbox/issues/405


.. _qc:

Quality Control
***************

Some SCT tools can generate Quality Control (QC) reports.
These reports consist in “appendable” HTML files, containing a table
of entries and allowing to show, for each entry, animated images
(background with overlay on and off).

To generate a QC report, add the `-qc` command-line argument,
with the location (folder, to be created by the SCT tool),
where the QC files should be generated.

