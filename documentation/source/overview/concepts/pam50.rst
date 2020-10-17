.. _pam50:

PAM50 Template
**************

.. warning:: TODO: Write a PAM50 template guide that is an analog to the older, outdated guide found here: https://sourceforge.net/p/spinalcordtoolbox/wiki/MNI-Poly-AMU/

Background information on templates:

- `*A Brief History of Advanced Normalization Tools (ANTs)*
  by Brian B. Avants (PENN) and Nicholas J. Tustison (UVA)
  <https://stnava.github.io/ANTsTalk/#/>`_

Here is the relevant section of the SCT repository which generated this template:
`PAM50 anatomical template <https://github.com/neuropoly/template>`_



White matter atlas
==================

.. image:: ../../imgs/white_matter_atlas_illustration.png

The White Matter atlas will be a useful tool for your studies of specific spinal cord tracts. It consists of 30 nifti
volumes named ``PAM50_atlas_<tract_number>.nii.gz`` where ``<tract_number>`` is the number identifying the tract.
Fifteen tracts for each side are available. The values of each voxel of the files ``PAM50_atlas_<tract_number>.nii.gz``
are the voxel volume proportions occupied by the corresponding tract.

The atlas is stored in the ``data/PAM50/atlas`` folder, and comes with a text file (``info_label.txt``) that provides a
more detailed breakdown of each tract.

As well, here is the relevant section of the SCT repository which generated this template:
`White matter atlas <https://github.com/neuropoly/spinalcordtoolbox/tree/master/dev/atlas>`_

References
++++++++++

`Benhamou M, Fonov V, Taso M, Le Troter A, Sdika M, Collins DL, Callot V, Cohen-Adad J. Atlas of
white-matter tracts in the human spinal cord. Proceedings of the 22th Annual Meeting of ISMRM, Milan, Italy 2014:0013
<https://dl.dropboxusercontent.com/u/20592661/publications/benhamou_irmsm14.pdf>`_.



Spinal levels
=============

.. image:: ../../imgs/spinal_levels_illustration.png

In the folder ``data/PAM50/spinal_levels``, you will find 11 nifti files, each one corresponding to one spinal level of
the spinal cord from C3 to T5. On the illustration above, you can see the spinal levels C4 (red-yellow) and T1
(blue-lightblue). In each nifti file, the value of each voxel is the probability for this voxel to belong to the
spinal level.

As well, here is the relevant section of the SCT repository which generated this template:
`Spinal levels <https://github.com/neuropoly/spinalcordtoolbox/tree/master/dev/spinal_level>`_

References
++++++++++

`Cadotte DW, Cadotte A, Cohen-Adad J, Fleet D, Livne M, Mikulis D, Fehlings MG. Resolving the anatomic
variability of the human cervical spinal cord: a solution to facilitate advanced neural imaging. Proceedings of the
22th Annual Meeting of ISMRM, Milan, Italy 2014:1719
<https://dl.dropboxusercontent.com/u/20592661/publications/cadotte_ismrm14.pdf>`_.