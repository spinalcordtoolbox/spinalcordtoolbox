.. _pam50:

PAM50 Template
**************

Introduction
============

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/pam50/pam50_template.png
    :figwidth: 40%
    :align: right

Template-based analysis of multi-parametric MRI data of the spinal cord sets
the foundation for standardization and reproducibility. Particularly, it
allows researchers to process their data with minimum bias, perform
multi-center studies, with the goal of improving patient diagnosis and
prognosis and helping the discovery of new biomarkers of spinal-related
diseases.

PAM50 is one such template for MRI of the full spinal cord and brainstem, and
is included in your installation of SCT. It has the following features:

* It is available for T1-, T2-and T2*-weighted MRI contrast.
* It is compatible with the ICBM152 brain template, allowing cerebrospinal studies.
* It includes atlases of white matter pathways and gray matter subregions.


Template (``PAM50/template``)
=============================

The template folder contains image files representing different template diffusion weightings, binary/probablistic masks (for the spinal cord, WM/GM tracts, etc.), and point-wise labels for vertebral levels and intervertebral discs.

The template folder also contains an ``info_label.txt`` file to explain what each file represents:

.. include:: info_label-template.txt
   :code:


White and grey matter atlas (``PAM50/atlas``)
=============================================

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/pam50/white_matter_atlas_illustration.png
    :figwidth: 40%
    :align: right

The White Matter atlas will be a useful tool for your studies of specific spinal cord tracts. It consists of 36 nifti
volumes named ``PAM50_atlas_<tract_number>.nii.gz`` where ``<tract_number>`` is the number identifying the tract.
Fifteen WM tracts and three GM regions are available for each side. The values of each voxel of the files ``PAM50_atlas_<tract_number>.nii.gz``
are the voxel volume proportions occupied by the corresponding tract.

The atlas folder also contains an ``info_label.txt`` file to explain what each file represents:

.. include:: info_label-atlas.txt
   :code:


Spinal levels (``PAM50/spinal_levels``)
=======================================

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/pam50/spinal_levels_illustration.png
    :figwidth: 40%
    :align: right

In the folder ``data/PAM50/spinal_levels``, you will find 20 label images corresponding to different spinal cord levels, including both C1:C8 and T1:T12. In each nifti file, the value of each voxel is the probability for this voxel to belong to the
spinal level.

The spinal_levels folder also contains an ``info_label.txt`` file to explain what each file represents:

.. include:: info_label-spinal_levels.txt
   :code:


References
==========

* `De Leener B, Fonov V, Collins DL, Callot V, Stikov N, Cohen-Adad J. PAM50: Multimodal template of the brainstem and spinal cord compatible with the ICBM152 space. Proceedings of the 25th Annual Meeting of ISMRM, Honolulu, USA. 2017. <https://www.sciencedirect.com/science/article/abs/pii/S1053811917308686>`_.
* `Benhamou M, Fonov V, Taso M, Le Troter A, Sdika M, Collins DL, Callot V, Cohen-Adad J. Atlas of white-matter tracts in the human spinal cord. Proceedings of the 22th Annual Meeting of ISMRM, Milan, Italy 2014:0013 <https://dl.dropboxusercontent.com/u/20592661/publications/benhamou_irmsm14.pdf>`_.
* `Cadotte DW, Cadotte A, Cohen-Adad J, Fleet D, Livne M, Mikulis D, Fehlings MG. Resolving the anatomic variability of the human cervical spinal cord: a solution to facilitate advanced neural imaging. Proceedings of the 22th Annual Meeting of ISMRM, Milan, Italy 2014:1719 <https://dl.dropboxusercontent.com/u/20592661/publications/cadotte_ismrm14.pdf>`_.

Additionally, the template was generated using the following tools:

* `neuropoly/template GitHub repository <https://github.com/neuropoly/template>`_
* `White matter atlas script <https://github.com/neuropoly/spinalcordtoolbox/tree/master/dev/atlas>`_
* `Spinal levels script <https://github.com/neuropoly/spinalcordtoolbox/tree/master/dev/spinal_level>`_
