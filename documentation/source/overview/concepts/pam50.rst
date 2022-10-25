.. _pam50:

PAM50 Template
**************

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/pam50/pam50_template.png
    :figwidth: 40%
    :align: right

Introduction
============

Template-based analysis of multi-parametric MRI data of the spinal cord sets the foundation for standardization and reproducibility. Particularly, it allows researchers to process their data with minimum bias and perform multi-center studies, with the goal of improving patient diagnosis and prognosis and discovering new biomarkers of spinal-related diseases.

To perform template-based analysis, a template is needed. Templates act as a common reference space to realign each subject to. PAM50 is one such template for MRI of the full spinal cord and brainstem, and is included in your installation of SCT. "PAM" stands for “Polytechnique”, “Aix-Marseille University” and “Montreal Neurological Institute”, the three institutions that collaborated to develop the template. The PAM50 template has the following features:

* It is available for T1-, T2-and T2*-weighted MRI contrast.
* It is compatible with the ICBM152 brain template (a.k.a. the MNI coordinate system), allowing researchers to conduct simultaneous brain/spine studies within the same coordinate system.
* It includes a probabilistic atlas for white matter, gray metter, and CSF regions.
* It is a straightened template rather than a curved one.

The template is stored in the ``data/PAM50/`` directory in your SCT installation directory. ``PAM50/`` contains several subfolders, each corresponding to different aspects of PAM50.

.. _pam50-template-section:

Template (``PAM50/template``)
=============================

The template folder contains the PAM50 template for various MRI contrast types. It also contains binary and probablistic masks for different spinal features, as well as point-wise labels for vertebral levels and intervertebral discs.

The template folder also contains an ``info_label.txt`` file to explain what each file represents:

.. include:: info_label-template.txt
   :code:

Please note that the white and gray matter masks found in ``template/`` are for the entire structures. If you are looking for individual tracts, the relevant volumes are provided within the :ref:`pam50-atlas-section` section below.


.. _pam50-atlas-section:

White and Grey Matter Atlas (``PAM50/atlas``)
=============================================

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/pam50/white_matter_atlas_illustration.png
    :align: right

The White Matter atlas will be a useful tool for your studies of specific spinal cord tracts. It consists of 36 NIfTI volumes named ``PAM50_atlas_<tract_number>.nii.gz`` where ``<tract_number>`` is the number identifying the tract. Fifteen WM tracts and three GM regions are available for each side. The values of each voxel of the files ``PAM50_atlas_<tract_number>.nii.gz`` are the voxel volume proportions occupied by the corresponding tract.

Notably, each volume is a probablistic mask, or "soft mask". The voxel values in these volumes range from 0 to 1; voxels in the center of each tract will equal 1, while voxels near the edges of each tract will be weighted closer to 0. This is to account for `partial volume effects <http://mriquestions.com/partial-volume-effects.html>`_.

The atlas folder also contains an ``info_label.txt`` file to explain what each file represents:

.. include:: info_label-atlas.txt
   :code:


.. _pam50-spinal-levels-section:

Spinal levels (``PAM50/spinal_levels``)
=======================================

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/pam50/spinal_levels_illustration.png
    :figwidth: 40%
    :align: right

In the folder ``data/PAM50/spinal_levels``, you will find 20 label images corresponding to different spinal cord levels, spanning both C1:C8 and T1:T12. In each nifti file, the value of each voxel is the probability for this voxel to belong to the spinal level.

The spinal_levels folder also contains an ``info_label.txt`` file to explain what each file represents:

.. include:: info_label-spinal_levels.txt
   :code:


References
==========

* `De Leener B, Fonov VS, Collins DL, Callot V, Stikov N, Cohen-Adad J. PAM50: Unbiased multimodal template of the brainstem and spinal cord aligned with the ICBM152 space. Neuroimage, 2018, 165:170-179. <https://paperpile.com/app/p/e74eced0-7d51-08e2-9c18-a34328fb4a86>`_
* `Levy S, Benhamou M, Naaman C, Rainville P, Callot V, Cohen-Adad J. White matter atlas of the human spinal cord with estimation of partial volume effect. Neuroimage, 2015, 119:262-271. <https://paperpile.com/app/p/e74eced0-7d51-08e2-9c18-a34328fb4a86>`_
* `Cadotte DW, Cadotte A, Cohen-Adad J, Fleet D, Livne M, Wilson JR, Mikulis D, Nugaeva N, Fehlings MG. Characterizing the location of spinal and vertebral levels in the human cervical spinal cord. AJNR Am J Neuroradiol, 2015, 36(4):803-810. <https://paperpile.com/app/p/5b580317-6921-06c8-a2ee-685d4dbaa44c>`_

Additionally, the template was generated using the following tools:

* `neuropoly/template GitHub repository <https://github.com/neuropoly/template>`_
* `White matter atlas script <https://github.com/spinalcordtoolbox/spinalcordtoolbox-dev/tree/master/dev/atlas>`_
* `Spinal levels script <https://github.com/spinalcordtoolbox/spinalcordtoolbox-dev/tree/master/dev/spinal_level>`_
