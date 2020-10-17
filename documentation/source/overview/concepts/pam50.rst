.. _pam50:

PAM50 Template
**************

Introduction
============

Template-based analysis of multi-parametric MRI data of the spinal cord sets
the foundation for standardization and reproducibility. Particularly, it
allows researchers to process their data with minimum bias, perform
multi-center studies, with the goal of improving patient diagnosis and
prognosis and helping the discovery of new biomarkers of spinal-related
diseases.

PAM50 is an MRI template of the full spinal cord and brainstem. It possesses
the following features:

  * It is available for T1-, T2-and T2*-weighted MRI contrast.
  * It is compatible with the ICBM152 brain template, allowing cerebrospinal studies.
  * It includes atlases of white matter pathways and gray matter subregions.

The PAM50 template is provided in the `data/PAM50` template. Included are the
following files and directories:
  * `PAM50/atlas`: Contains label images for the white matter atlas. These labels include WM and GM regions, a CSF contour label, and combined labels that group individual labels into categories (e.g. dorsal columns, funiculi, etc.).
  * `PAM50/spinal_levels`: Contains 20 label images corresponding to different spinal cord levels, including both C1:C8 and T1:T12.
  * `PAM50/template`: Contains image files representing different template diffusion weightings, binary/probablistic masks (for the spinal cord, WM/GM tracts, etc.), and point-wise labels for vertebral levels and intervertebral discs.
  * `PAM50/CHANGES.md`: A changelog for the template, including changes made after publication.

For further context, the PAM50 template is used extensively in Spinal Cord Toolbox scripts:

  * `sct_register_to_template`: Registers an anatomical image to the spinal cord MRI template. PAM50 is the default template to register images to.
  * `sct_analyze_legion`: Computes statistics on segmented lesions. PAM50 template is used to compute lesion volume across vertebral levels, GM, WM, within WM/GM tracts, etc.
  * `sct_label_vertebrae`: Takes an anatomical image and its cord segmentation and outputs the cord segmentation labeled with vertebral levels. PAM50 vertebral levels are used to detect vertebral levels via template matching.
  * `sct_process_segmentation`: Computes morphometric measures based on spinal cord segmentation. Vertebral levels from the PAM50 template are used by default when aggregating measures across vertebral levels.
  * `sct_extract_metric`: Extracts metrics (e.g., DTI or MTR) within labels. The label file from the PAM50 white matter atlas is used by default.

The PAM50 template was generated using a framework for creating unbiased MRI templates of the spinal cord. This framework can be found at the `neurpoly/template GitHub repository <https://github.com/neuropoly/template>`_


References
++++++++++

`De Leener B, Fonov V, Collins DL, Callot V, Stikov N, Cohen-Adad J. PAM50:
Multimodal template of the brainstem and spinal cord compatible with the
ICBM152 space. Proceedings of the 25th Annual Meeting of ISMRM, Honolulu, USA.
2017.
<https://www.sciencedirect.com/science/article/abs/pii/S1053811917308686>`_


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