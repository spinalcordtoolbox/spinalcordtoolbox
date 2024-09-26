.. _sct_deepseg:

sct_deepseg
===========

Here we provide a gallery of each model available in the ``sct_deepseg`` CLI tool.

Spinal cord
-----------

.. |seg_sc_contrast_agnostic| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/contrast_agnostic.png
   :target: deepseg/seg_sc_contrast_agnostic.html

.. |seg_lumbar_sc_t2w| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/lumbar_t2.png
   :target: deepseg/seg_lumbar_sc_t2w.html

.. |seg_sc_epi| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/epi_bold.png
   :target: deepseg/seg_sc_epi.html

.. |seg_sc_t2star| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/sc_t2star.png
    :target: deepseg/seg_sc_t2star.html

.. |mice_sc| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/mouse_t1.png
   :target: deepseg/mice_sc.html

Spinal cord segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg seg_sc_contrast_agnostic -i input.nii.gz

You can replace "``seg_sc_contrast_agnostic``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.

.. list-table::
   :align: center
   :widths: 25 25 25

   * - |seg_sc_contrast_agnostic| ``seg_sc_contrast_agnostic``
     - |seg_lumbar_sc_t2w| ``seg_lumbar_sc_t2w``
     - |seg_sc_epi| ``seg_sc_epi``
   * - |seg_sc_t2star| ``seg_sc_t2star``
     - |mice_sc| ``mice_sc``
     -

Gray matter
-----------

.. |seg_gm_sc_7t_t2star| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/gm_sc_7t_t2star.png
   :target: deepseg/seg_gm_sc_7t_t2star.html

.. |seg_exvivo_gm_wm_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/exvivo_gm_t2.png
   :target: deepseg/seg_exvivo_gm_wm_t2.html

.. |seg_mouse_gm_wm_t1w| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/gm_wm_mouse_t1.png
   :target: deepseg/seg_mouse_gm_wm_t1w.html

.. |mice_gm| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/gm_mouse_t1.png
   :target: deepseg/seg_mice_gm.html

Gray matter segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg seg_gm_sc_7t_t2star -i input.nii.gz

You can replace "``seg_gm_sc_7t_t2star``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.

.. list-table::
   :align: center
   :widths: 25 25 25

   * - |seg_gm_sc_7t_t2star| ``seg_gm_sc_7t_t2star``
     - |seg_exvivo_gm_wm_t2| ``seg_exvivo_gm_wm_t2``
     - |seg_mouse_gm_wm_t1w| ``seg_mouse_gm_wm_t1w``
   * - |mice_gm| ``mice_gm``
     -
     -


Pathologies
-----------

.. |seg_sc_lesion_t2w_sci| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/sci_lesion_sc_t2.png
   :target: deepseg/seg_sc_lesion_t2w_sci.html

.. |seg_ms_lesion_mp2rage| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_lesion_mp2rage.png
   :target: deepseg/seg_ms_lesion_mp2rage.html

.. |seg_sc_ms_lesion_axial_t2w| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_lesion_sc_axial_t2.png
   :target: deepseg/seg_sc_ms_lesion_axial_t2w.html

.. |seg_ms_lesion| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_lesion.png
   :target: deepseg/seg_ms_lesion.html

.. |seg_tumor_edema_cavity_t1_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/tumor_edema_cavity_t1_t2.png
   :target: deepseg/seg_tumor_edema_cavity_t1_t2.html

.. |tumor_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/tumor_t2.png
   :target: deepseg/seg_tumor_t2.html

Pathology segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg seg_sc_lesion_t2w_sci -i input.nii.gz

You can replace "``seg_sc_lesion_t2w_sci``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.


.. list-table::
   :align: center
   :widths: 25 25 25

   * - |seg_sc_lesion_t2w_sci| ``seg_sc_lesion_t2w_sci``
     - |seg_ms_lesion| ``seg_ms_lesion``
     - |seg_sc_ms_lesion_axial_t2w| ``seg_sc_ms_lesion_axial_t2w``
   * - |seg_ms_lesion_mp2rage| ``seg_ms_lesion_mp2rage``
     - |seg_tumor_edema_cavity_t1_t2| ``seg_tumor_edema_cavity_t1_t2``
     - |tumor_t2| ``tumor_t2``


Other structures
----------------

.. |seg_spinal_rootlets_t2w| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/spinal_rootlets_t2.png
   :target: deepseg/seg_spinal_rootlets_t2w.html

.. |totalspineseg| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/totalspineseg.png
   :target: deepseg/totalspineseg.html

Multiple structures may be segmented by running the following sample command:

.. code::

   sct_deepseg seg_spinal_rootlets_t2w -i input.nii.gz

You can replace "``seg_spinal_rootlets_t2w``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.


.. list-table::
   :align: center
   :widths: 25 25 25

   * - |seg_spinal_rootlets_t2w| ``seg_spinal_rootlets_t2w``
     - |totalspineseg| ``totalspineseg``
     -

.. |canal_t2w| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/spinal_canal_t2.png
   :target: deepseg/canal_t2w.html

Spinal Canal segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg canal_t2w -i input.nii.gz


.. list-table::
   :align: center
   :widths: 25 25 25

   * - |canal_t2w| ``canal_t2w``
     - 
     -


Retired models
--------------

.. |seg_sc_ms_lesion_stir_psir| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_lesion_sc_stir_psir.png
   :target: deepseg/seg_sc_ms_lesion_stir_psir.html

.. |ms_sc_mp2rage| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_sc_mp2rage.png
   :target: deepseg/seg_ms_sc_mp2rage.html

These models have been replaced by newer, more advanced models. We recommend switching to the model listed in the table below.

If you absolutely require these models, you can downgrade to version of SCT listed in the table below. If you do this, please let us know on the SCT Forum so we can better understand your use-case, and potentially reinstate the model if necessary.

.. list-table::
   :align: center
   :widths: 33 33 33

   * - Model
     - Last available
     - Superseded by
   * - |seg_sc_ms_lesion_stir_psir| ``seg_sc_ms_lesion_stir_psir``
     - SCT Version ``6.4``
     - ``seg_ms_lesion`` (contrast-agnostic MS lesion segmentation)
   * - |ms_sc_mp2rage| ``ms_sc_mp2rage``
     - SCT Version ``6.4``
     - ``seg_sc_contrast_agnostic`` (contrast-agnostic SC segmentation)

.. toctree::
   :hidden:
   :maxdepth: 2

   deepseg/seg_sc_contrast_agnostic
   deepseg/seg_ms_sc_mp2rage
   deepseg/seg_sc_epi
   deepseg/seg_sc_t2star
   deepseg/seg_mice_sc
   deepseg/seg_lumbar_sc_t2w
   deepseg/seg_exvivo_gm_wm_t2
   deepseg/seg_gm_sc_7t_t2star
   deepseg/seg_mice_gm
   deepseg/seg_mouse_gm_wm_t1w
   deepseg/seg_sc_lesion_t2w_sci
   deepseg/seg_sc_ms_lesion_stir_psir
   deepseg/seg_sc_ms_lesion_axial_t2w
   deepseg/seg_ms_lesion_mp2rage
   deepseg/seg_ms_lesion
   deepseg/seg_tumor_edema_cavity_t1_t2
   deepseg/seg_tumor_t2
   deepseg/seg_spinal_rootlets_t2w
   deepseg/canal_t2w
   deepseg/totalspineseg
