.. _sct_deepseg:

sct_deepseg
===========

Here we provide a gallery of each model available in the ``sct_deepseg`` CLI tool.

Spinal cord
-----------

.. |spinalcord| image:: https://i.imgur.com/1QXBTWf.png
   :target: deepseg/spinalcord.html

.. |lumbar_sc_t2w| image:: https://i.imgur.com/aVJNqp1.png
   :target: deepseg/lumbar_sc_t2w.html

.. |sc_epi| image:: https://i.imgur.com/zj9idbJ.png
   :target: deepseg/sc_epi.html

.. |ms_sc_mp2rage| image:: https://i.imgur.com/cqsrFWF.png
   :target: deepseg/ms_sc_mp2rage.html

.. |mice_sc| image:: https://i.imgur.com/MaajGC2.png
   :target: deepseg/mice_sc.html

Spinal cord segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg -task spinalcord -i input.nii.gz

You can replace "``spinalcord``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.

.. list-table::
   :align: center
   :widths: 25 25 25

   * - |spinalcord| ``spinalcord``
     - |lumbar_sc_t2w| ``lumbar_sc_t2w``
     - |sc_epi| ``sc_epi``
   * - |ms_sc_mp2rage| ``ms_sc_mp2rage``
     - |mice_sc| ``mice_sc``
     -


Gray matter
-----------

.. |gm_sc_7t_t2star| image:: https://i.imgur.com/EQ6cEsv.png
   :target: deepseg/gm_sc_7t_t2star.html

.. |exvivo_gm_wm_t2| image:: https://i.imgur.com/m4wB6Lk.png
   :target: deepseg/exvivo_gm_wm_t2.html

.. |mouse_gm_wm_t1w| image:: https://i.imgur.com/BMAHSD0.png
   :target: deepseg/mouse_gm_wm_t1w.html

.. |mice_gm| image:: https://i.imgur.com/oooqyjh.png
   :target: deepseg/mice_gm.html

Gray matter segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg -task gm_sc_7t_t2star -i input.nii.gz

You can replace "``gm_sc_7t_t2star``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.

.. list-table::
   :align: center
   :widths: 25 25 25

   * - |gm_sc_7t_t2star| ``gm_sc_7t_t2star``
     - |exvivo_gm_wm_t2| ``exvivo_gm_wm_t2``
     - |mouse_gm_wm_t1w| ``mouse_gm_wm_t1w``
   * - |mice_gm| ``mice_gm``
     -
     -


Pathologies
-----------

.. |sc_lesion_t2w_sci| image:: https://i.imgur.com/fZncKDj.png
   :target: deepseg/sc_lesion_t2w_sci.html

.. |sc_ms_lesion_stir_psir| image:: https://i.imgur.com/1U8LgQ0.png
   :target: deepseg/sc_ms_lesion_stir_psir.html

.. |ms_lesion_mp2rage| image:: https://i.imgur.com/1mP1IYt.png
   :target: deepseg/ms_lesion_mp2rage.html

.. |tumor_edema_cavity_t1_t2| image:: https://i.imgur.com/dFq8gkq.png
   :target: deepseg/tumor_edema_cavity_t1_t2.html

.. |tumor_t2| image:: https://i.imgur.com/CbYVizW.png
   :target: deepseg/tumor_t2.html

Pathology segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg -task sc_lesion_t2w_sci -i input.nii.gz

You can replace "``sc_lesion_t2w_sci``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.


.. list-table::
   :align: center
   :widths: 25 25 25

   * - |sc_lesion_t2w_sci| ``sc_lesion_t2w_sci``
     - |sc_ms_lesion_stir_psir| ``sc_ms_lesion_stir_psir``
     - |ms_lesion_mp2rage| ``ms_lesion_mp2rage``
   * - |tumor_edema_cavity_t1_t2| ``tumor_edema_cavity_t1_t2``
     - |tumor_t2| ``tumor_t2``
     -


Other structures
----------------

.. |spinal_rootlets_t2w| image:: https://i.imgur.com/bQBFKVs.png
   :target: deepseg/spinal_rootlets_t2w.html

Rootlets segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg -task spinal_rootlets_t2w -i input.nii.gz


.. list-table::
   :align: center
   :widths: 25 25 25

   * - |spinal_rootlets_t2w| ``spinal_rootlets_t2w``
     -
     -


.. toctree::
   :hidden:
   :maxdepth: 2

   deepseg/spinalcord
   deepseg/ms_sc_mp2rage
   deepseg/sc_epi
   deepseg/mice_sc
   deepseg/lumbar_sc_t2w
   deepseg/exvivo_gm_wm_t2
   deepseg/gm_sc_7t_t2star
   deepseg/mice_gm
   deepseg/mouse_gm_wm_t1w
   deepseg/sc_lesion_t2w_sci
   deepseg/sc_ms_lesion_stir_psir
   deepseg/ms_lesion_mp2rage
   deepseg/tumor_edema_cavity_t1_t2
   deepseg/tumor_t2
   deepseg/spinal_rootlets_t2w
