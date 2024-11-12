.. _sct_deepseg:

sct_deepseg
===========

Here we provide a gallery of each model available in the ``sct_deepseg`` CLI tool.

Spinal cord
-----------

.. |spinalcord| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/contrast_agnostic.png
   :target: deepseg/spinalcord.html

.. |sc_lumbar_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/lumbar_t2.png
   :target: deepseg/sc_lumbar_t2.html

.. |sc_epi| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/epi_bold.png
   :target: deepseg/sc_epi.html

.. |sc_MS_mp2rage| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_sc_mp2rage.png
   :target: deepseg/sc_MS_mp2rage.html

.. |sc_mouse_t1| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/mouse_t1.png
   :target: deepseg/sc_mouse_t1.html

Spinal cord segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg -task spinalcord -i input.nii.gz

You can replace "``spinalcord``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.

.. list-table::
   :align: center
   :widths: 25 25 25

   * - |spinalcord| ``spinalcord``
     - |sc_lumbar_t2| ``sc_lumbar_t2``
     - |sc_epi| ``sc_epi``
   * - |sc_MS_mp2rage| ``sc_MS_mp2rage``
     - |sc_mouse_t1| ``sc_mouse_t1``
     -


Gray matter
-----------

.. |gm_sc_7t_t2star| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/gm_sc_7t_t2star.png
   :target: deepseg/gm_sc_7t_t2star.html

.. |gm_wm_exvivo_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/exvivo_gm_t2.png
   :target: deepseg/gm_wm_exvivo_t2.html

.. |gm_wm_mouse_t1| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/gm_wm_mouse_t1.png
   :target: deepseg/gm_wm_mouse_t1.html

.. |gm_mouse_t1| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/gm_mouse_t1.png
   :target: deepseg/gm_mouse_t1.html

Gray matter segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg -task gm_sc_7t_t2star -i input.nii.gz

You can replace "``gm_sc_7t_t2star``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.

.. list-table::
   :align: center
   :widths: 25 25 25

   * - |gm_sc_7t_t2star| ``gm_sc_7t_t2star``
     - |gm_wm_exvivo_t2| ``gm_wm_exvivo_t2``
     - |gm_wm_mouse_t1| ``gm_wm_mouse_t1``
   * - |gm_mouse_t1| ``gm_mouse_t1``
     -
     -


Pathologies
-----------

.. |lesion_sc_SCI_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/sci_lesion_sc_t2.png
   :target: deepseg/lesion_sc_SCI_t2.html

.. |lesion_sc_MS_stir_psir| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_lesion_sc_stir_psir.png
   :target: deepseg/lesion_sc_MS_stir_psir.html

.. |lesion_MS_mp2rage| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/ms_lesion_mp2rage.png
   :target: deepseg/lesion_MS_mp2rage.html

.. |tumor_edema_cavity_t1_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/tumor_edema_cavity_t1_t2.png
   :target: deepseg/tumor_edema_cavity_t1_t2.html

.. |tumor_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/tumor_t2.png
   :target: deepseg/tumor_t2.html

Pathology segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg -task lesion_sc_SCI_t2 -i input.nii.gz

You can replace "``lesion_sc_SCI_t2``" with any of the task names in the table below to perform different tasks. Click on a task below for more information.


.. list-table::
   :align: center
   :widths: 25 25 25

   * - |lesion_sc_SCI_t2| ``lesion_sc_SCI_t2``
     - |lesion_sc_MS_stir_psir| ``lesion_sc_MS_stir_psir``
     - |lesion_MS_mp2rage| ``lesion_MS_mp2rage``
   * - |tumor_edema_cavity_t1_t2| ``tumor_edema_cavity_t1_t2``
     - |tumor_t2| ``tumor_t2``
     -


Other structures
----------------

.. |rootlets_t2| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/sct_deepseg/spinal_rootlets_t2.png
   :target: deepseg/rootlets_t2.html

Rootlets segmentation can be performed by running the following sample command:

.. code::

   sct_deepseg -task rootlets_t2 -i input.nii.gz


.. list-table::
   :align: center
   :widths: 25 25 25

   * - |rootlets_t2| ``rootlets_t2``
     -
     -


.. toctree::
   :hidden:
   :maxdepth: 2

   deepseg/spinalcord
   deepseg/sc_MS_mp2rage
   deepseg/sc_epi
   deepseg/sc_mouse_t1
   deepseg/sc_lumbar_t2
   deepseg/gm_wm_exvivo_t2
   deepseg/gm_sc_7t_t2star
   deepseg/gm_mouse_t1
   deepseg/gm_wm_mouse_t1
   deepseg/lesion_sc_SCI_t2
   deepseg/lesion_sc_MS_stir_psir
   deepseg/lesion_MS_mp2rage
   deepseg/tumor_edema_cavity_t1_t2
   deepseg/tumor_t2
   deepseg/rootlets_t2
