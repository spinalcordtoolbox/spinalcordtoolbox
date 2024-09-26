.. _sct_deepseg:

sct_deepseg
===========

Here we provide a gallery of each model available in the ``sct_deepseg`` CLI tool.

Spinal cord segmentation
------------------------

.. |spinalcord| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/image_contrasts.png
   :target: deepseg/spinalcord.html

.. |sc_lumbar_t2| image:: https://cdn.flintrehab.com/uploads/2020/12/lumbar-spinal-cord-injury-1.jpg
   :target: deepseg/sc_lumbar_t2.html

.. |sc_epi| image:: https://www.researchgate.net/profile/Lawrence-Tanenbaum/publication/236636641/figure/fig3/AS:720460652236801@1548782613891/Metastatic-disease-to-the-cord-FSE-T2-weighted-image-left-and-EPI-DWI-show-a_Q320.jpg
   :target: deepseg/sc_epi.html

.. |sc_MS_mp2rage| image:: https://www.ajnr.org/content/ajnr/early/2023/08/10/ajnr.A7964/F5.large.jpg?width=800&height=600&carousel=1
   :target: deepseg/sc_MS_mp2rage.html

.. |sc_mouse_t1| image:: https://www.hfsp.org/sites/default/files/webfm/Articles/Kathe2021b.jpg
   :target: deepseg/sc_mouse_t1.html


.. list-table::
   :align: center
   :widths: 25 25 25 25

   * - |spinalcord| Contrast agnostic
     - |sc_lumbar_t2| Lumbar (T2)
     - |sc_epi| EPI
     - |sc_MS_mp2rage| MS (MP2RAGE)
   * - |sc_mouse_t1| Mice (T1)
     -
     -
     -


Gray matter segmentation
------------------------

.. |gm_sc_7t_t2star| image:: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZGptCJRlgdsQCEL5T0T2Djoa1XPRWHLg51A&s
   :target: deepseg/gm_sc_7t_t2star.html

.. |gm_wm_exvivo_t2| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/gm_wm_exvivo_t2

.. |gm_wm_mouse_t1| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/gm_wm_mouse_t1

.. |gm_mouse_t1| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/gm_mouse_t1


.. list-table::
   :align: center
   :widths: 25 25 25 25

   * - |gm_sc_7t_t2star| 7T GM (T2*)
     - |gm_wm_exvivo_t2| Ex-vivo GM/WM (T2)
     - |gm_wm_mouse_t1| Mice GM/WM (T1)
     - |gm_mouse_t1| Mice GM (T1)


Tumors/lesions
--------------

.. |lesion_sc_SCI_t2| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/lesion_sc_SCI_t2.html

.. |lesion_sc_MS_stir_psir| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/lesion_sc_MS_stir_psir

.. |lesion_MS_mp2rage| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/lesion_MS_mp2rage

.. |tumor_edema_cavity_t1_t2| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/tumor_edema_cavity_t1_t2

.. |tumor_t2| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/tumor_t2


.. list-table::
   :align: center
   :widths: 25 25 25 25

   * - |lesion_sc_SCI_t2| SCI lesions (T2)
     - |lesion_sc_MS_stir_psir| MS lesions (STIR/PSIR)
     - |lesion_MS_mp2rage| MS lesions (MP2RAGE)
     - |tumor_edema_cavity_t1_t2| Tumor/edema/cavity
   * - |tumor_t2| Tumor (T2)
     -
     -
     -

Vertebrae
---------

.. |rootlets_t2| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/rootlets_t2

.. list-table::
   :align: center
   :widths: 25 25 25 25

   * - |rootlets_t2| Rootlets (T2)
     -
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
