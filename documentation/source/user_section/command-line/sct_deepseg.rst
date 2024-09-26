.. _sct_deepseg:

sct_deepseg
===========

Here we provide a gallery of each model available in the ``sct_deepseg`` CLI tool.

Spinal cord segmentation
------------------------

.. |spinalcord| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/image_contrasts.png
   :target: deepseg/spinalcord.html

.. |lumbar_sc_t2w| image:: https://cdn.flintrehab.com/uploads/2020/12/lumbar-spinal-cord-injury-1.jpg
   :target: deepseg/lumbar_sc_t2w.html

.. |sc_epi| image:: https://www.researchgate.net/profile/Lawrence-Tanenbaum/publication/236636641/figure/fig3/AS:720460652236801@1548782613891/Metastatic-disease-to-the-cord-FSE-T2-weighted-image-left-and-EPI-DWI-show-a_Q320.jpg
   :target: deepseg/sc_epi.html

.. |ms_sc_mp2rage| image:: https://www.ajnr.org/content/ajnr/early/2023/08/10/ajnr.A7964/F5.large.jpg?width=800&height=600&carousel=1
   :target: deepseg/ms_sc_mp2rage.html

.. |mice_sc| image:: https://www.hfsp.org/sites/default/files/webfm/Articles/Kathe2021b.jpg
   :target: deepseg/mice_sc.html

.. |gm_sc_7t_t2star| image:: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZGptCJRlgdsQCEL5T0T2Djoa1XPRWHLg51A&s
   :target: deepseg/gm_sc_7t_t2star.html

.. |sc_lesion_t2w_sci| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/sc_lesion_t2w_sci.html

.. list-table::
   :align: center
   :widths: 25 25 25 25

   * - |spinalcord| Contrast agnostic
     - |lumbar_sc_t2w| Lumbar (T2)
     - |sc_epi| EPI
     - |sc_lesion_t2w_sci| SCI (T2)
   * - |gm_sc_7t_t2star| 7T (T2*)
     - |ms_sc_mp2rage| MS (MP2RAGE)
     - |mice_sc| Mice
     -


Gray matter segmentation
------------------------

.. |mice_gm| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/mice_gm

.. |exvivo_gm_wm_t2| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/exvivo_gm_wm_t2

.. |mouse_gm_wm_t1w| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/mouse_gm_wm_t1w

.. list-table::
   :align: center
   :widths: 25 25 25 25

   * - |exvivo_gm_wm_t2| Ex-vivo GM/WM (T2)
     - |mice_gm| Mice GM
     - |mouse_gm_wm_t1w| Mice GM/WM
     -


Tumors/lesions
--------------

.. |tumor_t2| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/tumor_t2

.. |tumor_edema_cavity_t1_t2| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/tumor_edema_cavity_t1_t2

.. |ms_lesion_mp2rage| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/ms_lesion_mp2rage

.. |sc_ms_lesion_stir_psir| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/sc_ms_lesion_stir_psir

.. list-table::
   :align: center
   :widths: 25 25 25 25

   * - |ms_lesion_mp2rage| MS lesions (MP2RAGE)
     - |sc_ms_lesion_stir_psir| MS lesions (STIR/PSIR)
     - |tumor_edema_cavity_t1_t2| Tumor/edema/cavity
     - |tumor_t2| Tumor (T2)

Vertebrae
---------

.. |spinal_rootlets_t2w| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/spinal_rootlets_t2w

.. list-table::
   :align: center
   :widths: 25 25 25 25

   * - |spinal_rootlets_t2w| Rootlets (T2)
     -
     -
     -

.. toctree::
   :hidden:
   :maxdepth: 2

   deepseg/exvivo_gm_wm_t2
   deepseg/gm_sc_7t_t2star
   deepseg/lumbar_sc_t2w
   deepseg/mice_gm
   deepseg/mice_sc
   deepseg/mouse_gm_wm_t1w
   deepseg/ms_lesion_mp2rage
   deepseg/ms_sc_mp2rage
   deepseg/spinalcord
   deepseg/sc_epi
   deepseg/sc_lesion_t2w_sci
   deepseg/sc_ms_lesion_stir_psir
   deepseg/spinal_rootlets_t2w
   deepseg/tumor_edema_cavity_t1_t2
   deepseg/tumor_t2