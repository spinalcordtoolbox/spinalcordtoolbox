.. _sct_deepseg:

sct_deepseg
===========

Here we provide a gallery of each model available in the ``sct_deepseg`` CLI tool.

Spinal cord segmentation
------------------------

.. |seg_sc_contrast_agnostic| image:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/image_contrasts.png
   :target: deepseg/seg_sc_contrast_agnostic.html

.. |seg_lumbar_sc_t2w| image:: https://cdn.flintrehab.com/uploads/2020/12/lumbar-spinal-cord-injury-1.jpg
   :target: deepseg/seg_lumbar_sc_t2w.html

.. |seg_sc_epi| image:: https://www.researchgate.net/profile/Lawrence-Tanenbaum/publication/236636641/figure/fig3/AS:720460652236801@1548782613891/Metastatic-disease-to-the-cord-FSE-T2-weighted-image-left-and-EPI-DWI-show-a_Q320.jpg
   :target: deepseg/seg_sc_epi.html

.. |ms_sc_mp2rage| image:: https://www.ajnr.org/content/ajnr/early/2023/08/10/ajnr.A7964/F5.large.jpg?width=800&height=600&carousel=1
   :target: deepseg/seg_ms_sc_mp2rage.html

.. |mice_sc| image:: https://www.hfsp.org/sites/default/files/webfm/Articles/Kathe2021b.jpg
   :target: deepseg/mice_sc.html

.. |seg_gm_sc_7t_t2star| image:: https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRZGptCJRlgdsQCEL5T0T2Djoa1XPRWHLg51A&s
   :target: deepseg/seg_gm_sc_7t_t2star.html

.. |seg_sc_lesion_t2w_sci| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/seg_sc_lesion_t2w_sci.html

.. list-table::
   :align: center
   :widths: 25 25 25 25

   * - |seg_sc_contrast_agnostic| Contrast agnostic
     - |seg_lumbar_sc_t2w| Lumbar (T2)
     - |seg_sc_epi| EPI
     - |seg_sc_lesion_t2w_sci| SCI (T2)
   * - |seg_gm_sc_7t_t2star| 7T (T2*)
     - |ms_sc_mp2rage| MS (MP2RAGE)
     - |mice_sc| Mice
     -


Gray matter segmentation
------------------------

.. |mice_gm| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/seg_mice_gm

.. |seg_exvivo_gm_wm_t2| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/seg_exvivo_gm_wm_t2

.. |seg_mouse_gm_wm_t1w| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/seg_mouse_gm_wm_t1w

.. list-table::
   :align: center
   :widths: 25 25 25 25

   * - |seg_exvivo_gm_wm_t2| Ex-vivo GM/WM (T2)
     - |mice_gm| Mice GM
     - |seg_mouse_gm_wm_t1w| Mice GM/WM
     -


Tumors/lesions
--------------

.. |tumor_t2| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/seg_tumor_t2

.. |seg_tumor_edema_cavity_t1_t2| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/seg_tumor_edema_cavity_t1_t2

.. |seg_ms_lesion_mp2rage| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/seg_ms_lesion_mp2rage

.. |seg_sc_ms_lesion_stir_psir| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/seg_sc_ms_lesion_stir_psir

.. list-table::
   :align: center
   :widths: 25 25 25 25

   * - |seg_ms_lesion_mp2rage| MS lesions (MP2RAGE)
     - |seg_sc_ms_lesion_stir_psir| MS lesions (STIR/PSIR)
     - |seg_tumor_edema_cavity_t1_t2| Tumor/edema/cavity
     - |tumor_t2| Tumor (T2)

Vertebrae
---------

.. |seg_spinal_rootlets_t2w| image:: https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fsc.2011.107/MediaObjects/41393_2012_Article_BFsc2011107_Fig10_HTML.jpg
   :target: deepseg/seg_spinal_rootlets_t2w

.. list-table::
   :align: center
   :widths: 25 25 25 25

   * - |seg_spinal_rootlets_t2w| Rootlets (T2)
     -
     -
     -

.. toctree::
   :hidden:
   :maxdepth: 2

   deepseg/seg_exvivo_gm_wm_t2
   deepseg/seg_gm_sc_7t_t2star
   deepseg/seg_lumbar_sc_t2w
   deepseg/seg_mice_gm
   deepseg/seg_mice_sc
   deepseg/seg_mouse_gm_wm_t1w
   deepseg/seg_ms_lesion_mp2rage
   deepseg/seg_ms_sc_mp2rage
   deepseg/seg_sc_contrast_agnostic
   deepseg/seg_sc_epi
   deepseg/seg_sc_lesion_t2w_sci
   deepseg/seg_sc_ms_lesion_stir_psir
   deepseg/seg_spinal_rootlets_t2w
   deepseg/seg_tumor_edema_cavity_t1_t2
   deepseg/seg_tumor_t2