.. _citing-sct:

Citing SCT
##########

If you use SCT in your research or as part of your developments, please always cite the `main reference`_.
As well, please cite the reference(s) to the specific tool(s) you utilized, detailed in `specific references`_, whenever possible.


Main Reference
--------------

-  `De Leener B, Levy S, Dupont SM, Fonov VS, Stikov N, Louis Collins D,
   Callot V, Cohen-Adad J. SCT: Spinal Cord Toolbox, an open-source
   software for processing spinal cord MRI data. Neuroimage
   2017. <https://www.ncbi.nlm.nih.gov/pubmed/27720818>`__

.. code-block:: none

    @article{DeLeener201724,
        title = "SCT: Spinal Cord Toolbox, an open-source software for processing spinal cord \{MRI\} data ",
        journal = "NeuroImage ",
        volume = "145, Part A",
        number = "",
        pages = "24 - 43",
        year = "2017",
        note = "",
        issn = "1053-8119",
        doi = "https://doi.org/10.1016/j.neuroimage.2016.10.009",
        url = "https://www.sciencedirect.com/science/article/pii/S1053811916305560",
        author = "Benjamin De Leener and Simon Lévy and Sara M. Dupont and Vladimir S. Fonov and Nikola Stikov and D. Louis Collins and Virginie Callot and Julien Cohen-Adad",
        keywords = "Spinal cord",
        keywords = "MRI",
        keywords = "Software",
        keywords = "Template",
        keywords = "Atlas",
        keywords = "Open-source ",
    }


Specific References
-------------------


DeepSeg
^^^^^^^

The table below lists the references for :ref:`sct_deepseg` sub-tasks which have one:

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - DeepSeg Task
      - References
    * - :ref:`sct_deepseg_spinalcord`
      - `Bédard, Sandrine, et al. "Towards contrast-agnostic soft segmentation of the spinal cord." Medical Image Analysis (2025): 103473. <https://www.sciencedirect.com/science/article/pii/S1361841525000210>`__
    * - :ref:`sct_deepseg_sc_epi`
      - `Banerjee, Rohan, et al. "EPISeg: Automated segmentation of the spinal cord on echo planar images using open-access multi-center data." bioRxiv(2025): 2025-01. <https://www.biorxiv.org/content/10.1101/2025.01.07.631402v2.abstract>`__
    * - :ref:`sct_deepseg_gm_sc_7t_t2star`
      - `Medina, Nilser, et al. "2D multi-class model for gray and white matter segmentation of the cervical spinal cord at 7T." arXiv preprint arXiv:2110.06516 (2021). <https://arxiv.org/abs/2110.06516>`__
    * - :ref:`sct_deepseg_gm_wm_mouse_t1`
      - `Cohen-Adad, J. (2024). Segmentation model of ex vivo mouse spinal cord white and gray matter (v0.5). Zenodo. https://doi.org/10.5281/zenodo.10819207 <https://doi.org/10.5281/zenodo.10819207>`__
    * - :ref:`sct_deepseg_lesion_sci_t2`
      - `Karthik, Enamundram Naga, et al. "SCIsegV2: A Universal Tool for Segmentation of Intramedullary Lesions in Spinal Cord Injury." International Workshop on Applications of Medical AI. Cham: Springer Nature Switzerland, 2025. <https://link.springer.com/chapter/10.1007/978-3-031-82007-6_19>`__
    * - :ref:`sct_deepseg_lesion_ms_mp2rage`
      - `Valošek, Jan, et al. "Automatic segmentation of the spinal cord nerve rootlets." Imaging Neuroscience 2 (2024): 1-14. <https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00218/122601>`__
    * - :ref:`sct_deepseg_tumor_edema_cavity_t1_t2`
      - `Lemay, Andreanne, et al. "Automatic multiclass intramedullary spinal cord tumor segmentation on MRI with deep learning." NeuroImage: Clinical 31 (2021): 102766. <Lemay, Andreanne, et al. "Automatic multiclass intramedullary spinal cord tumor segmentation on MRI with deep learning." NeuroImage: Clinical 31 (2021): 102766.>`__

Command Line Tools
^^^^^^^^^^^^^^^^^^

The table below provides individual references for novel methods used in SCT's :ref:`command-line-tools`.

.. note::
   If you are using white matter/grey matter segmentation tools (:ref:`sct_deepseg_gm`/:ref:`sct_deepseg`) and registration tools (:ref:`sct_register_to_template`/:ref:`sct_register_multimodal`) together as part of a pipeline, please also consider this reference:

   `Dupont SM, De Leener B, Taso M, Le Troter A, Stikov N, Callot V, Cohen-Adad J. Fully-integrated framework for the segmentation and registration of the spinal cord white and gray matter. Neuroimage 2017 <https://www.ncbi.nlm.nih.gov/pubmed/27663988>`__

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Command line script
     - References
   * - :ref:`sct_deepseg_gm`
     - `Perone et al. Spinal cord gray matter segmentation using deep dilated convolutions. Sci Rep 2018 <https://www.nature.com/articles/s41598-018-24304-3>`__
   * -  :ref:`sct_deepseg_sc` / :ref:`sct_deepseg_lesion`
     - `Gros et al. Automatic segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional neural networks. Neuroimage 2019 <https://www.sciencedirect.com/science/article/pii/S1053811918319578>`__
   * - :ref:`sct_get_centerline`
     - `Gros et al. Automatic spinal cord localization, robust to MRI contrasts using global curve optimization. Med Image Anal 2018 <https://www.sciencedirect.com/science/article/pii/S136184151730186X>`__
   * - :ref:`sct_label_vertebrae`
     - `Ullmann et al. Automatic labeling of vertebral levels using a robust template-based approach. Int J Biomed Imaging 2014 <https://onlinelibrary.wiley.com/doi/10.1155/2014/719520>`__
   * - :ref:`sct_process_segmentation` ``-pmj`` / ``-normalize``
     - `Bédard S, Cohen-Adad J. Automatic measure and normalization of spinal cord cross-sectional area using the pontomedullary junction. Frontiers in Neuroimaging 2022 <https://doi.org/10.3389/fnimg.2022.1031253>`__
   * - :ref:`sct_propseg`
     - `De Leener et al. Robust, accurate and fast automatic segmentation of the spinal cord. Neuroimage 2014 <https://www.ncbi.nlm.nih.gov/pubmed/24780696>`__
   * - :ref:`sct_propseg` ``-CSF``
     - `De Leener et al. Automatic segmentation of the spinal cord and spinal canal coupled with vertebral labeling. IEEE Transactions on Medical Imaging 2015 <https://www.ncbi.nlm.nih.gov/pubmed/26011879>`__
   * - :ref:`sct_register_multimodal` / :ref:`sct_register_to_template`
     - `De Leener B, Fonov VS, Louis Collins D, Callot V, Stikov N, Cohen-Adad J. PAM50: Unbiased multimodal template of the brainstem and spinal cord aligned with the ICBM152 space. Neuroimage 2017. <https://www.sciencedirect.com/science/article/pii/S1053811917308686>`__
   * - :ref:`sct_register_multimodal` / :ref:`sct_register_to_template` ``--param algo=slicereg``
     - `Cohen-Adad et al. Slice-by-slice regularized registration for spinal cord MRI: SliceReg. Proc ISMRM 2015 <https://www.dropbox.com/s/v3bb3etbq4gb1l1/cohenadad_ismrm15_slicereg.pdf?dl=0>`__
   * - :ref:`sct_straighten_spinalcord`
     - `De Leener B et al. Topologically-preserving straightening of spinal cord MRI. J Magn Reson Imaging 2017 <https://www.ncbi.nlm.nih.gov/pubmed/28130805>`__

Template and Atlas
^^^^^^^^^^^^^^^^^^

The table below provides references relevant to the :ref:`pam50` used by SCT, including a reference for the template itself, as well as earlier works that the template builds on.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Template/atlas
     - References
   * - PAM50 template
     - `De Leener B, Fonov VS, Louis Collins D, Callot V, Stikov N, Cohen-Adad J. PAM50: Unbiased multimodal template of the brainstem and spinal cord aligned with the ICBM152 space. Neuroimage 2018. <https://www.sciencedirect.com/science/article/pii/S1053811917308686>`__
   * - MNI-Poly-AMU template
     - `Fonov et al. Framework for integrated MRI average of the spinal cord white and gray matter: The MNI-Poly-AMU template. Neuroimage 2014. <https://www.ncbi.nlm.nih.gov/pubmed/25204864>`__
   * - White matter atlas
     - `Lévy et al. White matter atlas of the human spinal cord with estimation of partial volume effect. Neuroimage 2015 <https://www.ncbi.nlm.nih.gov/pubmed/26099457>`__
   * - Probabilistic atlas (AMU40)
     - `Taso et al. A reliable spatially normalized template of the human spinal cord–Applications to automated white matter/gray matter segmentation and tensor-based morphometry (TBM) mapping of gray matter alterations occurring with age. Neuroimage 2015 <https://www.ncbi.nlm.nih.gov/pubmed/26003856>`__
   * - Spinal levels
     - `Cadotte DW, Cadotte A, Cohen-Adad J, Fleet D, Livne M, Wilson JR, Mikulis D, Nugaeva N, Fehlings MG. Characterizing the location of spinal and vertebral levels in the human cervical spinal cord. AJNR Am J Neuroradiol, 2015, 36(4):803-810. <https://paperpile.com/app/p/5b580317-6921-06c8-a2ee-685d4dbaa44c>`_
