.. _references:

References
##########

If you use SCT in your research or as part of your developments, please always cite the `main reference`_.
If you use specific tools such as ``sct_deepseg_sc`` or the ``PAM50`` template, please also cite the specific articles
listed in `specific references`_. You could also see some of the applications of SCT by other groups in `applications`_.


.. contents::
   :local:
..


Main reference
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
  url = "http://www.sciencedirect.com/science/article/pii/S1053811916305560",
  author = "Benjamin De Leener and Simon Lévy and Sara M. Dupont and Vladimir S. Fonov and Nikola Stikov and D. Louis Collins and Virginie Callot and Julien Cohen-Adad",
  keywords = "Spinal cord",
  keywords = "MRI",
  keywords = "Software",
  keywords = "Template",
  keywords = "Atlas",
  keywords = "Open-source ",
  }


Specific references
-------------------

Command line tools
^^^^^^^^^^^^^^^^^^

The table below provides individual references for novel methods used in SCT's :ref:`command-line-tools`.

.. note::
   If you are using white matter/grey matter segmentation tools (``sct_deepseg_gm``/``sct_deepseg``) and registration tools (``sct_register_to_template``/``sct_register_multimodal``) together as part of a pipeline, please also consider this reference:

   `Dupont SM, De Leener B, Taso M, Le Troter A, Stikov N, Callot V, Cohen-Adad J. Fully-integrated framework for the segmentation and registration of the spinal cord white and gray matter. Neuroimage 2017 <https://www.ncbi.nlm.nih.gov/pubmed/27663988>`__

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Command line script
     - References
   * - ``sct_deepseg_gm``
     - `Perone et al. Spinal cord gray matter segmentation using deep dilated convolutions. Sci Rep 2018 <https://www.nature.com/articles/s41598-018-24304-3>`__
   * - ``sct_deepseg_lesion``
     - `Gros et al. Automatic segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional neural networks. Neuroimage 2019 <https://www.sciencedirect.com/science/article/pii/S1053811918319578>`__
   * - ``sct_deepseg_sc``
     - `Gros et al. Automatic segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional neural networks. Neuroimage 2019 <https://www.sciencedirect.com/science/article/pii/S1053811918319578>`__
   * - ``sct_get_centerline``
     - `Gros et al. Automatic spinal cord localization, robust to MRI contrasts using global curve optimization. Med Image Anal 2018 <https://www.sciencedirect.com/science/article/pii/S136184151730186X>`__
   * - ``sct_label_vertebrae``
     - `Ullmann et al. Automatic labeling of vertebral levels using a robust template-based approach. Int J Biomed Imaging 2014 <http://downloads.hindawi.com/journals/ijbi/2014/719520.pdf>`__
   * - ``sct_propseg``
     - `De Leener et al. Robust, accurate and fast automatic segmentation of the spinal cord. Neuroimage 2014 <https://www.ncbi.nlm.nih.gov/pubmed/24780696>`__
   * - ``sct_propseg -CSF``
     - `De Leener et al. Automatic segmentation of the spinal cord and spinal canal coupled with vertebral labeling. IEEE Transactions on Medical Imaging 2015 <https://www.ncbi.nlm.nih.gov/pubmed/26011879>`__
   * - ``sct_register_multimodal``
     - `De Leener B, Fonov VS, Louis Collins D, Callot V, Stikov N, Cohen-Adad J. PAM50: Unbiased multimodal template of the brainstem and spinal cord aligned with the ICBM152 space. Neuroimage 2017. <http://www.sciencedirect.com/science/article/pii/S1053811917308686>`__
   * - ``sct_register_multimodal --param algo=slicereg``
     - `Cohen-Adad et al. Slice-by-slice regularized registration for spinal cord MRI: SliceReg. Proc ISMRM 2015 <https://www.dropbox.com/s/v3bb3etbq4gb1l1/cohenadad_ismrm15_slicereg.pdf?dl=0>`__
   * - ``sct_register_to_template``
     - `De Leener B, Fonov VS, Louis Collins D, Callot V, Stikov N, Cohen-Adad J. PAM50: Unbiased multimodal template of the brainstem and spinal cord aligned with the ICBM152 space. Neuroimage 2017. <http://www.sciencedirect.com/science/article/pii/S1053811917308686>`__
   * - ``sct_register_to_template --param algo=slicereg``
     - `Cohen-Adad et al. Slice-by-slice regularized registration for spinal cord MRI: SliceReg. Proc ISMRM 2015 <https://www.dropbox.com/s/v3bb3etbq4gb1l1/cohenadad_ismrm15_slicereg.pdf?dl=0>`__
   * - ``sct_straighten_spinalcord``
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
     - `De Leener B, Fonov VS, Louis Collins D, Callot V, Stikov N, Cohen-Adad J. PAM50: Unbiased multimodal template of the brainstem and spinal cord aligned with the ICBM152 space. Neuroimage 2017. <http://www.sciencedirect.com/science/article/pii/S1053811917308686>`__
   * - MNI-Poly-AMU template
     - `Fonov et al. Framework for integrated MRI average of the spinal cord white and gray matter: The MNI-Poly-AMU template. Neuroimage 2014. <https://www.ncbi.nlm.nih.gov/pubmed/25204864>`__
   * - White matter atlas
     - `Lévy et al. White matter atlas of the human spinal cord with estimation of partial volume effect. Neuroimage 2015 <https://www.ncbi.nlm.nih.gov/pubmed/26099457>`__
   * - Probabilistic atlas (AMU40)
     - `Taso et al. A reliable spatially normalized template of the human spinal cord–Applications to automated white matter/gray matter segmentation and tensor-based morphometry (TBM) mapping of gray matter alterations occurring with age. Neuroimage 2015 <https://www.ncbi.nlm.nih.gov/pubmed/26003856>`__
   * - Spinal levels
     - `Cadotte DW, Cadotte A, Cohen-Adad J, Fleet D, Livne M, Wilson JR, Mikulis D, Nugaeva N, Fehlings MG. Characterizing the location of spinal and vertebral levels in the human cervical spinal cord. AJNR Am J Neuroradiol, 2015, 36(4):803-810. <https://paperpile.com/app/p/5b580317-6921-06c8-a2ee-685d4dbaa44c>`_


.. _references-applications:

Applications
------------
The following studies (in chronological order) have used SCT:

-  `Kong et al. Intrinsically organized resting state networks in the human spinal cord. PNAS 2014 <http://www.pnas.org/content/111/50/18067.abstract>`__
-  `Duval et al. In vivo mapping of human spinal cord microstructure at 300mT/m. Neuroimage 2015 <https://www.ncbi.nlm.nih.gov/pubmed/26095093>`__
-  `Yiannakas et al. Fully automated segmentation of the cervical cord from T1-weighted MRI using PropSeg: Application to multiple sclerosis. NeuroImage: Clinical 2015 <https://www.ncbi.nlm.nih.gov/pubmed/26793433>`__
-  `Taso et al. Anteroposterior compression of the spinal cord leading to cervical myelopathy: a finite element analysis. Comput Methods Biomech Biomed Engin 2015 <http://www.tandfonline.com/doi/full/10.1080/10255842.2015.1069625>`__
-  `Eippert F. et al. Investigating resting-state functional connectivity in the cervical spinal cord at 3T. Neuroimage 2016 <https://www.ncbi.nlm.nih.gov/pubmed/28027960>`__
-  `Weber K.A. et al. Functional Magnetic Resonance Imaging of the Cervical Spinal Cord During Thermal Stimulation Across Consecutive Runs. Neuroimage 2016 <http://www.ncbi.nlm.nih.gov/pubmed/27616641>`__
-  `Weber et al. Lateralization of cervical spinal cord activity during an isometric upper extremity motor task with functional magnetic resonance imaging. Neuroimage 2016 <https://www.ncbi.nlm.nih.gov/pubmed/26488256>`__
-  `Eippert et al. Denoising spinal cord fMRI data: Approaches to acquisition and analysis. Neuroimage 2016 <https://www.ncbi.nlm.nih.gov/pubmed/27693613>`__
-  `Samson et al. ZOOM or non-ZOOM? Assessing Spinal Cord Diffusion Tensor Imaging protocols for multi-centre studies. PLOS One 2016 <http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155557>`__
-  `Taso et al. Tract-specific and age-related variations of the spinal cord microstructure: a multi-parametric MRI study using diffusion tensor imaging (DTI) and inhomogeneous magnetization transfer (ihMT). NMR Biomed 2016 <https://www.ncbi.nlm.nih.gov/pubmed/27100385>`__
-  `Massire A. et al. High-resolution multi-parametric quantitative magnetic resonance imaging of the human cervical spinal cord at 7T. Neuroimage 2016 <https://www.ncbi.nlm.nih.gov/pubmed/27574985>`__
-  `Duval et al. g-Ratio weighted imaging of the human spinal cord in vivo. Neuroimage 2016 <https://www.ncbi.nlm.nih.gov/pubmed/27664830>`__
-  `Ljungberg et al. Rapid Myelin Water Imaging in Human Cervical Spinal Cord. Magn Reson Med 2016 <https://www.ncbi.nlm.nih.gov/pubmed/28940333>`__
-  `Castellano et al. Quantitative MRI of the spinal cord and brain in adrenomyeloneuropathy: in vivo assessment of structural changes. Brain 2016 <http://brain.oxfordjournals.org/content/139/6/1735>`__
-  `Grabher et al. Voxel-based analysis of grey and white matter degeneration in cervical spondylotic myelopathy. Sci Rep 2016 <https://www.ncbi.nlm.nih.gov/pubmed/27095134>`__
-  `Talbott JF, Narvid J, Chazen JL, Chin CT, Shah V. An Imaging Based Approach to Spinal Cord Infection. Semin Ultrasound CT MR 2016 <http://www.journals.elsevier.com/seminars-in-ultrasound-ct-and-mri/recent-articles>`__
-  `McCoy et al. MRI Atlas-Based Measurement of Spinal Cord Injury Predicts Outcome in Acute Flaccid Myelitis. AJNR 2016 <http://www.ajnr.org/content/early/2016/12/15/ajnr.A5044.abstract>`__
-  `De Leener et al. Segmentation of the human spinal cord. MAGMA. 2016 <https://www.ncbi.nlm.nih.gov/pubmed/26724926>`__
-  `Cohen-Adad et al. Functional Magnetic Resonance Imaging of the Spinal Cord: Current Status and Future Developments. Semin Ultrasound CT MR 2016 <http://www.sciencedirect.com/science/article/pii/S088721711630049X>`__
-  `Ventura et al. Cervical spinal cord atrophy in NMOSD without a history of myelitis or MRI-visible lesions. Neurol Neuroimmunol Neuroinflamm 2016 <https://www.ncbi.nlm.nih.gov/pubmed/27144215>`__
-  `Combes et al. Cervical cord myelin water imaging shows degenerative changes over one year in multiple sclerosis but not neuromyelitis optica spectrum disorder. Neuroimage: Clinical. 2016 <http://www.sciencedirect.com/science/article/pii/S221315821730150X>`__
-  `Battiston et al. Fast and reproducible in vivo T1 mapping of the human cervical spinal cord. Magn Reson Med 2017 <http://onlinelibrary.wiley.com/doi/10.1002/mrm.26852/full>`__
-  `Panara et al. Spinal cord microstructure integrating phase-sensitive inversion recovery and diffusional kurtosis imaging. Neuroradiology 2017 <https://link.springer.com/article/10.1007%2Fs00234-017-1864-5>`__
-  `Martin et al. Clinically Feasible Microstructural MRI to Quantify Cervical Spinal Cord Tissue Injury Using DTI, MT, and T2*-Weighted Imaging: Assessment of Normative Data and Reliability. AJNR 2017 <https://www.ncbi.nlm.nih.gov/pubmed/28428213>`__
-  `Martin et al. A Novel MRI Biomarker of Spinal Cord White Matter Injury: T2*-Weighted White Matter to Gray Matter Signal Intensity Ratio. AJNR 2017 <https://www.ncbi.nlm.nih.gov/pubmed/28428212>`__
-  `David et al. The efficiency of retrospective artifact correction methods in improving the statistical power of between-group differences in spinal cord DTI. Neuroimage 2017 <http://www.sciencedirect.com/science/article/pii/S1053811917305220>`__
-  `Battiston et al. An optimized framework for quantitative Magnetization Transfer imaging of the cervical spinal cord in vivo. Magnetic Resonance in Medicine 2017 <http://onlinelibrary.wiley.com/doi/10.1002/mrm.26909/full>`__
-  `Rasoanandrianina et al. Region-specific impairment of the cervical spinal cord (SC) in amyotrophic lateral sclerosis: A preliminary study using SC templates and quantitative MRI (diffusion tensor imaging/inhomogeneous magnetization transfer). NMR Biomed 2017 <http://onlinelibrary.wiley.com/doi/10.1002/nbm.3801/full>`__
-  `Weber et al. Thermal Stimulation Alters Cervical Spinal Cord Functional Connectivity in Humans. Neurocience 2017 <http://www.sciencedirect.com/science/article/pii/S0306452217307637>`__
-  `Grabher et al. Neurodegeneration in the Spinal Ventral Horn Prior to Motor Impairment in Cervical Spondylotic Myelopathy. Journal of Neurotrauma 2017 <http://online.liebertpub.com/doi/abs/10.1089/neu.2017.4980>`__
-  `Duval et al. Scan–rescan of axcaliber, macromolecular tissue volume, and g-ratio in the spinal cord. Magn Reson Med 2017 <http://onlinelibrary.wiley.com/doi/10.1002/mrm.26945/full>`__
-  `Smith et al. Lateral corticospinal tract damage correlates with motor output in incomplete spinal cord injury. Archives of Physical Medicine and Rehabilitation 2017 <http://www.sciencedirect.com/science/article/pii/S0003999317312844>`__
-  `Prados et al. Spinal cord grey matter segmentation challenge. Neuroimage 2017 <https://www.sciencedirect.com/science/article/pii/S1053811917302185#f0005>`__
-  `Peterson et al. Test-Retest and Interreader Reproducibility of Semiautomated Atlas-Based Analysis of Diffusion Tensor Imaging Data in Acute Cervical Spine Trauma in Adult Patients. AJNR Am J Neuroradiol. 2017 Oct;38(10):2015-2020 <https://www.ncbi.nlm.nih.gov/pubmed/28818826>`__
-  `Kafali et al. Phase-correcting non-local means filtering for diffusion-weighted imaging of the spinal cord. Magn Reson Med 2018 <http://onlinelibrary.wiley.com/doi/10.1002/mrm.27105/full>`__
-  `Albrecht et al. Neuroinflammation of the spinal cord and nerve roots in chronic radicular pain patients. Pain. 2018 May;159(5):968-977. doi: 10.1097/j.pain.0000000000001171 <https://www.ncbi.nlm.nih.gov/pubmed/29419657>`__
-  `Hori et al. Application of Quantitative Microstructural MR Imaging with Atlas-based Analysis for the Spinal Cord in Cervical Spondylotic Myelopathy. Sci Rep 2018 <https://www.nature.com/articles/s41598-018-23527-8>`__
-  `Huber et al. Dorsal and ventral horn atrophy is associated with clinical outcome after spinal cord injury. Neurology 2018 <https://www.ncbi.nlm.nih.gov/pubmed/29592888>`__
-  `Dostal et al. Analysis of diffusion tensor measurements of the human cervical spinal cord based on semiautomatic segmentation of the white and gray matter. J Magn Reson Imaging 2018 <https://www.ncbi.nlm.nih.gov/pubmed/29707834>`__
-  `Calabrese et al. Postmortem diffusion MRI of the entire human spinal cord at microscopic resolution. Neuroimage Clin, 2018 <https://www.ncbi.nlm.nih.gov/pubmed/29876281>`__
-  `Paquin et al. Spinal Cord Gray Matter Atrophy in Amyotrophic Lateral Sclerosis. AJNR 2018 <http://www.ajnr.org/content/39/1/184>`__
-  `Combès et al. Focal and diffuse cervical spinal cord damage in patients with early relapsing-remitting MS: A multicentre magnetisation transfer ratio study. Multiple Sclerosis Journal, 2018 <https://www.ncbi.nlm.nih.gov/m/pubmed/29909771/>`__
-  `Martin et al. Monitoring for myelopathic progression with multiparametric quantitative MRI. PLoS One. 2018 Apr 17;13(4):e0195733 <https://www.ncbi.nlm.nih.gov/pubmed/29664964>`__
-  `Martin et al. Can microstructural MRI detect subclinical tissue injury in subjects with asymptomatic cervical spinal cord compression? A prospective cohort study. BMJ Open, 2018 <https://www.ncbi.nlm.nih.gov/pubmed/29654015>`__
-  `Querin et al. The spinal and cerebral profile of adult spinal-muscular atrophy: A multimodal imaging study. NeuroImage Clin, 2018 <https://www.sciencedirect.com/science/article/pii/S2213158218303668>`__
-  `Shokur et al. Training with brain-machine interfaces, visuo-tactile feedback and assisted locomotion improves sensorimotor, visceral, and psychological signs in chronic paraplegic patients. Plos One, 2018 <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0206464>`__
-  `Panara et al. Correlations between cervical spinal cord magnetic resonance diffusion tensor and diffusion kurtosis imaging metrics and motor performance in patients with chronic ischemic brain lesions of the corticospinal tract. Neuroradiology, 2018 <https://link.springer.com/article/10.1007/s00234-018-2139-5>`__
-  `Moccia et al. Advances in spinal cord imaging in multiple sclerosis. Ther Adv Neurol Disord, 2019 <https://journals.sagepub.com/doi/pdf/10.1177/1756286419840593>`__
-  `Kitany et al. Functional imaging of rostrocaudal spinal activity during upper limb motor tasks. Neuroimage, 2019 <https://www.sciencedirect.com/science/article/pii/S1053811919304288>`__
-  `Lorenzi et al. Unsuspected Involvement of Spinal Cord in Alzheimer Disease. Front Cell Neurosci, 2020 <https://www.frontiersin.org/articles/10.3389/fncel.2020.00006/full>`__
-  `Papinutto et al. Evaluation of Intra- and Interscanner Reliability of MRI Protocols for Spinal Cord Gray Matter and Total Cross-Sectional Area Measurements. J Magn Reson Imaging, 2019 <https://onlinelibrary.wiley.com/doi/epdf/10.1002/jmri.26269>`__
-  `Weeda et al. Validation of mean upper cervical cord area (MUCCA) measurement techniques in multiple sclerosis (MS): High reproducibility and robustness to lesions, but large software and scanner effects. NeuroImage Clin, 2019 <https://www.sciencedirect.com/science/article/pii/S2213158219303122>`__
-  `Moccia et al. Longitudinal spinal cord atrophy in multiple sclerosis using the generalised boundary shift integral. Ann Neurol, 2019 <https://onlinelibrary.wiley.com/doi/abs/10.1002/ana.25571>`__
-  `Rasoanandrianina et al. Regional T1 mapping of the whole cervical spinal cord using an optimized MP2RAGE sequence. NMR Biomed, 2019 <https://onlinelibrary.wiley.com/doi/full/10.1002/nbm.4142>`__
-  `Hopkins et al. Machine Learning for the Prediction of Cervical Spondylotic Myelopathy: A Post Hoc Pilot Study of 28 Participants. World Neurosurg, 2019 <https://www.sciencedirect.com/science/article/pii/S1878875019308459>`__
-  `Karbasforoushan et al. Brainstem and spinal cord MRI identifies altered sensorimotor pathways post-stroke. Nat Commun, 2019 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6684621/>`__
-  `Seif et al. Guidelines for the conduct of clinical trials in spinal cord injury: Neuroimaging biomarkers. Spinal Cord, 2019 <https://www.ncbi.nlm.nih.gov/pubmed/31267015>`__
-  `Lorenzi et al. Unsuspected Involvement of Spinal Cord in Alzheimer Disease. Front Cell Neurosci, 2019 <https://www.frontiersin.org/articles/10.3389/fncel.2020.00006/full>`__
-  `Sabaghian et al. Fully Automatic 3D Segmentation of the Thoracolumbar Spinal Cord and the Vertebral Canal From T2-weighted MRI Using K-means Clustering Algorithm. Spinal Cord, 2020 <https://pubmed.ncbi.nlm.nih.gov/32132652/>`__
-  `Bonacci et al. Clinical Relevance of Multiparametric MRI Assessment of Cervical Cord Damage in Multiple Sclerosis. Radiology, 2020 <https://pubmed.ncbi.nlm.nih.gov/32573387/>`__
-  `Hori. Sodium in the Relapsing - Remitting Multiple Sclerosis Spinal Cord: Increased Concentrations and Associations With Microstructural Tissue Anisotropy. JMRI, 2020 <https://onlinelibrary.wiley.com/doi/abs/10.1002/jmri.27253>`__
-  `Lersy et al. Identification and measurement of cervical spinal cord atrophy in neuromyelitis optica spectrum disorders (NMOSD) and correlation with clinical characteristics and cervical spinal cord MRI data. Revue Neurologique, 2020 <https://www.sciencedirect.com/science/article/pii/S0035378720306159>`__
-  `Dahlberg et al. Heritability of cervical spinal cord structure. Neurol Genet, 2020 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7061306/>`__
-  `Shinn et al. Magnetization transfer and diffusion tensor imaging in dogs with intervertebral disk herniation. Journal of Veterinary Internal Medicine, 2020 <https://pubmed.ncbi.nlm.nih.gov/33006411/>`__
-  `Azzarito et al. Simultaneous voxel‐wise analysis of brain and spinal cord morphometry and microstructure within the SPM framework. Human Brain Mapping, 2020 <https://pubmed.ncbi.nlm.nih.gov/32991031/>`__
-  `Paliwal et al. Magnetization Transfer Ratio and Morphometrics Of the Spinal Cord Associates withSurgical Recovery in Patients with Degenerative Cervical Myelopathy. World Neurosurgery, 2020 <https://pubmed.ncbi.nlm.nih.gov/33010502/>`__
-  `Tinnermann et al. Cortico-spinal imaging to study pain. NeuroImage.2020 <https://www.sciencedirect.com/science/article/pii/S1053811920309241?via%3Dihub>`__
-  `Rejc et al. Spinal Cord Imaging Markers and Recovery of Volitional Leg Movement With Spinal Cord Epidural Stimulation in Individuals With Clinically Motor Complete Spinal Cord Injury. Front. Syst. Neurosci., 2020 <https://www.frontiersin.org/articles/10.3389/fnsys.2020.559313/full>`__
-  `Labounek et al. HARDI-ZOOMit protocol improves specificity to microstructural changes in presymptomatic myelopathy. Scientific Reports, 2020 <https://www.nature.com/articles/s41598-020-70297-3>`__
-  `Henmar et al. What are the gray and white matter volumes of the human spinal cord? J Neurophysiol, 2020 <https://pubmed.ncbi.nlm.nih.gov/33085549/>`__
-  `Burke et al. Injury Volume Extracted from MRI Predicts Neurologic Outcome in Acute Spinal Cord Injury: A Prospective TRACK-SCI Pilot Study. J Clin Neurosci, 2020 <https://www.sciencedirect.com/science/article/abs/pii/S0967586820316192>`__
-  `Mossa-Basha et al. Segmented quantitative diffusion tensor imaging evaluation of acute traumatic cervical spinal cord injury. Br J Radiol, 2020 <https://pubmed.ncbi.nlm.nih.gov/33180553/>`__
-  `Mariano et al. Quantitative spinal cord MRI in MOG-antibody disease, neuromyelitis optica and multiple sclerosis. Brain, 2020 <https://pubmed.ncbi.nlm.nih.gov/33206944/>`__
-  `Fratini et al. Multiscale Imaging Approach for Studying the Central Nervous System: Methodology and Perspective. Front Neurosci, 2020 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7019007/>`__
-  `Hoggarth et al. Macromolecular changes in spinal cord white matter characterize whiplash outcome at 1-year post motor vehicle collision. Scientific Reports, 2020 <https://www.nature.com/articles/s41598-020-79190-5>`__
-  `Stroman et al. A comparison of the effectiveness of functional MRI analysis methods for pain research: The new normal. PLoS One, 2020 <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243723>`__
-  `Johnson et al. In vivo detection of microstructural spinal cord lesions in dogs with degenerative myelopathy using diffusion tensor imaging. J Vet Intern Med. 2020 <https://onlinelibrary.wiley.com/doi/10.1111/jvim.16014>`_
-  `Kinawy et al. Dynamic Functional Connectivity of Resting-State Spinal Cord fMRI Reveals Fine-Grained Intrinsic Architecture. Neuron. 2020 <https://pubmed.ncbi.nlm.nih.gov/32910894/>`_
-  `Weber et al. Assessing the spatial distribution of cervical spinal cord activity during tactile stimulation of the upper extremity in humans with functional magnetic resonance imaging. Neuroimage 2020 <https://www.sciencedirect.com/science/article/pii/S1053811920303918>`_
-  `Azzarito et al. Tracking the neurodegenerative gradient after spinal cord injury. NeuroImage Clinical, 2020 <https://pubmed.ncbi.nlm.nih.gov/32145681/>`_
-  `Querin et al. Development of new outcome measures for adult SMA type III and IV: a multimodal longitudinal study. J Neurol 2021 <https://pubmed.ncbi.nlm.nih.gov/33388927/>`_
-  `McLachlin et al. Spatial correspondence of spinal cord white matter tracts using diffusion tensor imaging, fibre tractography, and atlas-based segmentation. Neuroradiology 2021 <https://link.springer.com/article/10.1007/s00234-021-02635-9>`_
-  `Dvorak et al. Comparison of multi echo T2 relaxation and steady state approaches for myelin imaging in the central nervous system. Scientific reports 2021 <https://www.nature.com/articles/s41598-020-80585-7>`_
-  `Adanyeguh et al. Multiparametric in vivo analyses of the brain and spine identify structural and metabolic biomarkers in men with adrenomyeloneuropathy. NeuroImage: Clinical, 2021 <https://www.sciencedirect.com/science/article/pii/S2213158221000103>`_
-  `Meyer et al. Optimized cervical spinal cord perfusion MRI after traumatic injury in the rat. J. of Cerebral Blood Flow & Metabolism, 2021 <https://journals.sagepub.com/doi/10.1177/0271678X20982396>`_
-  `Solanes et al. 3D patient-specific spinal cord computational model for SCS management: potential clinical applications. Journal of Neural Engineering, 2021 <https://pubmed.ncbi.nlm.nih.gov/33556926/>`_
-  `Johnson et al. Changes in White Matter of the Cervical Spinal Cord after a Single Season of Collegiate Football. Neurotrauma Reports, 2021 <https://www.liebertpub.com/doi/10.1089/neur.2020.0035>`_
-  `Ost et al. Spinal Cord Morphology in Degenerative Cervical Myelopathy Patients; Assessing Key Morphological Characteristics Using Machine Vision Tools. Journal of Clinical Medicine, 2021 <https://www.mdpi.com/2077-0383/10/4/892>`_
