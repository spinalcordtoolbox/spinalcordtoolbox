
# Spinal Cord Toolbox [![Build Status](https://travis-ci.org/neuropoly/spinalcordtoolbox.svg?branch=master)](https://travis-ci.org/neuropoly/spinalcordtoolbox) [![Project Stats](https://www.openhub.net/p/spinalcordtoolbox/widgets/project_thin_badge.gif)](https://www.openhub.net/p/spinalcordtoolbox) [![User mailing list](http://img.shields.io/badge/mailing%20list-users-green.svg?style=flat)](https://groups.google.com/forum/#!forum/neuropoly) [![Developer mailing list](http://img.shields.io/badge/mailing%20list-development-green.svg?style=flat)](https://groups.google.com/forum/#!forum/sct_developers)


- [Introduction](#introduction)
- [Installation](https://sourceforge.net/p/spinalcordtoolbox/wiki/installation/)
- [List of tools](http://sourceforge.net/p/spinalcordtoolbox/wiki/tools/)
- [Getting started](https://github.com/neuropoly/spinalcordtoolbox/blob/master/batch_processing.sh)
- [Video tutorials](#video-tutorials)
- [List of changes](https://github.com/neuropoly/spinalcordtoolbox/blob/master/CHANGES.md)
- [References](#references)
- [How to cite SCT](#citing-spinalcordtoolbox)
- [License](#licence)


## Introduction
![Spinal Cord Toolbox overview](abstract.png "Spinal Cord Toolbox overview")

For the past 25 years, the field of neuroimaging has witnessed the
development of several software packages for processing multi-parametric
magnetic resonance imaging (mpMRI) to study the brain. These software packages
are now routinely used by researchers and clinicians, and have contributed to
important breakthroughs for the understanding of brain anatomy and function.
However, no software package exists to process mpMRI data of the spinal cord.
Despite the numerous clinical needs for such advanced mpMRI protocols (multiple
sclerosis, spinal cord injury, cervical spondylotic myelopathy, etc.),
researchers have been developing specific tools that, while necessary, do not
provide an integrative framework that is compatible with most usages and that is
capable of reaching the community at large. This hinders cross-validation and
the possibility to perform multi-center studies.

Spinal Cord Toolbox (SCT) is a comprehensive, free and open-source software dedicated to the
processing and analysis of spinal cord **MRI** data. **SCT** builds on
previously-validated methods and includes state-of-the-art **MRI** templates and
atlases of the spinal cord, algorithms to segment and register new data to the
templates, and motion correction methods for diffusion and functional time
series. **SCT** is tailored towards standardization and automation of the
processing pipeline, versatility, modularity, and it follows guidelines of
software development and distribution. Preliminary applications of **SCT** cover
a variety of studies, from cross-sectional area measures in large databases of
patients, to the precise quantification of mpMRI metrics in specific spinal
pathways. We anticipate that **SCT** will bring together the spinal cord
neuroimaging community by establishing standard templates and analysis
procedures.


## Video tutorials

#### Manual vertebral labeling

<a href="http://www.youtube.com/watch?feature=player_embedded&v=Q3DhKOCEl5s
" target="_blank"><img src="http://img.youtube.com/vi/Q3DhKOCEl5s/0.jpg" 
alt="Manual vertebral labeling" width="240" height="180" border="10" /></a>


## References

#### Spinal Cord Toolbox
- [De Leener B, Levy S, Dupont SM, Fonov VS, Stikov N, Louis Collins D, Callot V, Cohen-Adad J. SCT: Spinal Cord Toolbox, an open-source software for processing spinal cord MRI data. Neuroimage. 2017 Jan 15;145(Pt A):24-43.](https://www.ncbi.nlm.nih.gov/pubmed/27720818)

#### Template and Atlas
- [De Leener B, Fonov VS, Louis Collins D, Callot V, Stikov N, Cohen-Adad J. PAM50: Unbiased multimodal template of the brainstem and spinal cord aligned with the ICBM152 space. Neuroimage. 2017.](http://www.sciencedirect.com/science/article/pii/S1053811917308686)
- [Fonov et al. Framework for integrated MRI average of the spinal cord white and gray matter: The MNI-Poly-AMU template. Neuroimage 2014;102P2:817-827.](https://www.ncbi.nlm.nih.gov/pubmed/25204864)
- [Taso et al. Construction of an in vivo human spinal cord atlas based on high-resolution MR images at cervical and thoracic levels: preliminary results. MAGMA, Magn Reson Mater Phy 2014;27(3):257-267](https://www.ncbi.nlm.nih.gov/pubmed/24052240)
- [Lévy et al. White matter atlas of the human spinal cord with estimation of partial volume effect. Neuroimage. 2015 Oct 1;119:262-71](https://www.ncbi.nlm.nih.gov/pubmed/26099457)
- [Cadotte et al. Characterizing the Location of Spinal and Vertebral Levels in the Human Cervical Spinal Cord. AJNR Am J Neuroradiol 2014;36(5):1-8.](https://www.ncbi.nlm.nih.gov/pubmed/25523587)

#### Segmentation
- [Dupont SM, De Leener B, Taso M, Le Troter A, Stikov N, Callot V, Cohen-Adad J. Fully-integrated framework for the segmentation and registration of the spinal cord white and gray matter. Neuroimage. 2017 Apr 15;150:358-372.](https://www.ncbi.nlm.nih.gov/pubmed/27663988)
- [De Leener et al. Robust, accurate and fast automatic segmentation of the spinal cord. Neuroimage 2014;98:528-536.](https://www.ncbi.nlm.nih.gov/pubmed/24780696)
- [Ullmann et al. Automatic labeling of vertebral levels using a robust template-based approach. Int J Biomed Imaging 2014;Article ID 719520.](http://downloads.hindawi.com/journals/ijbi/2014/719520.pdf)
- [De Leener et al. Automatic segmentation of the spinal cord and spinal canal coupled with vertebral labeling. IEEE Transactions on Medical Imaging 2015;34(8):1705-1718](https://www.ncbi.nlm.nih.gov/pubmed/26011879)
- [Perone, C. S., Calabrese, E., & Cohen-Adad, J. (2017). Spinal cord gray matter segmentation using deep dilated convolutions.](https://arxiv.org/abs/1710.01269)


#### Registration
- [De Leener B, Mangeat G, Dupont S, Martin AR, Callot V, Stikov N, Fehlings MG, Cohen-Adad J. Topologically-preserving straightening of spinal cord MRI. J Magn Reson Imaging. 2017 Oct;46(4):1209-1219](https://www.ncbi.nlm.nih.gov/pubmed/28130805)
- [De Leener et al. Template-based analysis of multi-parametric MRI data with the Spinal Cord Toolbox. Proc. ISMRM, Toronto, Canada 2015](https://www.dropbox.com/s/zb2earfp7aqmfl2/deleener_ismrm15_sct.pdf?dl=0)
- [Cohen-Adad et al. Slice-by-slice regularized registration for spinal cord MRI: SliceReg. Proc. ISMRM, Toronto, Canada 2015](https://www.dropbox.com/s/v3bb3etbq4gb1l1/cohenadad_ismrm15_slicereg.pdf?dl=0)
- [Taso et al. A reliable spatially normalized template of the human spinal cord--Applications to automated white matter/gray matter segmentation and tensor-based morphometry (TBM) mapping of gray matter alterations occurring with age. Neuroimage. 2015 Aug 15;117:20-8](https://www.ncbi.nlm.nih.gov/pubmed/26003856)

#### Applications
- [Kong et al. Intrinsically organized resting state networks in the human spinal cord. PNAS 2014](http://www.pnas.org/content/111/50/18067.abstract)
- [Eippert F. et al. Investigating resting-state functional connectivity in the cervical spinal cord at 3T. Neuroimage 2016](https://www.ncbi.nlm.nih.gov/pubmed/28027960)
- [Weber K.A. et al. Functional Magnetic Resonance Imaging of the Cervical Spinal Cord During Thermal Stimulation Across Consecutive Runs. Neuroimage 2016](http://www.ncbi.nlm.nih.gov/pubmed/27616641)
- [Weber et al. Lateralization of cervical spinal cord activity during an isometric upper extremity motor task with functional magnetic resonance imaging. Neuroimage 2016](https://www.ncbi.nlm.nih.gov/pubmed/26488256)
- [Eippert et al. Denoising spinal cord fMRI data: Approaches to acquisition and analysis. Neuroimage 2016](https://www.ncbi.nlm.nih.gov/pubmed/27693613)
- [Samson et al., ZOOM or non-ZOOM? Assessing Spinal Cord Diffusion Tensor Imaging protocols for multi-centre studies. PLOS One 2016](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0155557)
- [Taso et al. Tract-specific and age-related variations of the spinal cord microstructure: a multi-parametric MRI study using diffusion tensor imaging (DTI) and inhomogeneous magnetization transfer (ihMT). NMR Biomed 2016](https://www.ncbi.nlm.nih.gov/pubmed/27100385)
- [Duval et al. In vivo mapping of human spinal cord microstructure at 300mT/m. Neuroimage 2015](https://www.ncbi.nlm.nih.gov/pubmed/26095093)
- [Massire A. et al. High-resolution multi-parametric quantitative magnetic resonance imaging of the human cervical spinal cord at 7T. Neuroimage 2016](https://www.ncbi.nlm.nih.gov/pubmed/27574985)
- [Duval et al. g-Ratio weighted imaging of the human spinal cord in vivo. Neuroimage 2016](https://www.ncbi.nlm.nih.gov/pubmed/27664830)
- [Ljungberg et al. Rapid Myelin Water Imaging in Human Cervical Spinal Cord. Magnetic Resonance in Medicine 2016](http://onlinelibrary.wiley.com/doi/10.1002/mrm.26551/abstract)
- [Yiannakas et al. Fully automated segmentation of the cervical cord from T1-weighted MRI using PropSeg: Application to multiple sclerosis. NeuroImage: Clinical 2015](https://www.ncbi.nlm.nih.gov/pubmed/26793433)
- Martin et al. Next-Generation MRI of the Human Spinal Cord: Quantitative Imaging Biomarkers for Cervical Spondylotic Myelopathy (CSM). Proc. 31th Annual Meeting of The Congress of Neurological Surgeons 2015
- [Castellano et al., Quantitative MRI of the spinal cord and brain in adrenomyeloneuropathy: in vivo assessment of structural changes. Brain 2016](http://brain.oxfordjournals.org/content/139/6/1735)
- [Grabher et al., Voxel-based analysis of grey and white matter degeneration in cervical spondylotic myelopathy. Sci Rep 2016;6:24636.](https://www.ncbi.nlm.nih.gov/pubmed/27095134)
- [Talbott JF, Narvid J, Chazen JL, Chin CT, Shah V. An Imaging Based Approach to Spinal Cord Infection. Semin Ultrasound CT MR. 2016](http://www.journals.elsevier.com/seminars-in-ultrasound-ct-and-mri/recent-articles)
- [McCoy et al. MRI Atlas-Based Measurement of Spinal Cord Injury Predicts Outcome in Acute Flaccid Myelitis. AJNR 2016](http://www.ajnr.org/content/early/2016/12/15/ajnr.A5044.abstract)
- [Taso et al. Anteroposterior compression of the spinal cord leading to cervical myelopathy: a finite element analysis. Comput Methods Biomech Biomed Engin 2015](http://www.tandfonline.com/doi/full/10.1080/10255842.2015.1069625)
- [De Leener et al. Segmentation of the human spinal cord. MAGMA. 2016](https://www.ncbi.nlm.nih.gov/pubmed/26724926)
- [Cohen-Adad et al. Functional Magnetic Resonance Imaging of the Spinal Cord: Current Status and Future Developments. Semin Ultrasound CT MR. 2016](http://www.sciencedirect.com/science/article/pii/S088721711630049X)
- [Ventura et al. Cervical spinal cord atrophy in NMOSD without a history of myelitis or MRI-visible lesions. Neurol Neuroimmunol Neuroinflamm 2016](https://www.ncbi.nlm.nih.gov/pubmed/27144215)
- [Battiston et al. Fast and reproducible in vivo T1 mapping of the human cervical spinal cord. Magnetic Resonance in Medicine 2017](http://onlinelibrary.wiley.com/doi/10.1002/mrm.26852/full)
- [Panara et al. Spinal cord microstructure integrating phase-sensitive inversion recovery and diffusional kurtosis imaging. Neuroradiology 2017](https://link.springer.com/article/10.1007%2Fs00234-017-1864-5)
- [Taso et al. Tract-specific and age-related variations of the spinal cord microstructure: a multi-parametric MRI study using diffusion tensor imaging (DTI) and inhomogeneous magnetization transfer (ihMT). NMR Biomed 2016](https://www.ncbi.nlm.nih.gov/pubmed/27100385)
- [Martin et al. Clinically Feasible Microstructural MRI to Quantify Cervical Spinal Cord Tissue Injury Using DTI, MT, and T2*-Weighted Imaging: Assessment of Normative Data and Reliability. AJNR 2017](https://www.ncbi.nlm.nih.gov/pubmed/28428213)
- [Martin et al. A Novel MRI Biomarker of Spinal Cord White Matter Injury: T2*-Weighted White Matter to Gray Matter Signal Intensity Ratio. AJNR 2017](https://www.ncbi.nlm.nih.gov/pubmed/28428212)
- [David et al. The efficiency of retrospective artifact correction methods in improving the statistical power of between-group differences in spinal cord DTI. Neuroimage 2017](http://www.sciencedirect.com/science/article/pii/S1053811917305220)
- [Battiston et al. An optimized framework for quantitative Magnetization Transfer imaging of the cervical spinal cord in vivo. Magnetic Resonance in Medicine 2017](http://onlinelibrary.wiley.com/doi/10.1002/mrm.26909/full)
- [Rasoanandrianina et al. Region-specific impairment of the cervical spinal cord (SC) in amyotrophic lateral sclerosis: A preliminary study using SC templates and quantitative MRI (diffusion tensor imaging/inhomogeneous magnetization transfer). NMR in Biomedicine 2017](http://onlinelibrary.wiley.com/doi/10.1002/nbm.3801/full)
- [Weber et al. Thermal Stimulation Alters Cervical Spinal Cord Functional Connectivity in Humans. Neurocience 2017](http://www.sciencedirect.com/science/article/pii/S0306452217307637)
- [Combes et al. Cervical cord myelin water imaging shows degenerative changes over one year in multiple sclerosis but not neuromyelitis optica spectrum disorder. Neuroimage: Clinical. 2016](http://www.sciencedirect.com/science/article/pii/S221315821730150X)
- [Grabher et al. Neurodegeneration in the Spinal Ventral Horn Prior to Motor Impairment in Cervical Spondylotic Myelopathy. Journal of Neurotrauma 2017](http://online.liebertpub.com/doi/abs/10.1089/neu.2017.4980)
- [Duval et al. Scan–rescan of axcaliber, macromolecular tissue volume, and g-ratio in the spinal cord. MRM 2017](http://onlinelibrary.wiley.com/doi/10.1002/mrm.26945/full)
- [Smith et al. Lateral corticospinal tract damage correlates with motor output in incomplete spinal cord injury. Archives of Physical Medicine and Rehabilitation 2017](http://www.sciencedirect.com/science/article/pii/S0003999317312844)
- [Kafali et al. Phase-correcting non-local means filtering for diffusion-weighted imaging of the spinal cord. MRM 2018](http://onlinelibrary.wiley.com/doi/10.1002/mrm.27105/full)
- [Hori et al. Application of Quantitative Microstructural MR Imaging with Atlas-based Analysis for the Spinal Cord in Cervical Spondylotic Myelopathy. Sci Rep. 2018 Mar 26;8(1):5213](https://www.nature.com/articles/s41598-018-23527-8)
- [Huber et al. Dorsal and ventral horn atrophy is associated with clinical outcome after spinal cord injury. Neurology. 2018 Mar 28](https://www.ncbi.nlm.nih.gov/pubmed/29592888)


## Citing spinalcordtoolbox

When citing SCT please use this BibTeX entry:

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

## License

The MIT License (MIT)

Copyright (c) 2014 École Polytechnique, Université de Montréal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
