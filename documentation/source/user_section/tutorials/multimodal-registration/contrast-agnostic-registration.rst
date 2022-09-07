.. _contrast-agnostic-registration:

Contrast-agnostic registration with deep learning
#################################################

This tutorial will demonstrate how to coregister two images together that have different contrasts using deep learning. The algorithm is based on [SynthMorph](https://arxiv.org/pdf/2004.10282.pdf). More details of its implementation in SCT can be found [here](https://github.com/ivadomed/multimodal-registration).

.. toctree::
   :maxdepth: 1

   contrast-agnostic-registration/before-starting
   contrast-agnostic-registration/preprocessing-t2
   contrast-agnostic-registration/preprocessing-t1
   contrast-agnostic-registration/coregistering-t1-t2