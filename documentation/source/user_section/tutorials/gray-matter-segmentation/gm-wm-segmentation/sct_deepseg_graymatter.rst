Gray matter segmentation algorithm: ``sct_deepseg graymatter``
##############################################################

For segmenting the gray matter, SCT features the ``graymatter`` task of :ref:`sct_deepseg`. This model
uses a 2D `nnU-Net <https://github.com/MIC-DKFZ/nnUNet>`_ architecture and outputs a binary segmentation.

* **Algorithm:** 2D nnU-Net (nnUNetV2)
* **Pros:** Agnostic to MRI contrast and spinal cord region; trained on diverse pathologies
* **Cons:** 2D model (processes axial slices independently)

The model was trained on datasets from **>20 sites**, covering:

* **3 magnetic field strengths:** 1.5T, 3T, 7T
* **9 MRI sequences:** T2*w, MTR, T1w (axial), PSIR, rAMIRA, PDw, MP2RAGE (UNI/T1map), QSM, SWI
* **Spinal cord regions:** Cervical, thoracic, and lumbar
* **1367 subjects** from healthy controls, pediatrics, multiple sclerosis, spinal muscular atrophy,
  cervical degenerative myelopathy, spinal cord injury, amyotrophic lateral sclerosis, post-polio
  syndrome, and stroke

More details about the model and its training data are available at
`github.com/ivadomed/model-gm-contrast-region-agnostic <https://github.com/ivadomed/model-gm-contrast-region-agnostic>`_.
