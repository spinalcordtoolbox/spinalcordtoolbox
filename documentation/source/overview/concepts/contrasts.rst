.. _contrasts:

Contrast-specific vs. contrast-agnostic
#######################################

Contrast-specific segmentation tools
------------------------------------

Historically, SCT's automatic spinal cord segmentation tools (:ref:`sct_propseg`, :ref:`sct_deepseg_sc`) have been limited to specific contrasts (`T1-weighted <https://radiopaedia.org/articles/t1-weighted-image>`__, `T2 weighted <https://radiopaedia.org/articles/t2-weighted-image>`__, `T2*-weighted <https://radiopaedia.org/articles/t2-weighted-image>`__, and `diffusion weighted images (DWI) <https://radiopaedia.org/articles/diffusion-weighted-imaging-2?lang=us>`__.). Each contrast required its own dedicated method, and users have had to specify the contrast when running each tool.

Additionally, for image data that uses a contrast not listed above, it has been necessary to select the closest visual match among the available options. For example, fMRI images have bright cerebrospinal fluid (CSF) regions and dark spinal cord regions, so the T2 contrast option would be an appropriate choice.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/image_contrasts.png
  :align: center
  :figwidth: 75%

  Image contrasts and their corresponding ``-c`` arguments

However, there is a caveat: contrast-specific tools (as described above) can provide less consistent morphometrics measures across MRI contrasts. For example, if we compute the spinal cord cross-sectional area averaged at C2-C3 vertebral levels and compare it across contrasts, we find a significant amount of variability.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/csa_contrast-specific.png
  :align: center
  :figwidth: 75%

  CSA variability across contrast-specific tools

Contrast-agnostic tools (2024 and onwards)
------------------------------------------

To combat these issues, SCT has increasingly moved towards developing contrast-agnostic segmentation tools. To do this, we train a single model using many contrasts. We also use “soft” ground truth images (averaged across contrasts) during training to ensure that the same ground truth for all contrasts is used.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/contrast-agnostic-development.png
  :align: center
  :figwidth: 75%

  Process of creating a contrast-agnostic tool

In our research, we have found that contrast-agnostic tools achieve their goal of providing more consistent morphometrics measures across MRI contrasts. For further details, please refer to `Bédard et al. (2025) <https://doi.org/10.1016/j.media.2025.103473>`__.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/csa-contrast-agnostic.png
  :align: center
  :figwidth: 75%

  CSA consistency in contrast-agnostic tools

All new contrast-agnostic deep learning models are released through the :ref:`sct_deepseg` command-line tool.

