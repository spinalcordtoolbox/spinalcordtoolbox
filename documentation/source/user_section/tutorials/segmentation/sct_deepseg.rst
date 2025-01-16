Specialized segmentation models: ``sct_deepseg``
################################################

The :ref:`sct_deepseg` script is separate from :ref:`sct_deepseg_sc`, and provides access to specialized models created by created with different frameworks (`ivadomed`: https://ivadomed.org/, `nnUNet`: https://github.com/MIC-DKFZ/nnUNet, `monai`: https://monai.io>). These specialized models focus on tasks outside of basic spinal cord segmentation. In recent versions, new models for the segmentation of spinal tumors, human multi-class SC & GM at 7T, MS lesion segmentation on MP2RAGE, mouse SC, etc. have been added.

You can list the available tasks by running :ref:`sct_deepseg` ``-h``, or you view detailed descriptions of each task by running :ref:`sct_deepseg` ``-list-tasks``.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord-segmentation/sct_deepseg_models.png
   :align: center

   Sample models. (**Tumor segmentation**: `Lemay et al. NeuroImage Clinical (2021) <https://pubmed.ncbi.nlm.nih.gov/34352654/>`_,
   **WM/GM cord segmentation at 7T**: `Laines Medina et al. arXiv (2021) <https://arxiv.org/pdf/2110.06516.pdf>`_,
   **MP2RAGE SC and lesion segmentation**: `Cohen-Adad et al. Zenodo release (2023) <https://doi.org/10.5281/zenodo.8376754>`_)

