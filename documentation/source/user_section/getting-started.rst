.. _getting-started:

Getting Started
###############

To get started using SCT, you may wish to read our 2024 Review "`Reproducible Spinal Cord Quantitative MRI Analysis with the Spinal Cord Toolbox <https://doi.org/10.2463/mrms.rev.2023-0159>`__".

For a more hands-on approach, you can take a look at the `Batch Processing Example`_, or follow the longer :ref:`tutorials` materials.


Batch Processing Example
************************

The best way to learn how to use SCT is to look at the example `batch_processing <https://github.com/spinalcordtoolbox/spinalcordtoolbox/blob/master/batch_processing.sh>`_ script.
This script performs a typical analysis of multi-parametric MRI data, including cross-sectional area measurements, magnetization transfer, diffusion tensor imaging metrics computation and extraction within specific tracts, and functional MRI pre-processing. This script will download an example dataset containing T1w, T2w, T2\*w, MT, DTI and fMRI data.

To launch the script run:

.. code:: sh

  $SCT_DIR/batch_processing.sh

While the script is running, we encourage you to understand the meaning of each command line that is listed in the script. Comments are here to help justify some choices of parameters. If you have any question, please do not hesitate to ask for :ref:`help`.

The script source is shown below:

.. literalinclude:: ../../../batch_processing.sh
   :language: sh

If you would like to get more examples about what SCT can do, please visit the `sct-pipeline repository <https://github.com/sct-pipeline/>`_. Each repository is a pipeline dedicated to a specific research project.
