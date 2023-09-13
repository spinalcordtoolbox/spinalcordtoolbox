.. _normalization-pipeline:

Normalization pipeline
######################

Morphometrics measures are output from ``sct_process_segmentation`` and include cross-sectional area (area), anteroposterior diameter (diameter_AP), right-left diameter (diameter_RL), eccentricity and solidity.
Metrics are normalized using the non-compressed levels above and below the compression site
using the equation in the figure below.

Additionally, if the ``-normalize-hc 1`` flag is used, metrics are normalized using a database
built from adult healthy participants. This database uses the PAM50 template as an anatomical
reference system.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/sb/4158-add-tutorial-sct-compute-compression/spinalcord-compresssion-norm/normalized_metrics_hc_figure.png
   :align: center

   Normalized metric ratio using a database of healthy controls.


:Legend:
   - ``mi`` : metric at the compression level.
   - ``ma`` : metric above the compression levels.
   - ``mb`` : metric below the compression level.

