.. _normalization-pipeline:

Normalization pipeline
######################

Morphometrics measures are output from ``sct_process_segmentation`` and include cross-sectional area, anterior-posterior diameter, right-left diameter, eccentricity and solidity.
Metrics are normalized using the non-compressed levels above and below the compression site
using the equation in the figure below. 

Additionally, if the "-normalize-hc" flag is used, metrics are normalized using a database
built from healthy control subjects. This database uses the PAM50 template as an anatomical
reference system.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/sb/4158-add-tutorial-sct-compute-compression/spinalcord-compresssion-norm/normalized_metrics_hc_figure.png
   :align: center

   Normalized metric ratio using a database of healthy controls.

*Legend:*
- ``mi`` : metric at the compression level.
- ``ma`` : metric above the compression levels.
- ``mb`` : metric below the compression level.

Here are the differents shape morphometric of the spinal cord that the ratio can be computed and normalized woth helthy controls. 

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/sb/4158-add-tutorial-sct-compute-compression/spinalcord-compresssion-norm/morphometric_measures_figure.png
   :align: center

   Morphometrics measures.

Reference: Miyanji F, Furlan JC, Aarabi B, Arnold PM, Fehlings MG. Acute cervical traumatic
spinal cord injury: MR imaging findings correlated with neurologic outcome--prospective
study with 100 consecutive patients. Radiology 2007;243(3):820-827.
