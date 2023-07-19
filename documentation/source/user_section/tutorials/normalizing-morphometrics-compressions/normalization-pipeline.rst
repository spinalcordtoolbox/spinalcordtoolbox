.. _normalization-pipeline:

Normalization pipeline (Averaged across vertebral levels)
######################################

Morphometrics measures are ouput from ``sct_process_segmentation`` and include cross-sectional area, anterior-posterior diameter, right-left diameter, eccentricity and solidity.
Metrics are normalized using the non-compressed levels above and below the compression site
using the following equation:

    ratio = (1 - mi/((ma+mb)/2))

Where mi: metric at the compression level, ma: metric above the compression level, mb:
metric below the compression level.

Additionally, if the "-normalize-hc" flag is used, metrics are normalized using a database
built from healthy control subjects. This database uses the PAM50 template as an anatomical
reference system.


# TODO put figure here



Reference: Miyanji F, Furlan JC, Aarabi B, Arnold PM, Fehlings MG. Acute cervical traumatic
spinal cord injury: MR imaging findings correlated with neurologic outcome--prospective
study with 100 consecutive patients. Radiology 2007;243(3):820-827.
