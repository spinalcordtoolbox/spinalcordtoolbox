SCT Concepts
############


Template / Atlas
****************

- MNI-Poly-AMU - Template of the spinal cord including probabilistic
  white and gray matter.

  https://sourceforge.net/p/spinalcordtoolbox/wiki/MNI-Poly-AMU/

  .. TODO

- Spinal levels - Spinal levels of the spinal cord.

  https://sourceforge.net/p/spinalcordtoolbox/wiki/Spinal_levels/

  .. TODO

- White Matter atlas - Atlas of white matter spinal pathways.

  https://sourceforge.net/p/spinalcordtoolbox/wiki/White%20Matter%20atlas/

  .. TODO

Tips & Tricks
*************

- registration to a metric - Improve the registration of a template to
  a metric image by taking into account the spinal cord's internal
  structure

  https://sourceforge.net/p/spinalcordtoolbox/wiki/register_to_metric/

  .. TODO


- registration tricks - informations of the parameters available in the registration functions, and how to use them.

  https://sourceforge.net/p/spinalcordtoolbox/wiki/registration_tricks/

  .. TODO


Segmentation of GM/WM
*********************


.. _qc:

Quality Control
***************

Some SCT tools can generate Quality Control (QC) reports.
These reports consist in “appendable” HTML files, containing a table
of entries and allowing to show, for each entry, animated images
(background with overlay on and off).

To generate a QC report, add the `-qc` command-line argument,
with the location (folder, to be created by the SCT tool),
where the QC files should be generated.

