SCT Concepts
############


This section documents some SCT concepts and other useful things to know
when using it.

.. contents::
   :local:
..

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


Segmentation of the Spinal Cord
*******************************

SCT provides several tools to perform SC segmentation:

- :ref:`sct_propseg`
- :ref:`sct_deepseg_sc`

The latter one, using a deep learning model, is giving the best results on most
cases, but is not configurable.

The former one is the fallback tool. It has lots of options that can
be useful when segmenting tricky volumes.
You may use it if :ref:`sct_deepseg_sc` is performing worse results
than :ref:`sct_propseg` with default parameters.

.. TODO additional information, performance info, paper

Segmentation of GM/WM
*********************

SCT provides several tools to perform GM/WM segmentation:

- :ref:`sct_segment_graymatter`
- :ref:`sct_deepseg_gm`

The latter one, using a deep learning model, is giving the best results on most
cases.

The former one is the fallback tool.

.. TODO additional information, performance info, paper


Temporary Directories
*********************

Many SCT commands will create in temporary directories to operate,
and there is an option to avoid removing temporary directories, to be
used for troubleshooting purposes.

If you don't know where your temporary directory is located, you can
look at:
https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir


Matlab Integration on Mac
*************************

Matlab took the liberty of setting ``DYLD_LIBRARY_PATH`` and in order
for SCT to run, you have to run:

.. code:: matlab

   setenv('DYLD_LIBRARY_PATH', '');

Prior to running SCT commands. See
 https://github.com/neuropoly/spinalcordtoolbox/issues/405


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

