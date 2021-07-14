.. _python-api:

Python API
##########

.. admonition:: Warning

   Because SCT is not currently installable via pip, we strongly recommend against using SCT's Python API in your code. Instead, we ask that you stick with SCT's :ref:`command-line-tools`.

   This page is maintained primarily for internal use by SCT developers. If you have arrived at this page, please let us know so that we can update any links to this page.


.. contents::
..


Segmentation
************

Spinal Cord Segmentation
========================

deepseg
+++++++

.. automodule:: spinalcordtoolbox.deepseg.models
   :members:


deepseg_sc
++++++++++

.. automodule:: spinalcordtoolbox.deepseg_sc.core
   :members:

.. automodule:: spinalcordtoolbox.deepseg_sc.postprocessing
   :members:


GM/WM Segmentation
==================

deepseg_gm
++++++++++

.. automodule:: spinalcordtoolbox.deepseg_gm.deepseg_gm
   :members:

.. automodule:: spinalcordtoolbox.deepseg_gm.model
   :members:


MS Lesion Segmentation
======================

deepseg_lesion
++++++++++++++

.. automodule:: spinalcordtoolbox.deepseg_lesion.core
   :members:


Centerline
**********

spinalcordtoolbox.centerline
============================

.. automodule:: spinalcordtoolbox.centerline.core
   :members:

.. automodule:: spinalcordtoolbox.centerline.curve_fitting
   :members:

.. automodule:: spinalcordtoolbox.centerline.optic
   :members:


Segmentation Processing
***********************

spinalcord.process_seg
=======================

.. automodule:: spinalcordtoolbox.process_seg
   :members:

spinalcord.straightening
========================

.. automodule:: spinalcordtoolbox.straightening
   :members:


QMRI
****

spinalcordtoolbox.qmri
======================

.. automodule:: spinalcordtoolbox.qmri.mt
   :members:


Quality Control
***************

The modules spinalcordtoolbox.reports.qc_ and
spinalcordtoolbox.reports.slice_ are used to generate :ref:`qc` reports.


spinalcordtoolbox.reports.qc
============================

.. automodule:: spinalcordtoolbox.reports.qc
   :members:

spinalcordtoolbox.reports.slice
===============================

.. automodule:: spinalcordtoolbox.reports.slice
   :members:


Vertebrae Labeling
******************

spinalcordtoolbox.vertebrae
===========================

.. automodule:: spinalcordtoolbox.vertebrae.core
   :members:

.. automodule:: spinalcordtoolbox.vertebrae.detect_c2c3
   :members:


Metrics Aggregation
*******************

spinalcordtoolbox.aggregate_slicewise
=====================================

.. automodule:: spinalcordtoolbox.aggregate_slicewise
   :members:


Image Manipulation
******************

spinalcordtoolbox.cropping
==========================

.. automodule:: spinalcordtoolbox.cropping
   :members:

spinalcordtoolbox.image
=======================

.. automodule:: spinalcordtoolbox.image
   :members:

spinalcordtoolbox.resampling
============================

.. automodule:: spinalcordtoolbox.resampling
   :members:

Image Labelling
***************

spinalcordtoolbox.labels
========================

.. automodule:: spinalcordtoolbox.labels
   :members:

Spinal Cord Flattening
**********************

spinalcordtoolbox.flattening
============================

.. automodule:: spinalcordtoolbox.flattening
   :members:


Motion Correction
*****************

spinalcordtoolbox.moco
======================

.. automodule:: spinalcordtoolbox.moco
   :members:


Helpers and Utilities
*********************

spinalcordtoolbox.math
======================

.. automodule:: spinalcordtoolbox.math
   :members:

spinalcordtoolbox.metadata
==========================

.. automodule:: spinalcordtoolbox.metadata
   :members:

spinalcordtoolbox.types
=======================

.. automodule:: spinalcordtoolbox.types
   :members:

spinalcordtoolbox.template
==========================

.. automodule:: spinalcordtoolbox.template
   :members:

