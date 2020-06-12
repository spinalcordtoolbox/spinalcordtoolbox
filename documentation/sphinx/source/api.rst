.. _python-api:

Python API
##########

.. admonition:: Note

   The Python API is not stable yet, so be prepared to update your
   code with SCT updates.


.. contents::
..


Segmentation API
****************

Spinal Cord Segmentation API
============================

deepseg API
+++++++++++

.. automodule:: spinalcordtoolbox.deepseg.core
   :members:

.. automodule:: spinalcordtoolbox.deepseg.models
   :members:


deepseg_sc API
++++++++++++++

.. automodule:: spinalcordtoolbox.deepseg_sc.cnn_models
   :members:

.. automodule:: spinalcordtoolbox.deepseg_sc.cnn_models_3d
   :members:

.. automodule:: spinalcordtoolbox.deepseg_sc.core
   :members:

.. automodule:: spinalcordtoolbox.deepseg_sc.postprocessing
   :members:


GM/WM Segmentation API
======================

deepseg_gm API
++++++++++++++

.. automodule:: spinalcordtoolbox.deepseg_gm.deepseg_gm
   :members:

.. automodule:: spinalcordtoolbox.deepseg_gm.model
   :members:


MS Lesion Segmentation API
==========================

deepseg_lesion API
++++++++++++++++++

.. automodule:: spinalcordtoolbox.deepseg_lesion.core
   :members:


Centerline API
**************

spinalcordtoolbox.centerline
============================

.. automodule:: spinalcordtoolbox.centerline.core
   :members:

.. automodule:: spinalcordtoolbox.centerline.curve_fitting
   :members:

.. automodule:: spinalcordtoolbox.centerline.nurbs
   :members:

.. automodule:: spinalcordtoolbox.centerline.optic
   :members:


QMRI API
********

spinalcordtoolbox.qmri
======================

.. automodule:: spinalcordtoolbox.qmri.mt
   :members:


Quality Control API
*******************

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


Vertebrae Labeling API
**********************

spinalcordtoolbox.vertebrae
===========================

.. automodule:: spinalcordtoolbox.vertebrae.core
   :members:

.. automodule:: spinalcordtoolbox.vertebrae.detect_c2c3
   :members:
