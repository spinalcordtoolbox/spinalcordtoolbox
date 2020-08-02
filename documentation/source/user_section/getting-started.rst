.. _getting-started:

Getting Started
###############

To get started using SCT, you may take a look at the `Batch Processing
Example`_, or follow the longer `Courses`_ materials.

.. contents::
   :local:
..


Batch Processing Example
************************

The best way to learn how to use SCT is to look at the example `batch_processing
<https://github.com/neuropoly/spinalcordtoolbox/blob/master/batch_processing.sh>`_ script. This script performs a
typical analysis of multi-parametric MRI data, including cross-sectional area measurements, magnetization transfer,
diffusion tensor imaging metrics computation and extraction within specific tracts, and functional MRI pre-processing.
This script will download an example dataset containing T1w, T2w, T2\*w, MT, DTI and fMRI data.

To launch the script run:

.. code:: sh

  $SCT_DIR/batch_processing.sh

While the script is running, we encourage you to understand the meaning of each command line that is listed in the
script. Comments are here to help justify some choices of parameters. If you have any question, please do not
hesitate to ask for :ref:`support`.

The script source is shown below:

.. literalinclude:: ../../../batch_processing.sh
   :language: sh

If you would like to get more examples about what SCT can do, please visit the `sct-pipeline repository
<https://github.com/sct-pipeline/>`_. Each repository is a pipeline dedicated to a specific research project.


Courses
*******

We organize **free** SCT courses, each year after the ISMRM conference.
If you’d like to be added to the mailing list, please send an email to
spinalcordtoolbox@gmail.com. The past courses handouts are listed
below:

-  `SCT course (v4.2.1), London, 2020-01-21`_ \| `Video recording`_
-  `SCT course (v4.0.0), Beijing, 2019-08-02`_ \| `Slides with Chinese
   translation`_
-  `SCT course (v4.0.0_beta.4), London, 2019-01-22`_
-  `SCT course (v3.2.2), Paris, 2018-06-12`_
-  `SCT course (v3.0.3), Honolulu, 2017-04-28`_
-  `SCT course (v3.0_beta14), Geneva, 2016-06-28`_
-  `SCT course (v3.0_beta9), Singapore, 2016-05-13`_
-  `SCT course (v3.0_beta1), Montreal, 2016-04-19`_
-  `SCT Hands-on Workshop (v2.0.4), Toronto, 2015-06-15`_

.. _SCT course (v4.2.1), London, 2020-01-21: https://www.icloud.com/keynote/0th8lcatyVPkM_W14zpjynr5g#SCT%5FCourse%5F20200121
.. _Video recording: https://www.youtube.com/watch?v=whbtjYNtHko&feature=youtu.be
.. _SCT course (v4.0.0), Beijing, 2019-08-02: https://osf.io/arfv7/
.. _Slides with Chinese translation: https://osf.io/hnmr2/
.. _SCT course (v4.0.0_beta.4), London, 2019-01-22: https://osf.io/gvs6f/
.. _SCT course (v3.2.2), Paris, 2018-06-12: https://osf.io/386h7/
.. _SCT course (v3.0.3), Honolulu, 2017-04-28: https://osf.io/fvnjq/
.. _SCT course (v3.0_beta14), Geneva, 2016-06-28: https://sourceforge.net/p/spinalcordtoolbox/wiki/Home/attachment/SCT_Course_20160628.pdf
.. _SCT course (v3.0_beta9), Singapore, 2016-05-13: https://drive.google.com/file/d/0Bx3A13n3Q_EAa3NQYjBOWjhjZm8/view?usp=sharing
.. _SCT course (v3.0_beta1), Montreal, 2016-04-19: https://drive.google.com/file/d/0Bx3A13n3Q_EAenltM2ZvZUNEdjQ/view?usp=sharing
.. _SCT Hands-on Workshop (v2.0.4), Toronto, 2015-06-15: https://www.dropbox.com/s/f9887yrbkcfujn9/sct_handsOn_20150605.pdf?dl=0
