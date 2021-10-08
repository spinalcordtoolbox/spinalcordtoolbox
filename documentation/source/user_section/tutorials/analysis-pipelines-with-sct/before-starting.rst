Before starting this tutorial
#############################

1. Read through the following page to familiarize yourself with key SCT concepts:

   * :ref:`inspecting-your-results`: After some steps in this tutorial, instructions are provided to open the output images using :ref:`Quality Control (QC) <qc>` reports and :ref:`fsleyes-instructions`.

2. Make sure that you have the following files in your working directory:

   * ``multi_subject/data/`` : A folder containing 3 subjects, each with six T2 and MT data files in total.
   * ``multi_subject/process_data.sh`` : A script containing SCT commands to process T2 and MT data.
   * ``multi_subject/config.yml`` : A config file containing input arguments for ``sct_run_batch``.

   You can get these files by downloading :sct_tutorial_data:`data_batch-processing-of-subjects.zip`.


3. Open a terminal and navigate to the ``multi_subject/`` directory:

.. code:: sh

   cd {PATH_TO_DOWNLOADED_DATA}/multi_subject