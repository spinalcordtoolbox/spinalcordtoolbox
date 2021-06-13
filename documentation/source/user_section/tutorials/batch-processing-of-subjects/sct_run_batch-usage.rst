Running the sample script (``process_data.sh``) using ``sct_run_batch``
#######################################################################

In your working directory, you should have the batch processing script, a config file, and a data directory containing three subjects. You can double-check this using the ``ls`` command:

.. code:: sh

   ls
   # Output should be: data/  config.yml  process_data.sh  README.txt
   ls data
   # Output should be: sub-01  sub-03  sub-05

In order to apply the ``process_data.sh`` script to the multi-subject dataset, we can use an SCT tool called ``sct_run_batch``. This command loops across each subject directory and launches the script individually for each subject.

Running this command will take 10-30m depending on the capabilities of your processor.

.. code::

   sct_run_batch -script ./process_data.sh -config ./config.yml

:Input arguments:
   - ``-script`` : The path to the script you wish to run. This script should be written to accept a single argument (a path to the subject folder). That way, ``sct_run_batch`` can launch the script once for each subject, passing along each subject folder as an argument to the script.
   - ``-config`` : A config file containing input arguments for ``sct_run_batch``. Inside this file, you will find:

     - ``path-data`` : Path to the folder containing the BIDS dataset. ``sct_run_batch`` will look for subject folders inside, so that it can loop across subjects.
     - ``subject-prefix`` : The prefix that each subject folder has.
     - ``path-output`` : Path to save the output to. This path is what determines the output directories for results, QC reports, and logs.
     - ``path-segmanual`` : Path to the folder containing manually-corrected segmentations.
     - ``jobs`` : Number of jobs for parallel processing.
     - ``itk-threads`` : Number of jobs for ANTs routines.

:Output files/folders:
   - ``./output/results/`` : A folder containing the processed data, as well as CSV files containing computed metrics (CSA, and MTR in WM). This is equivalent to the ``derivatives/`` folder of BIDS-compliant datasets.
   - ``./output/qc/`` : A folder containing the Quality Control (QC) reports for the commands in the pipeline.
   - ``./output/log/`` : A folder containing the log files for each subject. This is where you would go to troubleshoot any errors or issues that occur during the processing.

.. note::

   We strongly recommend using a config file as shown here (rather than passing arguments directly to ``sct_run_batch``) because it helps with reproducibility.

   To learn about the full list of options that are available to tweak in your config file, run ``sct_run_batch -h``.