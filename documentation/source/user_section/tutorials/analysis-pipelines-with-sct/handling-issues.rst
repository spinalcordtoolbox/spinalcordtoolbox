What if things go wrong?
########################

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/batch-processing-of-subjects/segmentation-issue.png
   :align: center

If you spot any issues during QC, e.g., error during segmentation, you can correct the issue (manually or by adjusting SCT parameters), and then re-run the processing.

Manual corrections
==================

The script ``manual_correction.py`` will be used to conveniently correct the segmentations for the failed subjects.

Setting up the environment
--------------------------

As a prerequisite, make sure you have an image editor installed (in this case: FSLeyes):

.. code:: sh

    # Next, make sure that your image viewer is callable from the
    # Terminal. In the course we use FSLeyes but you can do the same
    # with another viewer (eg: ITKsnap).
    fsleyes --version

Next, download the manual correction script and check its options:

.. code:: sh

    # First, download the manual correction script into a folder called `manual-correction`
    sct_download_data -d manual-correction -o manual-correction

    # Then you can look at the files in the folder
    ls manual-correction

    # Make sure to access the version of Python inside the SCT installation
    sct_python="$SCT_DIR/python/envs/venv_sct/bin/python"

    # Now we can check the options of the manual correction script (using the help flag -h)
    $sct_python manual-correction/manual_correction.py -h

.. note::

    By typing ``$SCT_DIR/python/envs/venv_sct/bin/python`` instead of just ``python``, we can directly access the version of Python that lives inside the SCT installation. This is useful because the script depends on packages that are installed with SCT, but that may not be installed in your default Python environment.

Running the correction script
-----------------------------

Once your setup is ready, you can run the manual correction script.

.. code:: sh

    # Run the manual correction script
    $sct_python manual-correction/manual_correction.py -config qc_fail.yml -path-img output/data_processed/ -path-label output/data_processed -path-out data/derivatives/labels
    # Check the files output to the ‘derivatives/’ folder
    tree data/derivatives

The output segmentations will be located under the derivatives/labels/ folder, according to BIDS convention. Also, JSON sidecar files will be created with name and date, for traceability.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/batch-processing-of-subjects/correction_outputs.png
   :align: center

Re-run analysis (with corrections)
==================================

``process_data.sh`` contains a number of convenience functions that will prioritize any file with suffix ``-manual`` (if it exists) and use that file instead of re-computing the segmentation. That way, you can manually process outlier subjects without interrupting the automated processing for the dataset as a whole.

.. code:: sh

    sct_run_batch -script process_data.sh -path-data data/ -path-output output_correction -jobs 3

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/batch-processing-of-subjects/demonstrating_usage_of_manually_corrected_image.png
   :align: center