#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests to validate the reults of the batch_processing.sh script

import os
import pathlib
import csv

import pytest

from spinalcordtoolbox.utils.sys import sct_dir_local_path


@pytest.mark.skipif(
    not os.getenv('BATCH_PROCESSING_CI_JOB'),  # This environment variable should be set by the CI workflow file
    reason="Run only for batch processing CI job"
)
# The parametrization below checks only 6 values (one from each csv file -- same as actual batch_processing.sh)
# TODO: We can and should be verifying more results produced by this pipeline, but which values?
@pytest.mark.parametrize("csv_filepath,row,column",
                         [("t2/csa_c2c3.csv", -1, "MEAN(area)"),
                          ("t2s/csa_gm.csv", -1, "MEAN(area)"),
                          ("t2s/csa_wm.csv", -1, "MEAN(area)"),
                          ("mt/mtr_in_wm.csv", -1, "MAP()"),
                          ("dmri/fa_in_cst.csv", -1, "WA()"),
                          ("dmri/fa_in_cst.csv", -2, "WA()")])
def test_batch_processing_results(csv_filepath, row, column):
    """Ensure that new batch_processing.sh results are approximately equal to the cached baseline results."""
    sct_dir = pathlib.Path(sct_dir_local_path())
    csv_filepath_old = sct_dir / "unit_testing/batch_processing/cached_results" / csv_filepath
    csv_filepath_new = sct_dir / "sct_example_data" / csv_filepath
    assert csv_filepath_old.is_file(), f"{csv_filepath_old} not present. Please check the SCT installation."
    assert csv_filepath_new.is_file(), f"{csv_filepath_new} not present. Was batch_processing.sh run beforehand?"

    with open(csv_filepath_old, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        metric_value_old = float([row for row in reader][row][column])  # Row/position varies depending on metric

    with open(csv_filepath_new, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        metric_value_new = float([row for row in reader][row][column])  # Row/position varies depending on metric

    assert metric_value_new == pytest.approx(metric_value_old)  # Default rel_tolerance: 1e-6
