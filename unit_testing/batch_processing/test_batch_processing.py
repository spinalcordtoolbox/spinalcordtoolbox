#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests to validate the reults of the batch_processing.sh script

import os
import pathlib
import csv

import pytest

from spinalcordtoolbox.utils.sys import sct_dir_local_path

SCT_DIR = pathlib.Path(sct_dir_local_path())
CACHE_DIR = SCT_DIR / "unit_testing" / "batch_processing" / "cached_results"
OUTPUT_DIR = SCT_DIR / "sct_example_data" 

# TODO: We can and should be verifying more results produced by this pipeline, but which values?
TESTED_VALUES = [("t2/csa_c2c3.csv", -1, "MEAN(area)"),
                 ("t2s/csa_gm.csv", -1, "MEAN(area)"),
                 ("t2s/csa_wm.csv", -1, "MEAN(area)"),
                 ("mt/mtr_in_wm.csv", -1, "MAP()"),
                 ("dmri/fa_in_cst.csv", -1, "WA()"),
                 ("dmri/fa_in_cst.csv", -2, "WA()")]


def get_csv_value(csv_filepath, row, column):
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        value = [row for row in reader][row][column]
    return value


@pytest.mark.skipif(not os.getenv('BATCH_PROCESSING_CI_JOB'), reason="Run only for batch processing CI job")
@pytest.mark.parametrize("csv_filepath,row,column", TESTED_VALUES)
def test_batch_processing_results(csv_filepath, row, column):
    """Ensure that new batch_processing.sh results are approximately equal to the cached baseline results."""
    csv_output = OUTPUT_DIR / csv_filepath
    csv_cached = CACHE_DIR / csv_filepath
    assert csv_cached.is_file(), f"{csv_cached} not present. Please check the SCT installation."
    assert csv_output.is_file(), f"{csv_output} not present. Was batch_processing.sh run beforehand?"
    assert (get_csv_value(csv_output, row, column) == pytest.approx(  # Default rel_tolerance: 1e-6
            get_csv_value(csv_cached, row, column)))
