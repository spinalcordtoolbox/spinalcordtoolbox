#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests to validate the reults of the batch_processing.sh script

import os
import pathlib
import csv

import pytest

from spinalcordtoolbox.utils.sys import sct_dir_local_path

SCT_DIR = pathlib.Path(sct_dir_local_path())
CACHE_DIR = SCT_DIR / "testing" / "batch_processing" / "cached_results"
OUTPUT_DIR = SCT_DIR / "sct_example_data"

# TODO: We can and should be verifying more results produced by this pipeline, but which values?
TESTED_VALUES = [("t2/csa_c2c3.csv", 0, "MEAN(area)"),
                 ("t2s/csa_gm.csv", 3, "MEAN(area)"),
                 ("t2s/csa_wm.csv", 3, "MEAN(area)"),
                 ("mt/mtr_in_wm.csv", 0, "MAP()"),
                 ("dmri/fa_in_cst.csv", 0, "WA()"),
                 ("dmri/fa_in_cst.csv", 1, "WA()")]


def get_csv_float_value(csv_filepath, row, column):
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        value = [row for row in reader][row][column]
    return float(value)


@pytest.mark.skipif(not os.getenv('TEST_BATCH_PROCESSING'), reason="Run only for batch processing CI job")
@pytest.mark.parametrize("csv_filepath,row,column", TESTED_VALUES)
def test_batch_processing_results(csv_filepath, row, column):
    """Ensure that new batch_processing.sh results are approximately equal to the cached baseline results."""
    csv_output = OUTPUT_DIR / csv_filepath
    csv_cached = CACHE_DIR / csv_filepath
    assert csv_cached.is_file(), f"{csv_cached} not present. Please check the SCT installation."
    assert csv_output.is_file(), f"{csv_output} not present. Was batch_processing.sh run beforehand?"
    assert (get_csv_float_value(csv_output, row, column) == pytest.approx(  # Default rel_tolerance: 1e-6
            get_csv_float_value(csv_cached, row, column)))


def display_batch_processing_results():
    """Utility function to avoid having to use 'awk' in the batch_processing.sh script."""
    max_len = len(max([v[0] for v in TESTED_VALUES], key=len))
    for csv_filepath, row, column in TESTED_VALUES:
        value = get_csv_float_value(OUTPUT_DIR/csv_filepath, row, column)
        print(f"{f'{csv_filepath}:':<{max_len+2}} {value:<18} [Row {row+1}, {column}]")


if __name__ == "__main__":
    display_batch_processing_results()
