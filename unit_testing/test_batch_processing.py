#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for the batch_processing.sh script

import os
import zipfile
import pathlib

import pytest

from spinalcordtoolbox.utils.sys import sct_dir_local_path


@pytest.fixture()
def unzipped_batch_processing_results(tmp_path):
    filepath = sct_dir_local_path() / pathlib.Path(os.getenv('BATCH_PROCESSING_ZIP_FILEPATH'))
    assert filepath.suffix == ".zip"
    output_dir = tmp_path / filepath.stem  # Use filename as dirname ('.stem' strips extension)

    if not output_dir.is_dir():
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(path=output_dir)

    return output_dir


@pytest.mark.skipif(not os.getenv('BATCH_PROCESSING_ZIP_FILEPATH'),  # This should be set by the CI workflow file
                    reason="Run only for batch processing CI job")
def test_skipif_decorator(unzipped_batch_processing_results):
    # TODO: Compare batch_processing.sh values within a certain tolerance.

    assert unzipped_batch_processing_results.stem == "batch_processing_results"
