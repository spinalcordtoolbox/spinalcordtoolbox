#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for the batch_processing.sh script

import os
import pytest


@pytest.mark.skipif(not os.path.isfile('batch_processing_results.zip'), reason="Run only for batch processing CI job")
def test_skipif_decorator():
    # TODO: Compare batch_processing.sh values within a certain tolerance. This is just to check "skipif"
    pass
