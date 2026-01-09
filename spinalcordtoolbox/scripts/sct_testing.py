#!/usr/bin/env python
#
# A wrapper for pytest to provide backwards compatability for `sct_testing`
#
# Context: SCT previously provided its own custom testing framework using a
#          script called `sct_testing`. This was replaced with `pytest`, however
#          sct_testing is likely still remembered by users, and may still be in
#          use in scripts. So, this script provides a way to use `sct_testing`
#          to run the pytest test suite.
#
# Copyright (c) 2022 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import pytest
import os
import sys

from spinalcordtoolbox.utils.sys import __sct_dir__

if __name__ == "__main__":
    # Treat `sct_testing` as an alias to `pytest`. If no arguments are passed,
    # simply point pytest at the repo (to avoid looking for tests in the pwd).
    argv = sys.argv[1:] if sys.argv[1:] else [__sct_dir__]

    # For SCT's CI, add extra logging to a log file.
    # We don't specify this in `setup.cfg` because we don't want to clutter
    # users' working directories
    if "GITHUB_ACTIONS" in os.environ and "PYTEST_LOG_FILENAME" in os.environ:
        log_file = os.path.join(__sct_dir__, os.environ["PYTEST_LOG_FILENAME"])
        argv += [
            f"--log-file={log_file}",
            "--log-file-level=DEBUG",
        ]

    sys.exit(pytest.main(argv))
