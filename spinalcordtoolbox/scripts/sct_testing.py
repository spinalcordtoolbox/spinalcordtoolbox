#!/usr/bin/env python
"""
A wrapper for pytest to provide backwards compatability for `sct_testing`.

Context: SCT previously provided its own custom testing framework using a
         script called `sct_testing`. This was replaced with `pytest`, however
         sct_testing is likely still remembered by users, and may still be in
         use in scripts. So, this script provides a way to use `sct_testing`
         to run the pytest test suite.
"""

import pytest
import sys

from spinalcordtoolbox import __sct_dir__

if __name__ == "__main__":
    # Treat `sct_testing` as an alias to `pytest`. If no arguments are passed,
    # simply point pytest at the repo (to avoid looking for tests in the pwd).
    sys.exit(pytest.main(sys.argv[1:] if sys.argv[1:] else [__sct_dir__]))
