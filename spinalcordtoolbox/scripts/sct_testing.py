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
import sys

from spinalcordtoolbox import __sct_dir__


def main(argv):
    # Treat `sct_testing` as an alias to `pytest`. If no arguments are passed,
    # simply point pytest at the repo (to avoid looking for tests in the pwd).
    sys.exit(pytest.main(argv if sys.argv[1:] else [__sct_dir__]))


if __name__ == "__main__":
    main(sys.argv[1:])

