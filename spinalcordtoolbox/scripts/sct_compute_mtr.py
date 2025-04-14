#!/usr/bin/env python
#
# Compute magnetization transfer ratio (MTR)
#
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
import os
from typing import Sequence

from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser, display_viewer_syntax
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.qmri.mt import compute_mtr


def get_parser():
    parser = SCTArgumentParser(
        description='Compute magnetization transfer ratio (MTR). Output is given in percentage.'
    )

    mandatory = parser.mandatory_arggroup
    mandatory.add_argument(
        '-mt0',
        help='Image without MT pulse (MT0)',
        metavar=Metavar.float,
    )
    mandatory.add_argument(
        '-mt1',
        help='Image with MT pulse (MT1)',
        metavar=Metavar.float,
    )

    optional = parser.optional_arggroup
    optional.add_argument(
        "-thr",
        type=float,
        help="Threshold to clip MTR output values in case of division by small number. This implies that the output image "
             "range will be [-thr, +thr]. Default: `100`.",
        default=100
    )
    optional.add_argument(
        '-o',
        help='Path to output file.',
        metavar=Metavar.str,
        default=os.path.join('.', 'mtr.nii.gz')
    )

    # Arguments which implement shared functionality
    parser.add_common_args()

    return parser


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    fname_mtr = arguments.o

    # compute MTR
    printv('\nCompute MTR...', verbose)
    nii_mtr = compute_mtr(nii_mt1=Image(arguments.mt1), nii_mt0=Image(arguments.mt0), threshold_mtr=arguments.thr)

    # save MTR file
    nii_mtr.save(fname_mtr, dtype='float32')

    display_viewer_syntax([arguments.mt0, arguments.mt1, fname_mtr], verbose=verbose)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
