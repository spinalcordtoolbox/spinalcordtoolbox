#!/usr/bin/env python
#
# Compute flow from velocity encoding (VENC) sequence, based on the MRI phase image. More details in: https://mriquestions.com/what-is-venc.html
#
# Copyright (c) 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
# License: see the file LICENSE

import sys
import textwrap
from typing import Sequence

from spinalcordtoolbox.image import Image, add_suffix
from spinalcordtoolbox.qmri import flow
from spinalcordtoolbox.utils.shell import Metavar, SCTArgumentParser, display_viewer_syntax
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel


def get_parser():
    parser = SCTArgumentParser(
        description=("""\
Compute velocity from the MRI phase image for velocity encoding (VENC) sequences.
More details in: https://mriquestions.com/what-is-venc.html

Further features are planned for this script. Please refer to this issue for more info:
  - https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/4298
        """)
    )

    mandatory = parser.add_argument_group("MANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-i",
        required=True,
        help="4D phase image. The 4th dimension should be the velocity encoding (VENC) in cm/s.",
        metavar=Metavar.file,
    )
    mandatory.add_argument(
        "-venc",
        required=True,
        help="Maximum velocity encoding (VENC) in cm/s.",
        metavar=Metavar.float,
    )

    optional = parser.add_argument_group("OPTIONAL ARGUMENTS")
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        help='Output filename. Example: velocity.nii.gz')

    # Arguments which implement shared functionality
    parser.add_common_args(optional)

    return parser


def main(argv: Sequence[str]):
    """Main function

    Args:
        argv (Sequence[str]): Command-line arguments
    """
    parser = get_parser()
    arguments = parser.parse_args(argv)
    venc = float(arguments.venc)
    verbose = arguments.v
    set_loglevel(verbose=verbose, caller_module_name=__name__)

    printv(f"Load data: {arguments.i}", verbose)
    nii_phase = Image(arguments.i)

    # Scale phase data between -pi and pi
    printv('Scale phase data between -pi and pi', verbose)
    data_phase_scaled = flow.scale_phase(nii_phase.data)

    # Calculate velocity
    printv('Calculate velocity', verbose)
    velocity = flow.calculate_velocity(data_phase_scaled, venc)

    # Save velocity
    if arguments.o is not None:
        fname_velocity = arguments.o
    else:
        fname_velocity = add_suffix(arguments.i, '_velocity')
    printv(f"Save velocity: {fname_velocity}", verbose)
    nii_velocity = nii_phase.copy()
    nii_velocity.data = velocity
    nii_velocity.save(fname_velocity)

    display_viewer_syntax(
        [fname_velocity, ],
        verbose=verbose,
    )


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
