#!/usr/bin/env python
# -*- coding: utf-8
"""
This command-line tool allow users to easily output the BibTex citation in their terminal and allow them to
do some automation
"""

import sys
import logging
from textwrap import dedent

from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar
from spinalcordtoolbox.utils.sys import init_sct, printv, set_loglevel

logger = logging.getLogger(__name__)


def get_parser():
    parser = SCTArgumentParser(
        description="Output the BibTex citation of SCT"
    )

    misc = parser.add_argument_group('\nMISC')
    misc.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode")
    misc.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit")

    return parser


def main(argv=None):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    reference = """\
    @article{DeLeener201724,
    title = "SCT: Spinal Cord Toolbox, an open-source software for processing spinal cord \\{MRI\\} data ",
    journal = "NeuroImage ",
    volume = "145, Part A",
    number = "",
    pages = "24 - 43",
    year = "2017",
    note = "",
    issn = "1053-8119",
    doi = "https://doi.org/10.1016/j.neuroimage.2016.10.009",
    url = "http://www.sciencedirect.com/science/article/pii/S1053811916305560",
    author = "Benjamin De Leener and Simon Lévy and Sara M. Dupont and Vladimir S. Fonov and Nikola Stikov and D. Louis Collins and Virginie Callot and Julien Cohen-Adad",
    keywords = "Spinal cord",
    keywords = "MRI",
    keywords = "Software",
    keywords = "Template",
    keywords = "Atlas",
    keywords = "Open-source ",
    }"""

    # Use dedent in order to remove indentation problem with block of text and keep visual indentation in source code
    # see more here if needed: https://docs.python.org/3/library/textwrap.html#textwrap.dedent
    printv(dedent(reference))


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
