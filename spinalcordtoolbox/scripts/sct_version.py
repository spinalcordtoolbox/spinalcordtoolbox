#!/usr/bin/env python

from spinalcordtoolbox import __version__


def main(argv=None):  # Unused, but necessary since `launcher.py` passes sys.argv[:1] to main()
    print(__version__)


if __name__ == "__main__":
    main()
