#!/usr/bin/env python
# coding: utf-8
#
# Check if the platform requires conda tensorflow.
#
# Author: Christian S. Perone

import platform


# This is the list of distros that will require
# Tensorflow to be installed using conda.
# The format is:
#
#     (distro name, distro version, distro id)
#
DISTRO_MATCH = [
    ('centos', '6.8'),
    ('debian',)
]


def run_main():
    sysplatform = platform.dist()

    for spec in DISTRO_MATCH:
        match = 0
        for spec_value, sys_value in zip(spec, sysplatform):
            if spec_value == sys_value:
                match += 1
        if match == len(spec):
            print '1'
            return

    print '0'


if __name__ == '__main__':
    run_main()
