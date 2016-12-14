#!/usr/bin/env python
#
# Installer for requirements
#

from commands import getstatusoutput

print '\n--------------------------'
print 'INSTALLER FOR REQUIREMENTS'
print '--------------------------'

# Installation with conda
print '\nInstallation with conda...'
status, output = getstatusoutput('conda install --yes --file requirementsConda.txt')
if status:
    print 'ERROR: Installation with conda failed.\n'+output
else:
    print output

# Installation with pip
print '\nInstallation with pip...'
status, output = getstatusoutput('pip install -r requirementsPip.txt')
if status:
    print 'ERROR: Installation with pip failed.\n'+output
else:
    print output
