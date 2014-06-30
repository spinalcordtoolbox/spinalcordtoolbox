SpinalCordToolbox
=================

Comprehensive and open-source library of analysis tools for MRI of the spinal cord.


Installation
-------------------
see: https://sourceforge.net/p/spinalcordtoolbox/wiki/installation/


List of tools
-------------------
see: http://sourceforge.net/p/spinalcordtoolbox/wiki/tools/


Get started
-------------------
see: https://sourceforge.net/p/spinalcordtoolbox/wiki/get_started/


License
-------------------
All files are licensed as described in the LICENSE.TXT file.


Environment variable
-------------------
If you install the toolbox from GitHub, DO NOT RUN THE INSTALLER. The installer is only made for packages (created by create_package.py). So, if you get the toolbox from GitHub, you need to edite your .bash_profile accordingly:

````
# SPINALCORDTOOLBOX
SCT_DIR="path_to_the_toolbox"
export PATH=${PATH}:$SCT_DIR/scripts
export PATH=${PATH}:$SCT_DIR/bin/osx
export SCT_DIR PATH
export PATH=${PATH}:$SCT_DIR/install/osx/ants
export PATH=${PATH}:$SCT_DIR/install/osx/c3d
````
N.B. If you are running debian/ubuntu, replace "osx" by "debian"
