.. _linux_installation:

**********************
Installation for Linux
**********************

.. warning::

   If you use Windows Subsystem for Linux (WSL), please refer to the :ref:`Windows installation section <wsl-installation>`.

Requirements
============

Supported Operating Systems
---------------------------

* Debian >=9
* Ubuntu >= 16.04
* Fedora >= 19
* RedHat/CentOS >= 7


Gnu Compiler Collection (gcc)
-----------------------------

You need to have ``gcc`` installed. We recommend installing it via your package manager.

For example on Debian/Ubuntu:

.. code:: sh

  apt install gcc


On CentOS/RedHat:

.. code:: sh

  yum -y install gcc


Installation Options
====================


Option 1: Install from Package (recommended)
--------------------------------------------

The simplest way to install SCT is to do it via a stable release. First, navigate to the `latest release <https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases>`_, then download the install script for SCT (``install_sct-<version>_linux.sh``). Major changes to each release are listed in the :doc:`/dev_section/CHANGES`.

Once you have downloaded SCT, open a new Terminal in the location of the downloaded script, then launch the installer using the ``bash`` command. For example, if the script was downloaded to `Downloads/`, then you would run:

.. code:: sh

  cd ~/Downloads
  bash install_sct-<version>_linux.sh


Option 2: Install from GitHub (development)
-------------------------------------------

If you wish to benefit from the cutting-edge version of SCT, or if you wish to contribute to the code, we recommend you download the GitHub version.

#. Retrieve the SCT code

   Clone the repository and hop inside:

   .. code:: sh

      git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox

      cd spinalcordtoolbox

#. (Optional) Checkout the revision of interest, if different from `master`:

   .. code:: sh

      git checkout ${revision_of_interest}

#. Run the installer and follow the instructions

   .. code:: sh

      ./install_sct

Option 3: Install with pip (experimental)
-----------------------------------------

SCT can be installed using pip, with some caveats:

- The installation is done in-place, so the folder containing SCT must be kept around

- In order to ensure coexistence with other packages, the dependency specifications are loosened, and it is possible that your package combination has not been tested with SCT.

  So in case of problems, try again with the reference installation, and report a bug indicating the dependency versions retrieved using `sct_check_dependencies`.


Procedure:

#. Retrieve the SCT code to a safe place

   Clone the repository and hop inside:

   .. code:: sh

      git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox

      cd spinalcordtoolbox

#. Checkout the revision of interest, if different from `master`:

   .. code:: sh

      git checkout ${revision_of_interest}

#. If numpy is not already on the system, install it, either using your distribution package manager or pip.

#. Install sct using pip

   If running in a virtualenv:

   .. code:: sh

      pip install -e .

   else:

   .. code:: sh

      pip install --user -e .


Option 4: Install with Docker
-----------------------------

`Docker <https://www.docker.com/what-container>`_ is a portable (Linux, macOS, Windows) container platform.

In the context of SCT, it can be used:

- To run SCT on Windows, until SCT can run natively there
- For development testing of SCT, faster than running a full-fledged
  virtual machine
- <your reason here>

Basic Installation (No GUI)
***************************

First, `install Docker <https://docs.docker.com/engine/install/#server>`_. Be sure to install from your distribution's repository.

.. note::
   Docker Desktop for Linux is not recommended if you intend to use the GUI.
   Instead install the `Docker Server Engine <https://docs.docker.com/engine/install/#server>`_, which is separate to the Docker Desktop Engine.
   For example on Ubuntu/Debian, follow the instructions for installing Docker from the `apt repository <https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository>`_.
   
By default, Docker commands require the use of ``sudo`` for additional permissions. If you want to run Docker commands without needing to add ``sudo``, please follow `these instructions <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_ to create a Unix group called ``docker``, then add users your user account to it.

Then, follow the example below to create an OS-specific SCT installation.


Docker Image: Ubuntu
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # Start from the Terminal
   sudo docker pull ubuntu:22.04
   # Launch interactive mode (command-line inside container)
   sudo docker run -it ubuntu:22.04
   # Now, inside Docker container, install dependencies
   apt-get update
   apt install -y git curl bzip2 libglib2.0-0 libgl1-mesa-glx libxrender1 libxkbcommon-x11-0 libdbus-1-3 gcc
   # Note for above: libglib2.0-0, libgl1-mesa-glx, libxrender1, libxkbcommon-x11-0, libdbus-1-3 are required by PyQt
   # Install SCT
   git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git sct
   cd sct
   ./install_sct -y
   source /root/.bashrc
   # Test SCT
   sct_testing
   # Save the state of the container as a docker image. 
   # Back on the Host machine, open a new terminal and run:
   sudo docker ps -a  # list all containers (to find out the container ID)
   # specify the ID, and also choose a name to use for the docker image, such as "sct_v6.0"
   sudo docker commit <CONTAINER_ID> <IMAGE_NAME>/ubuntu:ubuntu22.04


Enable GUI Scripts (Optional)
*****************************

In order to run scripts with GUI you need to allow X11 redirection.
First, save your Docker image if you haven't already done so:

1. Open another Terminal
2. List current docker images

   .. code:: bash

      sudo docker ps -a

3. If you haven't already, save the container as a new image

   .. code:: bash

      sudo docker commit <CONTAINER_ID> <IMAGE_NAME>/ubuntu:ubuntu22.04

Forward X11 server:

.. note::

   The following instructions have been tested with Xorg and xWayland.

   Set up may vary if you are using a different X11 server.

1. Install ``xauth`` and ``xhost`` on the host machine, if not already installed:

   For example on Debian/Ubuntu:

   .. code:: bash

      sudo apt install xauth x11-xserver-utils

2. Permit docker access to the X11 Server

   If hosting container from the local machine:

   .. code:: bash

      xhost +local:docker

3. In your Terminal window, run:
   
   .. code:: bash 

      sudo docker run -it --rm --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix <IMAGE_NAME>/ubuntu:ubuntu22.04``

4. You can test whether GUI scripts are available by running the following command in your Docker container:
 
   .. code:: bash
   
      sct_check_dependencies
      
   You should see two green ``[OK]`` symbols at the bottom of the report for "PyQT" and "matplotlib" checks, which represent the GUI features provided by SCT. 
   
Option 5: Hard-core Installation-less SCT usage
-----------------------------------------------

This is completely unsupported.


Procedure:

#. Retrieve the SCT code


#. Install dependencies

   Example for Ubuntu 18.04:

   .. code:: sh

      # The less obscure ones may be packaged in the distribution
      sudo apt install python3-{numpy,scipy,nibabel,matplotlib,h5py,mpi4py,keras,tqdm,sympy,requests,sklearn,skimage}
      # The more obscure ones would be on pip
      sudo apt install libmpich-dev
      pip3 install --user distribute2mpi nipy dipy

   Example for Debian 8 Jessie:

   .. code:: sh

      # The less obscure ones may be packaged in the distribution
      sudo apt install python3-{numpy,scipy,matplotlib,h5py,mpi4py,requests}
      # The more obscure ones would be on pip
      sudo apt install libmpich-dev
      pip3 install --user distribute2mpi sympy tqdm Keras nibabel nipy dipy scikit-image sklearn


#. Prepare the runtime environment

   .. code:: sh

      # Create launcher-less scripts
      mkdir -p bin
      find scripts/ -executable | while read file; do ln -sf "../${file}" "bin/$(basename ${file//.py/})"; done
      PATH+=":$PWD/bin"

      # Download binary programs
      mkdir bins
      pushd bins
      sct_download_data -d binaries_linux
      popd
      PATH+=":$PWD/bins"

      # Download models & cie
      mkdir data; pushd data; for x in PAM50 optic_models pmj_models deepseg_sc_models deepseg_gm_models deepseg_lesion_models c2c3_disc_models deepreg_models ; do sct_download_data -d $x; done; popd

      # Add path to spinalcordtoolbox to PYTHONPATH
      export PYTHONPATH="$PWD:$PWD/scripts"
