.. _mac_installation:

**********************
Installation for MacOS
**********************

Requirements
============

Supported Operating Systems
---------------------------

* MacOS >= 10.13


Gnu Compiler Collection (gcc)
-----------------------------

You need to have ``gcc`` installed. We recommend installing `Homebrew <https://brew.sh/>`_ and then run:

.. code:: sh

  brew install gcc


Installation Options
====================


Option 1: Install from Package (recommended)
--------------------------------------------

The simplest way to install SCT is to do it via a stable release. First, download the `latest release <https://github.com/neuropoly/spinalcordtoolbox/releases>`_. Major changes to each release are listed in the :doc:`/dev_section/CHANGES`.

Once you have downloaded SCT, unpack it (note: Safari will automatically unzip it). Then, open a new Terminal, go into the created folder and launch the installer:

.. code:: sh

  ./install_sct


Option 2: Install from GitHub (development)
-------------------------------------------

If you wish to benefit from the cutting-edge version of SCT, or if you wish to contribute to the code, we recommend you download the GitHub version.

#. Retrieve the SCT code

   Clone the repository and hop inside:

   .. code:: sh

      git clone https://github.com/neuropoly/spinalcordtoolbox

      cd spinalcordtoolbox

#. Checkout the revision of interest, if different from `master`:

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

      git clone https://github.com/neuropoly/spinalcordtoolbox

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

First, `install Docker <https://docs.docker.com/install/>`_. Then, follow the examples below to create an OS-specific SCT installation.


Docker Image: Ubuntu
^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # Start from the Terminal
   docker pull ubuntu:16.04
   # Launch interactive mode (command-line inside container)
   docker run -it ubuntu
   # Now, inside Docker container, install dependencies
   apt-get update
   apt install -y git curl bzip2 libglib2.0-0 gcc
   # Note for above: libglib2.0-0 is required by PyQt
   # Install SCT
   git clone https://github.com/neuropoly/spinalcordtoolbox.git sct
   cd sct
   ./install_sct -y
   export PATH="/sct/bin:${PATH}"
   # Test SCT
   sct_testing
   # save the state of the container. Open a new Terminal and run:
   docker ps -a  # list all containers
   docker commit <CONTAINER_ID> <YOUR_NAME>/ubuntu:ubuntu16.04

Docker Image: CentOS7
^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # Start from the Terminal
   docker pull centos:centos7
   # Launch interactive mode (command-line inside container)
   docker run -it centos:centos7
   # Now, inside Docker container, install dependencies
   yum install -y which gcc git curl
   # Install SCT
   git clone https://github.com/neuropoly/spinalcordtoolbox.git sct
   cd sct
   ./install_sct -y
   export PATH="/sct/bin:${PATH}"
   # Test SCT
   sct_testing
   # save the state of the container. Open a new Terminal and run:
   docker ps -a  # list all containers
   docker commit <CONTAINER_ID> <YOUR_NAME>/centos:centos7


Enable GUI Scripts (Optional)
*****************************

In order to run scripts with GUI you need to allow X11 redirection.
First, save your Docker image:

1. Open another Terminal
2. List current docker images

   .. code:: bash

      docker ps -a

3. Save container as new image

   .. code:: bash

      docker commit <CONTAINER_ID> <YOUR_NAME>/<DISTROS>:<VERSION>

Create an X11 server for handling display:

1. Install XQuartz X11 server.
2. Check ‘Allow connections from network clientsoption inXQuartz\` settings.
3. Quit and restart XQuartz.
4. In XQuartz window xhost + 127.0.0.1
5. In your other Terminal window, run:
   ``docker run -e DISPLAY=host.docker.internal:0 -it <CONTAINER_ID>``


Additional Notes
================

If MATLAB is Installed
----------------------

MATLAB took the liberty of setting ``DYLD_LIBRARY_PATH`` and in order for SCT to run, you have to run:

.. code:: matlab

   setenv('DYLD_LIBRARY_PATH', '');

Prior to running SCT commands.
See https://github.com/neuropoly/spinalcordtoolbox/issues/405
