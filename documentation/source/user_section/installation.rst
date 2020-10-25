.. _installation:

Installation
############

SCT works in macOS, Linux and Windows (see Requirements below). SCT bundles its own Python distribution (Miniconda),
installed with all the required packages, and uses specific package versions, in order to ensure reproducibility of
results. SCT offers various installation methods:

.. contents::
   :local:
..


Requirements
------------

* Operating System (OS):

  * macOS >= 10.13
  * Debian >=9
  * Ubuntu >= 16.04
  * Fedora >= 19
  * RedHat/CentOS >= 7
  * Windows, see `Install on Windows 10 with WSL`_.

* You need to have ``gcc`` installed. On macOS, we recommend installing `Homebrew <https://brew.sh/>`_ and then run
  ``brew install gcc``. On Linux, we recommend installing it via your package manager. For example on Debian/Ubuntu:
  ``apt install gcc``, and on CentOS/RedHat: ``yum -y install gcc``.



Install from package (recommended)
----------------------------------

The simplest way to install SCT is to do it via a stable release. First, download the
`latest release <https://github.com/neuropoly/spinalcordtoolbox/releases>`_. Major changes to
each release are listed in the `CHANGES.md <https://github.com/neuropoly/spinalcordtoolbox/blob/master/CHANGES.md>`_ file.

Once you have downloaded SCT, unpack it (note: Safari will automatically unzip it). Then, open a new Terminal,
go into the created folder and launch the installer:

.. code:: sh

  ./install_sct

.. note::
  The package installation only works on macOS and Linux.


Install from Github (development)
---------------------------------

If you wish to benefit from the cutting-edge version of SCT, or if you wish to contribute to the code, we
recommend you download the Github version.

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


Install on Windows 10 with WSL
------------------------------

Windows subsystem for Linux (WSL) is available on Windows 10 and it makes it possible to run native Linux programs,
such as SCT. Checkout the `installation tutorial for WSL <https://github.com/neuropoly/spinalcordtoolbox/wiki/SCT-on-Windows-10:-Installation-instruction-for-SCT-on-Windows-subsytem-for-linux>`_.


Install with Docker
-------------------

`Docker <https://www.docker.com/what-container>`_ is a portable (Linux, macOS, Windows) container platform.

In the context of SCT, it can be used:

- To run SCT on Windows, until SCT can run natively there
- For development testing of SCT, faster than running a full-fledged
  virtual machine
- <your reason here>

Option 1: Without GUI
=====================

First, `install Docker`_. Then, follow instructions below for creating an OS-specific SCT installation and testing.

Run Docker image
~~~~~~~~~~~~~~~~

**For Ubuntu:**

.. code:: bash

   # in the Terminal
   docker pull ubuntu:16.04
   docker run -it ubuntu
   # in docker container
   apt-get update
   yes | apt install git curl bzip2 libglib2.0-0 gcc
   # Note: libglib2.0-0 is required by PyQt

**For CentOS7:**

.. code:: bash

   # in the Terminal
   docker pull centos:centos7
   docker run -it centos:centos7
   # in docker container
   yum install -y which gcc git curl
   # save the state of the container. Open a new Terminal and run:
   docker ps -a  # list all containers
   docker commit <CONTAINER_ID> <YOUR_NAME>/centos:centos7

Install SCT
~~~~~~~~~~~

After having installed your favorite OS, run SCT installer and test it:

.. code:: bash

   git clone https://github.com/neuropoly/spinalcordtoolbox.git sct
   cd sct
   yes | ./install_sct
   export PATH="/sct/bin:${PATH}"
   sct_testing

Option 2: With GUI
==================

In order to run scripts with GUI you need to allow X11 redirection.
First, save your Docker image:

1. Open another Terminal
2. List current docker images

.. code:: bash

   docker ps -a

3. Save container as new image

.. code:: bash

   docker commit <CONTAINER_ID> <YOUR_NAME>/<DISTROS>:<VERSION>

For OSX and Linux users
~~~~~~~~~~~~~~~~~~~~~~~

Create an X11 server for handling display:

1. Install XQuartz X11 server.
2. Check ‘Allow connections from network clientsoption inXQuartz\`
   settings.
3. Quit and restart XQuartz.
4. In XQuartz window xhost + 127.0.0.1
5. In your other Terminal window, run:

   -  On OSX:
      ``docker run -e DISPLAY=host.docker.internal:0 -it <CONTAINER_ID>``
   -  On Linux:
      ``docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix <CONTAINER_ID>``

For windows users
~~~~~~~~~~~~~~~~~

| 1.Install Xming
| 2.Connect to it using Xming/SSH:
| Open a new CMD window and clone this repository:
| ``git clone https://github.com/neuropoly/sct_docker.git``

| If you are using Docker Desktop, run (double click)
  windows/sct-win.xlaunch. If you are using Docker Toolbox, run
  windows/sct-win_docker_toolbox.xlaunch
| If this is the first time you have done this procedure, the system
  will ask you if you want to add the remote PC (the docker container)
  as trust pc, type yes. Then type the password to enter the docker
  container (by default sct).

**Troubleshooting:**

| If there is no new open windows :
| - Double click on the ‘windows/Erase_fingerprint_docker’ program
| - Try again
| - if it is still not working :
| - Open the file manager and go to C:/Users/Your_username - In the
  searchbar type ‘.ssh’ - Open the found ‘.ssh’ folder. - Open the
  ‘known_hosts’ file with a text editor - Remove line starting with
  ``192.168.99.100`` or ``localhost`` - try again

The graphic terminal emulator LXterminal should appear (if not check the
task bar at the bottom of the screen), which allows copying and pasting
commands, which makes it easi

.. _install Docker: https://docs.docker.com/install/

Install with pip (experimental)
-------------------------------

SCT can be installed using pip, with some caveats:

- The installation is done in-place, so the folder containing SCT must
  be kept around

- In order to ensure coexistence with other packages, the dependency
  specifications are loosened, and it is possible that your package
  combination has not been tested with SCT.

  So in case of problem, try again with the reference installation,
  and report a bug indicating the dependency versions retrieved using
  `sct_check_dependencies`.


Procedure:

#. Retrieve the SCT code to a safe place

   Clone the repository and hop inside:

   .. code:: sh

      git clone https://github.com/neuropoly/spinalcordtoolbox

      cd spinalcordtoolbox

#. Checkout the revision of interest, if different from `master`:

   .. code:: sh

      git checkout ${revision_of_interest}

#. If numpy is not already on the system, install it, either using
   your distribution package manager or pip.

#. Install sct using pip

   If running in a virtualenv:

   .. code:: sh

      pip install -e .

   else:

   .. code:: sh

      pip install --user -e .


Hard-core Installation-less SCT usage
-------------------------------------

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
      mkdir data; pushd data; for x in PAM50 gm_model optic_models pmj_models deepseg_sc_models deepseg_gm_models ; do sct_download_data -d $x; done; popd

      # Add path to spinalcordtoolbox to PYTHONPATH
      export PYTHONPATH="$PWD:$PWD/scripts"


Matlab Integration on Mac
-------------------------

Matlab took the liberty of setting ``DYLD_LIBRARY_PATH`` and in order
for SCT to run, you have to run:

.. code:: matlab

   setenv('DYLD_LIBRARY_PATH', '');

Prior to running SCT commands. See
 https://github.com/neuropoly/spinalcordtoolbox/issues/405



