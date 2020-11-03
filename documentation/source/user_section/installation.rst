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
each release are listed in the :doc:`/dev_section/CHANGES`.

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

Basic Installation (No GUI)
===========================

First, `install Docker <https://docs.docker.com/install/>`_. Then, follow the examples below to create an OS-specific SCT installation.


Ubuntu-based installation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # Start from the Terminal
   docker pull ubuntu:16.04
   # Launch interactive mode (command-line inside container)
   docker run -it ubuntu
   # Now, inside Docker container, install dependencies
   apt-get update
   yes | apt install git curl bzip2 libglib2.0-0 gcc
   # Note for above: libglib2.0-0 is required by PyQt
   # Install SCT
   git clone https://github.com/neuropoly/spinalcordtoolbox.git sct
   cd sct
   yes | ./install_sct
   export PATH="/sct/bin:${PATH}"
   # Test SCT
   sct_testing
   # save the state of the container. Open a new Terminal and run:
   docker ps -a  # list all containers
   docker commit <CONTAINER_ID> <YOUR_NAME>/ubuntu:ubuntu16.04

CentOS7-based installation
~~~~~~~~~~~~~~~~~~~~~~~~~~

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
   yes | ./install_sct
   export PATH="/sct/bin:${PATH}"
   # Test SCT
   sct_testing
   # save the state of the container. Open a new Terminal and run:
   docker ps -a  # list all containers
   docker commit <CONTAINER_ID> <YOUR_NAME>/centos:centos7


Using GUI Scripts (Optional)
============================

In order to run scripts with GUI you need to allow X11 redirection.
First, save your Docker image:

1. Open another Terminal
2. List current docker images

   .. code:: bash

      docker ps -a

3. Save container as new image

   .. code:: bash

      docker commit <CONTAINER_ID> <YOUR_NAME>/<DISTROS>:<VERSION>

For macOS and Linux users
~~~~~~~~~~~~~~~~~~~~~~~~~

Create an X11 server for handling display:

1. Install XQuartz X11 server.
2. Check ‘Allow connections from network clientsoption inXQuartz\`
   settings.
3. Quit and restart XQuartz.
4. In XQuartz window xhost + 127.0.0.1
5. In your other Terminal window, run:

   -  On macOS:
      ``docker run -e DISPLAY=host.docker.internal:0 -it <CONTAINER_ID>``
   -  On Linux:
      ``docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix <CONTAINER_ID>``

For Windows users
~~~~~~~~~~~~~~~~~

#. Install Xming
#. Connect to it using Xming/SSH:

   - If you are using Docker Desktop, please download and run (double click) the following script: :download:`sct-win.xlaunch<../../../contrib/docker/sct-win.xlaunch>`.
   - If you are using Docker Toolbox, please download and run the following script instead: :download:`sct-win_docker_toolbox.xlaunch<../../../contrib/docker/sct-win_docker_toolbox.xlaunch>`
   - If this is the first time you have done this procedure, the system will ask you if you want to add the remote PC (the docker container) as trust pc, type yes. Then type the password to enter the docker container (by default sct).

**Troubleshooting:**

The graphic terminal emulator LXterminal should appear (if not check the task bar at the bottom of the screen), which allows copying and pasting commands, which makes it easier for users to use it. If there are no new open windows:

- Please download and run the following file: :download:`Erase_fingerprint_docker.sh<../../../contrib/docker/Erase_fingerprint_docker.sh>`
- Try again
- If it is still not working:

  - Open the file manager and go to C:/Users/Your_username
  - In the searchbar type ‘.ssh’ - Open the found ‘.ssh’ folder.
  - Open the ‘known_hosts’ file with a text editor
  - Remove line starting with ``192.168.99.100`` or ``localhost``
  - Try again

To check that X forwarding is working well write ``fsleyes &`` in LXterminal and FSLeyes should open, depending on how fast your computer is FSLeyes may take a few seconds to open. If fsleyes is not working in the LXterminal:

- Check if it's working on the docker machine by running ``fsleyes &`` in the docker quickstart terminal
- If it works, run all the commands in the docker terminal.
- If it throws the error ``Unable to access the X Display, is $DISPLAY set properly?`` follow these next steps:

  - Run ``echo $DISPLAY`` in the LXterminal
  - Copy the output address
  - Run ``export DISPLAY=<previously obtained address>`` in the docker quickstart terminal
  - Run ``fsleyes &`` (in the docker quickstart terminal) to check if it is working. A new Xming window with fsleyes should appear.

Notes:

- If after closing a program with graphical interface (i.e. FSLeyes) LXterminal does not raise the shell ($) prompt then press Ctrl + C to finish closing the program.
- Docker exposes the forwarded SSH server at different endpoints depending on whether Docker Desktop or Docker Toolbox is installed.

  - Docker Desktop:

    .. code:: bash

       ssh -Y -p 2222 sct@127.0.0.1

  - Docker Toolbox:

    .. code:: bash

       ssh -Y -p 2222 sct@192.168.99.100


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



