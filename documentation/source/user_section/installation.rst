.. _installation:

************
Installation
************

SCT works in macOS, Linux and Windows (see Requirements below). SCT bundles its own Python distribution (Miniconda), installed with all the required packages, and uses specific package versions, in order to ensure reproducibility of results. SCT offers various installation methods:

.. contents::
   :local:
..


Requirements
============

* Operating System (OS):

  * macOS >= 10.13
  * Debian >=9
  * Ubuntu >= 16.04
  * Fedora >= 19
  * RedHat/CentOS >= 7
  * Windows, see `Install on Windows 10 with WSL`_.

* You need to have ``gcc`` installed. On macOS, we recommend installing `Homebrew <https://brew.sh/>`_ and then run ``brew install gcc``. On Linux, we recommend installing it via your package manager. For example on Debian/Ubuntu: ``apt install gcc``, and on CentOS/RedHat: ``yum -y install gcc``.



Install from package (recommended)
==================================

The simplest way to install SCT is to do it via a stable release. First, download the `latest release <https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases>`_. Major changes to each release are listed in the :doc:`/dev_section/CHANGES`.

Once you have downloaded SCT, unpack it (note: Safari will automatically unzip it). Then, open a new Terminal, go into the created folder and launch the installer:

.. code:: sh

  ./install_sct

.. note::
  The package installation only works on macOS and Linux.


Install from Github (development)
=================================

If you wish to benefit from the cutting-edge version of SCT, or if you wish to contribute to the code, we recommend you download the Github version.

#. Retrieve the SCT code

   Clone the repository and hop inside:

   .. code:: sh

      git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox

      cd spinalcordtoolbox

#. Checkout the revision of interest, if different from `master`:

   .. code:: sh

      git checkout ${revision_of_interest}

#. Run the installer and follow the instructions

   .. code:: sh

      ./install_sct


Install on Windows 10 with WSL
==============================

Windows subsystem for Linux (WSL) is available on Windows 10 and it makes it possible to run native Linux programs, such as SCT.

#. Install Windows Subsystem for Linux (WSL)

   - Install `Xming <https://sourceforge.net/projects/xming/>`_.

   - Install  `Windows subsystem for linux and initialize it <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_.

     .. important::

        Make sure to install WSL1. SCT can work with WSL2, but the installation procedure described here refers to WSL1.
        If you are comfortable with installing SCT with WSL2, please feel free to do so.

        When asked what Linux version to install, select the Ubuntu 18.04 LTS distro.

#. Environment preparation

   Run the following command to install various packages that will be needed to install FSL and SCT. This will require your password

   .. code-block:: sh

      sudo apt-get update
      sudo apt-get -y install gcc
      sudo apt-get -y install unzip
      sudo apt-get install -y python-pip python
      sudo apt-get install -y psmisc net-tools
      sudo apt-get install -y git
      sudo apt-get install -y gfortran
      sudo apt-get install -y libjpeg-dev
      echo 'export DISPLAY=127.0.0.1:0.0' >> .profile

#. Install SCT

   Download SCT:

   .. code-block:: sh

      git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git sct
      cd sct

   To select a `specific release <https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases>`_, replace X.Y.Z below with the proper release number. If you prefer to use the development version, you can skip this step.

   .. code-block:: sh

      git checkout X.Y.Z

   Install SCT:

   .. code:: sh

      ./install_sct -y

   To complete the installation of these software run:

   .. code:: sh

      cd ~
      source .profile
      source .bashrc

   You can now use SCT. Your local C drive is located under ``/mnt/c``. You can access it by running:

   .. code:: sh

      cd /mnt/c




Install with Docker
===================

`Docker <https://www.docker.com/what-container>`_ is a portable (Linux, macOS, Windows) container platform.

In the context of SCT, it can be used:

- To run SCT on Windows, until SCT can run natively there
- For development testing of SCT, faster than running a full-fledged
  virtual machine
- <your reason here>

Basic Installation (No GUI)
---------------------------

First, `install Docker <https://docs.docker.com/install/>`_. Then, follow the examples below to create an OS-specific SCT installation.


Ubuntu-based installation
^^^^^^^^^^^^^^^^^^^^^^^^^

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
   git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git sct
   cd sct
   ./install_sct -y
   export PATH="/sct/bin:${PATH}"
   # Test SCT
   pytest .
   # save the state of the container. Open a new Terminal and run:
   docker ps -a  # list all containers
   docker commit <CONTAINER_ID> <YOUR_NAME>/ubuntu:ubuntu16.04

CentOS7-based installation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # Start from the Terminal
   docker pull centos:centos7
   # Launch interactive mode (command-line inside container)
   docker run -it centos:centos7
   # Now, inside Docker container, install dependencies
   yum install -y which gcc git curl
   # Install SCT
   git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git sct
   cd sct
   ./install_sct -y
   export PATH="/sct/bin:${PATH}"
   # Test SCT
   pytest .
   # save the state of the container. Open a new Terminal and run:
   docker ps -a  # list all containers
   docker commit <CONTAINER_ID> <YOUR_NAME>/centos:centos7


Using GUI Scripts (Optional)
----------------------------

In order to run scripts with GUI you need to allow X11 redirection.
First, save your Docker image:

1. Open another Terminal
2. List current docker images

   .. code:: bash

      docker ps -a

3. Save container as new image

   .. code:: bash

      docker commit <CONTAINER_ID> <YOUR_NAME>/<DISTROS>:<VERSION>

For MacOS and Linux users
^^^^^^^^^^^^^^^^^^^^^^^^^

Create an X11 server for handling display:

1. Install XQuartz X11 server.
2. Check ‘Allow connections from network clientsoption inXQuartz\` settings.
3. Quit and restart XQuartz.
4. In XQuartz window xhost + 127.0.0.1
5. In your other Terminal window, run:

   -  On macOS:
      ``docker run -e DISPLAY=host.docker.internal:0 -it <CONTAINER_ID>``
   -  On Linux:
      ``docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix <CONTAINER_ID>``

For Windows users
^^^^^^^^^^^^^^^^^

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


Matlab Integration on Mac
=========================

Matlab took the liberty of setting ``DYLD_LIBRARY_PATH`` and in order for SCT to run, you have to run:

.. code:: matlab

   setenv('DYLD_LIBRARY_PATH', '');

Prior to running SCT commands.
See https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/405
