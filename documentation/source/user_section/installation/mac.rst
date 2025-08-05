.. _mac_installation:

**********************
Installation for MacOS
**********************

Requirements
============

Supported Operating Systems
---------------------------

* MacOS >= 10.13


Mac computers with Apple silicon
--------------------------------

If your computer uses Apple silicon (M1, M2, M3, etc.), you need to make sure `Rosetta 2 <https://en.wikipedia.org/wiki/Rosetta_(software)#Rosetta_2>`__ is installed before installing or running SCT. To do this, you can open a Terminal and run:

.. code:: sh

    softwareupdate --install-rosetta


Gnu Compiler Collection (gcc)
-----------------------------

You need to have ``gcc`` installed. Check to see if ``gcc`` is installed by opening a Terminal and running:

.. code:: sh

    gcc --version


If it isn't installed, we recommend installing `Homebrew <https://brew.sh/>`__ and then run:

.. code:: sh

  brew install gcc


Installation Options
====================


Option 1: Install from Package (recommended)
--------------------------------------------

The simplest way to install SCT is to do it via a stable release. First, navigate to the `latest release <https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases>`__, then download the install script for SCT (``install_sct-<version>_macos.sh``). Major changes to each release are listed in the :doc:`/dev_section/CHANGES`.

Once you have downloaded SCT, open a new Terminal in the location of the downloaded script, then launch the installer the ``bash`` command. For example, if the script was downloaded to `Downloads/`, then you would run:

.. code:: sh

  cd ~/Downloads
  bash install_sct-<version>_macos.sh


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

.. _docker-install-macos:

Option 4: Install with Docker
-----------------------------

`Docker <https://www.docker.com/what-container/>`__ is a portable (Linux, macOS, Windows) container platform.

In the context of SCT, it can be used to test SCT in a specific OS environment; this is much faster than running a fully fledged virtual machine.

The instructions below are for installing Docker itself, and optionally for enabling GUI scripts. Once this is done, you can follow the :ref:`instructions for installing SCT within Docker <docker-install-sct>`.

Basic Installation (No GUI)
***************************

First, `install Docker Desktop <https://docs.docker.com/desktop/install/mac-install/>`__. 

Next, download the Docker image for Ubuntu 22.04 with the following Terminal command (this only needs to be done once):

.. code:: bash

    # Start from the Terminal
    docker pull ubuntu:22.04
    # If the previous command says 'Cannot connect to the Docker daemon', make sure you have launched Docker Desktop

If you want to enable GUI scripts, follow :ref:`the instructions below <docker-gui-macos>`. Otherwise, if you don't need to enable GUI scripts, you can launch an interactive terminal within Docker by running the following command (outside Docker):

.. code:: bash

    # Launch interactive mode (command-line inside container)
    docker run -it ubuntu:22.04

And, after :ref:`installing SCT within Docker <docker-install-sct>`, you can save your container with the following commands (outside Docker):

.. code:: bash

    # Back on the Host machine, run:
    docker ps -a  # list all containers (to find out the container ID)
    # specify the ID, and also choose a name to use for the docker image, such as "sct_v6.0"
    docker commit <CONTAINER_ID> <IMAGE_NAME>/ubuntu:ubuntu22.04

Once the container is saved, you can use it as many times as you want to launch a terminal inside Docker and run SCT commands, by running:

.. code:: bash

    # Make sure you have launched Docker Desktop first
    # Replace <IMAGE_NAME> with the name you chose above
    docker run -it --rm <IMAGE_NAME>/ubuntu:ubuntu22.04

.. _docker-gui-macos:

Enable GUI Scripts (Optional)
*****************************

In order to run scripts with GUI you will need to allow X11 redirection.
Create an X11 server for handling display:

1. Install `XQuartz X11 <https://www.xquartz.org/>`__ server.
2. Check ‘Allow connections from network clients option in XQuartz\` settings. (Run XQuartz; in the top-of-screen menu bar, choose "XQuartz", then "Preferences..."; the relevant option is under the "Security" tab.)
3. Quit and restart XQuartz.
4. In the XQuartz "xterm" window, run the command:

   .. code:: bash

      xhost + 127.0.0.1

5. Follow the :ref:`instructions for installing SCT within Docker <docker-install-sct>`. to create a Docker container with SCT.
6. Once you have a Docker container, you will need to run it with the following command in your Terminal (outside Docker) when you want to use GUI scripts:

   .. code:: bash

      docker run -e DISPLAY=host.docker.internal:0 -it <IMAGE_NAME>/ubuntu:ubuntu22.04

   (If this command says 'Cannot connect to the Docker daemon', try again after launching Docker Desktop.)

   This will launch a terminal within your Docker container, and you can test whether GUI scripts are available by running the following command (inside Docker):

   .. code:: bash

      sct_check_dependencies

   You should see two green ``[OK]`` symbols at the bottom of the report for "PyQT" and "matplotlib" checks, which mean that the GUI features provided by SCT are now available.

Additional Notes
================

If MATLAB is Installed
----------------------

MATLAB took the liberty of setting ``DYLD_LIBRARY_PATH`` and in order for SCT to run, you have to run:

.. code:: matlab

   setenv('DYLD_LIBRARY_PATH', '');

Prior to running SCT commands.
See https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/405
