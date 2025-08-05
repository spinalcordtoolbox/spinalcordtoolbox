.. _linux_installation:

**********************
Installation for Linux
**********************

.. warning::

    If you use Windows Subsystem for Linux (WSL), please refer to the :ref:`Windows installation section <wsl-installation>`.


Installation Options
====================

Depending on your use case, there are a number of different installation methods available:

- :ref:`Packaged Installation <native-install>`: For general use.
- :ref:`GitHub Installation <github-install>`: For developers.
- :ref:`Docker Installation <docker-install-linux>`: For cross-platform use/testing.
- **[Experimental]** :ref:`PIP Installation <pip-install>` Installation ``pip`` for use as a Python package.


Requirements
============

Supported Operating Systems
---------------------------

* Debian >=9
* Ubuntu >= 16.04
* Fedora >= 19
* RedHat/CentOS >= 7


GNU Compiler Collection (gcc)
-----------------------------

You need to have ``gcc`` installed. We recommend installing it via your package manager.

For example on Debian/Ubuntu:

.. code:: sh

    apt install gcc


On CentOS/RedHat:

.. code:: sh

    yum -y install gcc


.. _native-install:

Install from Package
--------------------

The simplest way to install SCT is to use an in-place, static version of a tested package release. If you do not have any special circumstances, we recommend using this installation method.

First, navigate to the `latest release <https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases>`__, then download the install script for SCT (``install_sct-<version>_linux.sh``). Major changes to each release are listed in the :doc:`/dev_section/CHANGES`.

Once you have downloaded SCT, open a new Terminal in the location of the downloaded script, then launch the installer using the ``bash`` command. For example, if the script was downloaded to `Downloads/`, then you would run:

.. code:: sh

    cd ~/Downloads
    bash install_sct-<version>_linux.sh


.. _github-install:

Install from GitHub
-------------------

If you wish to benefit from the cutting-edge version of SCT, or if you wish to contribute to or test changes to the code, we recommend you install SCT using this method.

#. Retrieve the SCT code

    Clone the repository and hop inside:

    .. code:: sh

        git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox

        cd spinalcordtoolbox

#. (Optional) Checkout the revision of interest, if different from `master`:

    .. code:: sh

      git checkout <revision_of_interest>

#. Run the installer and follow the instructions

    .. code:: sh

        ./install_sct

.. _docker-install-linux:

Install within Docker
---------------------

`Docker <https://www.docker.com/what-container/>`__ is a portable container platform. This is useful in some niche cases, such as:

In the context of SCT, it can be used to test SCT in a specific OS environment; this is much faster than running a fully fledged virtual machine.

The instructions below are for installing Docker itself, and optionally for enabling GUI scripts. Once this is done, you can follow the :ref:`instructions for installing SCT within Docker <docker-install-sct>`.

Basic Installation (No GUI)
***************************

First, `install Docker <https://docs.docker.com/engine/install/#server>`__. Be sure to install from your distribution's repository.

.. note::
    Docker Desktop for Linux is not recommended if you intend to use the GUI.
    Instead install the `Docker Server Engine <https://docs.docker.com/engine/install/#server>`__, which is separate from the Docker Desktop Engine.
    For example on Ubuntu/Debian, follow the instructions for installing Docker from the `apt repository <https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository>`__.

By default, Docker commands require the use of ``sudo`` for additional permissions. If you want to run Docker commands without needing to add ``sudo``, please follow `these instructions <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`__ to create a Unix group called ``docker``, then add your user account to it.

Next, download the Docker image for Ubuntu 22.04 with the following command (this only needs to be done once):

.. code:: bash

    # Pull the Docker image for Ubuntu 22.04
    sudo docker pull ubuntu:22.04

If you want to enable GUI scripts, follow :ref:`the instructions below <docker-gui-linux>`. Otherwise, if you don't need to enable GUI scripts, you can launch an interactive terminal within Docker by running the following command (outside Docker):

.. code:: bash

    # Launch interactive mode (command-line inside container)
    sudo docker run -it ubuntu:22.04

And, after :ref:`installing SCT within Docker <docker-install-sct>`, you can save your container with the following commands (outside Docker):

.. code:: bash

    # Back on the Host machine, run:
    sudo docker ps -a  # list all containers (to find out the container ID)
    # specify the ID, and also choose a name to use for the docker image, such as "sct_v6.0"
    sudo docker commit <CONTAINER_ID> <IMAGE_NAME>/ubuntu:ubuntu22.04

Once the container is saved, you can use it as many times as you want to launch a terminal inside Docker and run SCT commands, by running:

.. code:: bash

    # Replace <IMAGE_NAME> with the name you chose above
    sudo docker run -it --rm <IMAGE_NAME>/ubuntu:ubuntu22.04

.. _docker-gui-linux:

Enabling GUI Scripts
********************

In order to run scripts with GUI you need to allow X11 redirection.

.. note::

    The following instructions have been tested with Xorg and xWayland.

    Set up may vary if you are using a different X11 server.

#. Install ``xauth`` and ``xhost`` on the host machine, if not already installed:

   For example on Debian/Ubuntu:

    .. code:: bash

        sudo apt install xauth x11-xserver-utils

#. Permit docker access to the X11 Server

   If you are hosting the container from the local machine:

    .. code:: bash

        xhost +local:docker

#. Follow the :ref:`instructions for installing SCT within Docker <docker-install-sct>` to create a Docker container with SCT.

#. Once you have a Docker container, you will need to run it with the following command in your terminal (outside Docker) when you want to use GUI scripts:

    .. code:: bash

        sudo docker run -it --rm --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix <IMAGE_NAME>/ubuntu:ubuntu22.04``

   This will launch a terminal within your Docker container, and you can test whether GUI scripts are available by running the following command (inside Docker):
 
    .. code:: bash
   
        sct_check_dependencies

   You should see two green ``[OK]`` symbols at the bottom of the report for "PyQT" and "matplotlib" checks, which mean that the GUI features provided by SCT are now available.

.. _pip-install:

**[EXPERIMENTAL]** Install as a ``pip`` Package
-----------------------------------------------

You should only install SCT this way if you need to access the internal functions of the package for use in a Python environment. As well, doing so comes with some caveats:

- The installation is done in-place, so the folder containing SCT must be kept around and in the same place it was originally.
- In order to ensure coexistence with other packages, the dependency specifications are loosened. As a result, it is much more likely that you will be running a combination that has not been tested, which may introduce unpredicable bugs or crashing.

If the installation fails, or you run into errors, please report a bug indicating the dependency versions retrieved using "sct_check_dependencies", and try again with a clean ``pip`` installation/environment.

#. [Optional] `Activate <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#activate-a-virtual-environment>`__ the ``virtualenv`` environment you want to install SCT within.

#. Clone the current SCT repository and enter it.

    .. code:: sh

        git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox

        cd spinalcordtoolbox

#. Checkout the revision of interest, if different from ``master``:

    .. code:: sh

        git checkout <revision_of_interest>

#. Install ``numpy``:

    .. code:: sh

        pip install numpy

#. Install SCT using ``pip``:

    If you're installing within a ``virtualenv``:

    .. code:: sh

        pip install -e .

    Otherwise (you want SCT available in your base environment):

    .. code:: sh

        pip install --user -e .
