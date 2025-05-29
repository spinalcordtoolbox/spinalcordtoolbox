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
- :ref:`Docker Installation <docker-install>`: For cross-platform use/testing.
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

First, navigate to the `latest release <https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases>`_, then download the install script for SCT (``install_sct-<version>_linux.sh``). Major changes to each release are listed in the :doc:`/dev_section/CHANGES`.

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

.. _docker-install:

Install within Docker
---------------------

`Docker <https://www.docker.com/what-container/>`_ is a portable container platform. This is useful in some niche cases, such as:

- When you want to test SCT in a specific OS environment; this is much faster than running a full-fledged virtual machine.
- For cross-platform use; Docker ensure's reproducibility while providing accessibility across operating systems.


Basic Installation (No GUI)
***************************

First, `install Docker <https://docs.docker.com/engine/install/#server>`_. Be sure to install from your distribution's repository.

.. note::
    Docker Desktop for Linux is not recommended if you intend to use the GUI.
    Instead install the `Docker Server Engine <https://docs.docker.com/engine/install/#server>`_, which is separate to the Docker Desktop Engine.
    For example on Ubuntu/Debian, follow the instructions for installing Docker from the `apt repository <https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository>`_.
   
By default, Docker commands require the use of ``sudo`` for additional permissions. If you want to run Docker commands without needing to add ``sudo``, please follow `these instructions <https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user>`_ to create a Unix group called ``docker``, then add users your user account to it.

Then, follow the example below to create an OS-specific SCT installation (in this case, for Ubuntu 22.04).

.. code:: bash

    # Pull the Docker image for Ubuntu 22.04
    sudo docker pull ubuntu:22.04
    # Launch interactive mode (command-line inside container)
    sudo docker run -it ubuntu:22.04
    # Now inside Docker container, install SCT dependencies
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


Enabling GUI Scripts
********************

In order to run scripts with GUI you need to allow X11 redirection.
First, save your Docker image if you haven't already done so:

#. Open another Terminal
#. List current docker images

    .. code:: bash

        sudo docker ps -a

#. If you haven't already, save the container as a new image

    .. code:: bash

        sudo docker commit <CONTAINER_ID> <IMAGE_NAME>/ubuntu:ubuntu22.04

Then, to forward the X11 server:

.. note::

    The following instructions have been tested with Xorg and xWayland.

    Set up may vary if you are using a different X11 server.

#. Install ``xauth`` and ``xhost`` on the host machine, if not already installed:

    For example on Debian/Ubuntu:

    .. code:: bash

        sudo apt install xauth x11-xserver-utils

#. Permit docker access to the X11 Server

   If hosting container from the local machine:

    .. code:: bash

        xhost +local:docker

#. In your Terminal window, run:
   
    .. code:: bash

        sudo docker run -it --rm --privileged -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix <IMAGE_NAME>/ubuntu:ubuntu22.04``

You can test whether GUI scripts are available by running the following command in your Docker container:
 
    .. code:: bash
   
        sct_check_dependencies
      
You should see two green ``[OK]`` symbols at the bottom of the report for "PyQT" and "matplotlib" checks, which represent the GUI features provided by SCT are now available.

.. _pip-install:

**[EXPERIMENTAL]** Install as a ``pip`` Package
-----------------------------------------------

You should only install SCT this way if you need to access the internal functions of the package for use in a Python environment. As well, doing so comes with some caveats:

- The installation is done in-place, so the folder containing SCT must be kept around and in the same place it was originally.
- In order to ensure coexistence with other packages, the dependency specifications are loosened. As a result, it is much more likely that you will be running a combination that has not been tested, which may introduce unpredicable bugs or crashing.

If the installation fails, or you run into errors, please report a bug indicating the dependency versions retrieved using "sct_check_dependencies", and try again with a clean ``pip`` installation/environment.

#. [Optional] `Activate <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#activate-a-virtual-environment>`_ the ``virtualenv`` environment you want to install SCT within.

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
