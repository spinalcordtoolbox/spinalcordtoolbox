.. _windows_installation:

************************
Installation for Windows
************************

We currently provide 3 different ways to install SCT on Windows machines:

- :ref:`Native Windows installation (recommended) <native-installation>`
- :ref:`Windows Subsystem for Linux (WSL) <wsl-installation>`
- :ref:`Docker <docker-installation>`


-----


.. _native-installation:

Native Windows installation (recommended)
-----------------------------------------

This set of instructions will show you how to install SCT directly on Windows.

.. note::

   This method was first introduced to SCT in April 2022 as part of the Version 5.6 release. Previous versions of SCT required the use of :ref:`Windows Subsystem for Linux (WSL) <wsl-installation>` or :ref:`Docker <docker-installation>` in order to support Windows.

   Because this method is new to SCT, we would greatly appreciate hearing any feedback you may have about your experiences using SCT on Windows. If you encounter any issues, or have any questions or concerns, feel free to post on the `Spinalcordmri.org forums <https://forum.spinalcordmri.org/c/sct/8>`_, and SCT's development team will be happy to help.

1. Installing Prerequisites
***************************

SCT depends on two pieces of software that must be set up prior to the installation of SCT.

Python 3.7
^^^^^^^^^^

Since SCT is a Python package, Python must be installed on your system before SCT can be installed.

1. Download a "Windows x86-64" installer from `the Python 3.7.9 download page <https://www.python.org/downloads/release/python-379/>`_.

2. Run the installer file. (**Important:** Before clicking "Install Now", make sure to first check the "Add Python 3.7 to PATH" checkbox at the bottom of the window.)

3. After the installation has finished, open your Start Menu and type Command Prompt, then run it. In the Command Prompt window, type ``python --version`` and press enter.

   (Make sure that you see the text ``Python 3.7.9`` before continuing.)


Git for Windows
^^^^^^^^^^^^^^^

The easiest way to try out different versions of SCT is using Git.

1. Download Git for Windows from `the Git download page <https://git-scm.com/download/win>`_.

2. Run the installer.

   - You can click "Next" for most of the options in the installer.
   - However, on the "Adjusting your PATH environment" page, we strongly recommend that you choose the "Use Git and optional Unix tools from the Command Prompt". Selecting this option will provide you with useful Unix utilities such as ``bash``, ``cd``, ``ls``, and more that combine nicely with SCT's command-line tools. In particular, ``bash`` will allow ``sct_run_batch`` to execute bash scripts for batch processing of subjects.
   - **Note:** If you prefer, you may instead choose to :ref:`install Cygwin<installing_cygwin>` (rather than selecting the "Use Git and optional Unix tools" option) in order to gain access to these same Unix utilities.

3. After the installation has finished, open your Start Menu and type Command Prompt, then run it. In the Command Prompt window, type ``git --version`` and press enter.

   (Make sure that you see the text ``git version <number>`` before continuing.)


2. Installing SCT
*****************

We recommend that you install SCT into a Python virtual environment. To help with this process, SCT provides an installer script that will automate the process of creating a Python virtual environment for you.

1. Navigate to the `Releases page <https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/>`_ , then download the ``install_sct.bat`` script from the "Assets" section of the latest release. 

2. Run the script by double-clicking it. The script will fetch the SCT source code, then install the `spinalcordtoolbox` package into a Python virtual environment for you.

3. Once the installer finishes, follow the instructions given at the end of the Command Prompt window, which will instruct you to add SCT to your PATH.

4. Finally, in the Command Prompt window, type ``sct_check_dependencies`` and press enter. Make sure that you see a status report containing all "OKs" before continuing.

5. You are now free to use SCT's command line tools to process your data. If you would like to learn how to use SCT, we recommend starting with SCT's :ref:`tutorials` pages.

.. _installing_cygwin:

3. (Optional) Installing Cygwin
*******************************

.. note:: You do not need to install Cygwin if you already selected the "Use Git and optional Unix tools from the Command Prompt" option during the Git installation step.

Cygwin is a collection of useful Unix utilities such as ``bash``, ``cd``, ``ls``, and more that combine nicely with SCT's command-line tools. In particular, ``bash`` will allow ``sct_run_batch`` to execute bash scripts for batch processing of subjects.

1. Download the Cygwin installer from `the Cygwin installation page <https://www.cygwin.com/install.html>`_.

2. Run the installer. (You can click "Next" for every section of the installer, as the default settings are sufficient.)

3. After the installer is finished, you will need to add Cygwin's programs to the PATH.

   - Open the Start Menu -> Type 'path' -> Open 'Edit environment variables for your account'
   - Under the section 'User variables for ____', highlight the 'Path' entry, then click the 'Edit...' button.
   - Click 'New', then copy and paste "``C:\cygwin64\bin``".
   - Finally, click "Ok" three times.

4. Finally, open your Start Menu and type Command Prompt, then run it. In the Command Prompt window, type ``cygcheck --version`` and press enter. Make sure that you see the text ``cygcheck (cygwin)`` before continuing.

   - Note: If you see a "not recognized" error, please repeat Step 3, making sure that the directory you added corresponds to the installation directory of Cygwin.


-----


.. _wsl-installation:

Windows Subsystem for Linux (WSL) installation
----------------------------------------------

Windows Subsystem for Linux (WSL) makes it possible to run native Linux programs on Windows 10. Here, WSL is used to install the Linux version of SCT within Windows (as opposed to the :ref:`native Windows version <native-installation>`).

Basic installation (No GUI)
***************************

#. Make sure that your version of Windows 10 is up to date before continuing.

   - In Windows, search for "System Information" and open the app. In the "Version" field, make sure that you are running "Build 19041" or higher.

   - Then, search for "Powershell" in your Start Menu, then right-click and "Run as administrator". Then run the following command:

     .. code::

        wsl --update

   - If this command is successful, then you can proceed to the next step. Otherwise, please try the following troubleshooting steps:

     - Make sure your version of Windows is up to date.
     - Make sure that you have sufficient administrative privileges for your Windows system.
     - If you cannot update Windows, then you can try the instructions from Microsoft's `"Manual installation steps for older versions of WSL" <https://docs.microsoft.com/en-us/windows/wsl/install-manual>`_ page.

#. Install an Ubuntu distribution in Windows Subsystem for Linux (WSL)

   - In Windows, search for "Powershell" in your Start Menu, then right-click and "Run as administrator".

   - In PowerShell, type the following command and press enter:

     .. code::

        wsl --install

   - After this command finishes, you will be prompted to restart your computer.

   - After restarting, the installation should automatically resume, and you will be able to create a user account inside Ubuntu by selecting a username and password.

#. Choose the WSL version (1/2).

   By default, Microsoft's instructions will create an Ubuntu environment using Version 2 of WSL. While version 2 has been tested to work with SCT, our development team tests more thoroughly using Version 1 of WSL (due to `better support from GitHub Actions <https://github.com/actions/virtual-environments/issues/50>`_).

   Because of this, we recommend that you convert the WSL2 Ubuntu environment to use WSL1 before continuing. To do this, first close Ubuntu, then re-open Powershell and run the following command:

   .. code::

      wsl --list --verbose

   If WSL installed correctly, you should see a list of installed distributions (as well as their WSL versions). Find the name of the distribution you just installed (which should be something like ``Ubuntu`` or ``Ubuntu-20.04``), then specify that name in the following command:

   .. code::

      wsl --set-version Ubuntu 1

   After you run this command, you can then run ``wsl --list --verbose`` again to check that the distribution has changed from WSL2 to WSL1.

#. Environment preparation

   Now that you have set up an Ubuntu environment with WSL, please open Ubuntu and run the following commands to install various packages that will be needed to install SCT.

   .. code-block:: sh

      sudo apt-get update && sudo apt-get upgrade -y
      sudo apt-get -y install gcc unzip python3-pip python3 psmisc net-tools git gfortran libjpeg-dev

#. Install SCT

   First, download SCT by running the following commands in Ubuntu:

   .. code-block:: sh

      cd ~
      git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git
      cd spinalcordtoolbox

   To select a `specific release <https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases>`_, replace ``X.Y`` below with the proper release number. If you prefer to use the development version, you can skip this step.

   .. code-block:: sh

      git checkout X.Y

   Install SCT:

   .. code:: sh

      ./install_sct -y

   .. note::

      At the end of this installation step, you may see the following warnings:

      .. code::

         Check if figure can be opened with matplotlib.......[FAIL] (Using non-GUI backend 'agg')
         Check if figure can be opened with PyQt.............[FAIL] ($DISPLAY not set on X11-supporting system)

      This is expected, because WSL does not come with the ability to display GUI programs by default. Later on in this page, there will be optional GUI settings you can configure for WSL to address these warnings.

   To complete the installation of SCT, run:

   .. code:: sh

      source ~/.bashrc

   You can now use SCT. Your local C drive is located under ``/mnt/c``. You can access it by running:

   .. code:: sh

      cd /mnt/c


WSL Installation with GUI (Optional)
************************************

If you would like to use SCT's GUI features, or if you would like to try FSLeyes within the same Ubuntu environment, first complete the previous "Basic Installation" section, then continue on to the steps below.

#. Download and install `VcXsrv <https://sourceforge.net/projects/vcxsrv/>`_, a program that makes it possible to run Linux GUI programs installed with WSL.

#. Run the newly installed ``XLaunch`` program, then click the following settings:

   - On the "Display settings" page, click "Next".
   - On the "Client startup" page, click "Next".
   - On the "Extra settings" page, check the "Disable access control" box, then click "Next".
   - Click "Finish", then click "Allow access" when prompted by Windows Firewall.
   - You should now see the X Server icon running in the bottom-right system tray in your taskbar.

   .. note::

      ``XLaunch`` must be running each time you wish to use GUI programs in WSL.

#. Next, run the following commands in Ubuntu, depending on the version of WSL you are using.

   WSL1:

   .. code::

      echo "export DISPLAY=localhost:0.0" >> ~/.bashrc
      echo "export LIBGL_ALWAYS_INDIRECT=0" >> ~/.bashrc
      source ~/.bashrc

   WSL2:

   .. code::

      echo "export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0.0" >> ~/.bashrc
      echo "export LIBGL_ALWAYS_INDIRECT=0" >> ~/.bashrc
      source ~/.bashrc

#. Finally, run the ``sct_check_dependencies`` command in Ubuntu to verify that matplotlib and PyQt figures can be opened by SCT.

#. Optionally, you can install FSLeyes using the following commands:

   .. code::

      source ${SCT_DIR}/python/etc/profile.d/conda.sh
      conda create -c conda-forge -p ~/fsleyes_env fsleyes -y
      sudo ln -s ~/fsleyes_env/bin/fsleyes /usr/local/bin/fsleyes

   These instructions will install FSLeyes into a fresh ``conda`` environment, then create a link to FSLeyes so that you can use the ``fsleyes`` command without having to activate the conda environment each time.


-----


.. _docker-installation:

Docker installation
-------------------

`Docker <https://www.docker.com/what-container>`_ is a portable (Linux, macOS, Windows) container platform.

Basic Installation (No GUI)
***************************

First, `install Docker <https://docs.docker.com/install/>`_. Then, follow either of the examples below to create an OS-specific SCT installation.

Option 1: Ubuntu Docker Image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
   sct_testing
   # save the state of the container. Open a new Terminal and run:
   docker ps -a  # list all containers
   docker commit <CONTAINER_ID> <YOUR_NAME>/ubuntu:ubuntu16.04

Option 2: CentOS7 Docker Image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

#. Install Xming
#. Connect to it using Xming/SSH:

   - If you are using Docker Desktop, please download and run (double click) the following script: :download:`sct-win.xlaunch<../../../../contrib/docker/sct-win.xlaunch>`.
   - If you are using Docker Toolbox, please download and run the following script instead: :download:`sct-win_docker_toolbox.xlaunch<../../../../contrib/docker/sct-win_docker_toolbox.xlaunch>`
   - If this is the first time you have done this procedure, the system will ask you if you want to add the remote PC (the docker container) as trust pc, type yes. Then type the password to enter the docker container (by default sct).

**Troubleshooting:**

The graphic terminal emulator LXterminal should appear (if not check the task bar at the bottom of the screen), which allows copying and pasting commands, which makes it easier for users to use it. If there are no new open windows:

- Please download and run the following file: :download:`Erase_fingerprint_docker.sh<../../../../contrib/docker/Erase_fingerprint_docker.sh>`
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
