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

Windows subsystem for Linux (WSL) is available on Windows 10 and it makes it possible to run native Linux programs, such as SCT.

#. Install Windows Subsystem for Linux (WSL)

   - Install `Xming <https://sourceforge.net/projects/xming/>`_.

   - Install  `Windows subsystem for linux and initialize it <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_.

       - Please install the Ubuntu 18.04 LTS distro.

#. Environment preparation

   Run the following command to install various packages that will be needed to install FSL and SCT. This will require your password

   .. code-block:: sh

      sudo apt-get -y update
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

      git clone https://github.com/neuropoly/spinalcordtoolbox.git sct
      cd sct

   To select a `specific release <https://github.com/neuropoly/spinalcordtoolbox/releases>`_, replace X.Y.Z below with the proper release number. If you prefer to use the development version, you can skip this step.

   .. code-block:: sh

      git checkout X.Y.Z

   Install SCT:

   .. code:: sh

      yes | ./install_sct

   To complete the installation of these software run:

   .. code:: sh

      cd ~
      source .profile
      source .bashrc

   You can now use SCT. Your local C drive is located under ``/mnt/c``. You can access it by running:

   .. code:: sh

      cd /mnt/c

#. OPTIONAL: Install FSLeyes

   FSLeyes is a viewer for NIfTI images. SCT features a plugin script to make SCT functions integrated into 
   FSLeyes' graphical user interface. To benefit from this functionality, you will need to install FSLeyes.
   
   Install the C/C++ compilers required to use wxPython:
   
   .. code-block:: sh

           sudo apt-get install build-essential
           sudo apt-get install libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libwebkitgtk-3.0-dev
           sudo apt-get install libjpeg-turbo8-dev libtiff5-dev libsdl1.2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libnotify-dev freeglut3-dev
           
   Activate SCT's conda environment (to run each time you wish to use FSLeyes):
   
   .. code-block:: sh
   
           source ${SCT_DIR}/python/etc/profile.d/conda.sh
           conda activate venv_sct

   Install wxPython using conda:
   
   .. code-block:: sh

           yes | conda install -c anaconda wxpython
           
   Install FSLeyes using conda-forge:
   
   .. code-block:: sh

           yes | conda install -c conda-forge fsleyes

   To use FSLeyes, run Xming from your computer before entering the fsleyes command.
   
   .. important::

      Each time you wish to use FSLeyes, you first need to activate SCT's conda environment (see above).


Install with Docker
-------------------

`Docker <https://www.docker.com/what-container>`_ is a portable (Linux, macOS, Windows) container platform.

In the context of SCT, it can be used:

- To run SCT on Windows, until SCT can run natively there
- For development testing of SCT, faster than running a full-fledged
  virtual machine
- <your reason here>

See https://github.com/neuropoly/sct_docker for more information. We also provide a
`tutorial to install SCT via Docker <https://github.com/neuropoly/spinalcordtoolbox/wiki/testing#run-docker-image>`_.


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



