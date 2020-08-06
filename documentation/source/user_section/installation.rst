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

  * macOS >= 10.12
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



