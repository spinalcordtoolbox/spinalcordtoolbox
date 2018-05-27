.. _installation:

Installation
############

.. contents::
   :local:
..


Prerequisites
*************

SCT runs on Linux and OSX;
`native Windows support is not there
<https://github.com/neuropoly/spinalcordtoolbox/issues/1682>`_ but
Docker_ might be used to run a Linux container on Windows.

For Linux, the prerequisites should be already available on mainstream
distributions.
SCT has been tested on Debian >= 7, Fedora >= 23, Ubuntu >= 14.04,
Gentoo et al.

For OSX, there are no prerequisites.

.. TODO minimum system version?


Recommended Procedure
*********************

When using the recommended procedure, SCT will bundle its own Python
distribution, installed with all the required packages, and using
specific package versions, in order to ensure reproducibility of
results.

Procedure:

#. Retrieve the SCT code from
   https://github.com/neuropoly/spinalcordtoolbox/releases

   - Unpack it to a folder, then open a shell inside


#. Run the installer and follow the instructions

   .. code:: sh

      ./install_sct


Installation using git
**********************

You may have good reasons to want to install a development version of
SCT.

SCT will still bundle its own Python distribution, installed with
all the required packages, and using specific package versions, in
order to ensure reproducibility of results.


Procedure:

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


Docker
******

`Docker <https://www.docker.com/what-container>`_ is a portable
(Linux, OSX, Windows) container platform.

In the context of SCT, it can be used:

- To run SCT on Windows, until SCT can run natively there
- For development testing of SCT, faster than running a full-fledged
  virtual machine
- <your reason here>

See https://github.com/neuropoly/sct_docker for more information.


Hard-core Installation-less SCT usage
*************************************

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
      sct_download_data -d binaries_debian
      popd
      PATH+=":$PWD/bins"

      # Download models & cie
      mkdir data; pushd data; for x in PAM50 gm_model optic_models pmj_models deepseg_sc_models deepseg_gm_models ; do sct_download_data -d $x; done; popd

      # Add path to spinalcordtoolbox to PYTHONPATH
      export PYTHONPATH="$PWD:$PWD/scripts"
