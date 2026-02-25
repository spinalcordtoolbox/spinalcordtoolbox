# Minimal SCT Apptainer Installation w/ Ubuntu 22.04

**_EXPERIMENTAL! Use at your own risk!_**

This directory contains files needed to install a minimal Apptainer (formally Singularity) instance of SCT, for use on High Performance Computing (HPC) clusters which do not support Conda/Mamba. In short, this will generate a compressed "image" of SCT into a `.sif` (Singularity Image Format) file, which can then be run as a program. This is done without you needing to install or provide SCT's usual dependencies yourself; they are all contained within the image file.

For those familiar with Docker, Apptainer is a close analog, with Apptainer `.def` files being equivalent to Docker `.docker` files. Likewise, the container resulting from using a `.def` file (a "Singularity Image Format", or `.sif`, file) can then be run as a program via the `apptainer exec {container_name}.sif` command, similar to `docker run {container_name}`.

## Caveats

There are a number of considerations you should keep in mind if running SCT from an Apptainer image:

* Apptainer can only be used and run on Linux. If you want to run it on Windows, you will need to do so through the Windows Subsystem for Linux (WSL), which may not be available on a Windows server. MacOS currently has no support for Apptainer whatsoever.
* Apptainer images are static, and isolated from any networks your device may have. This prevents the installation of new DeepSeg models with `sct_deepseg -install`. Instead, you define the models you want SCT to have access to during the initial `.sif` file creation; if you must install a model afterward, use `sct_model_install.def` instead.
* With datasets being contained within the image alongside the SCT installation, they can only be accessed when you are running a command with the SCT image. Currently only default datasets are available, and other datasets cannot be downloaded using `sct_download_data`. 
* Due to some `apt` packages requiring user permissions, which are not always given on HPC systems, we need to pin `openssh-client` and `dbus` to get around [this](https://github.com/apptainer/apptainer/issues/1822#issuecomment-2051581258) bug. While this does not appear to impact the installation currently, it is possible that a future update to SCT or one of its dependencies may break as a result. Caveat Emptor!

## Installation

### Prerequisites

To build the SCT Apptainer image, you will need the following:

* A Linux-based OS (see Caveats above).
* An installation of [Apptainer](https://apptainer.org/docs/admin/main/installation.html).
* A copy of this folder, with all of its files.

### Creating the Container

**NOTE:** If at all possible, we recommend doing this on the machine you intend to run SCT on; the resulting image is quite large (more than 2 GigaBytes), and copying it over post-creation can take quite a while depending on your network.

Firstly, navigate to the copy of this directory you have on the machine. You should see a copy of this `README.md` file, alongside four files: 
* `install_sct_containered.sh`: Will be used to create the initial SCT Apptainer file.
* `install_deepseg_model.sh`: Installs new DeepSeg models to an existing container.
* `sct.def`: An Apptainer definition file which designates how the SCT container will be defined.
* `model_install.def`: An Apptainer definition file which defines how to install a new `sct_deepseg` into an existing SCT container.

Once there, run the following command:

    ./install_sct_containered.sh

This will usually take a few minutes to complete. Optionally, if you know you will need to run certain `sct_deepseg` tasks, you can list them after the command to install their requisite models immediately (replacing the `task` entries with the task name(s) you want to run):

    ./install_sct_containered.sh task1 task2 task3

If you need to install a `sct_deepseg` model after the initial `sct` container is completed, you can do so with the following command instead: 

    ./install_deepseg_task.sh task1 task2 task3

Note that this can take a substantial amount of time, as to do this the `.sif` file must be packed and unpacked each time `install_deepseg_task.sh` is called. You should really try to avoid using this as much as possible. 

## Using SCT From Apptainer

### Running an SCT command
Once the `sct.sif` file has been generated, you can run any SCT command by prepending the following before it and running the command within this directory:

    apptainer exec sct.sif

For example, the following command will display the help output of `sct_deepseg spinalcord`:

    apptainer exec sct.sif sct_deepseg spinalcord -h    

### Using SCT Datasets

If you need to access a dataset managed by SCT (such as the PAM50 dataset), they can be referenced using `root/sct/data/{dataset_name}` while running a command using the `.sif` file. Note that only datasets downloaded by default (`PAM50`, `PAM50_normalized_metrics`, and `sct_testing_data`) are currently available. 
