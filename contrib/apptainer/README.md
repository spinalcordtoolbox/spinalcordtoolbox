# Minimal SCT Apptainer installation w/ Ubuntu 22.04

This directory contains files needed to install a minimal Apptainer (formally Singularity) instance of SCT, for use on High Performance Computing (HPC) clusters which do not support Conda/Mamba. In short, generates a compressed "image" of SCT into a single `.sif` (Singularity Image Format) file, which can then be run as a program, without needing to install or provide SCT's usual dependencies yourself.

For those familiar with Docker, Apptainer is a close analog, with Apptainer `.def` files being equivalent to Docker `docker` files. Likewise, the container resulting from using a `.def` file (a "Singularity Image Format", or `.sif`, file) can then be run as a program via the `apptainer exec {container_name}.sif` command, similar to `docker run {container_name}`.

There are some caveats to installing SCT via Apptainer rather than installing SCT natively, however: 

* Apptainer can only be used and run on Linux. If you want to run it on Windows, you will need to do so through the Windows Subsystem for Linux (WSL), which may not be available on a Windows server. There is currently 0 support for MacOS, period.
* Apptainer containers are isolated from the network, making any SCT commands which require downloading something off of the internet impossible. Currently, this only means `sct_deepseg task -install` will not work; if you need to install a new `sct_deepseg` model, use `sct_model_install.def` instead.
* Due to some `apt` packages requiring user permissions, which are not always given on HPC systems, we need to pin `openssh-client` and `dbus` to get around [this](https://github.com/apptainer/apptainer/issues/1822#issuecomment-2051581258) bug. While this does not appear to impact the installation currently, it is possible that a future update to SCT or one of its dependencies may break as a result. Caveat Emptor!

## Set up

Inside this directory, run:

    APPTAINER_BIND=' ' apptainer build sct.sif sct.def

If you want to run DeepSeg tasks, you can install the requisite models by adding the following argument (replacing `tasks` with a list of space-separated task-names; you can get the valid options listed within the output `sct_deepseg -h`):

    APPTAINER_BIND=' ' apptainer build --build-arg task_installs="tasks" sct.sif sct.def

If you need to install models for a task _after_ the `.sif` file has been created, you can run the following instead:

    APPTAINER_BIND=' ' apptainer build --build-arg task_installs="tasks" sct.sif sct_model_install.def

## Running SCT Tools

Once the `.sif` file has been generated, you can run any SCT command by prepending the following before it and running the command within this directory:

    apptainer exec sct.sif

For example, the following command will display the help output of `sct_deepseg spinalcord`:

    apptainer exec sct.sif sct_deepseg spinalcord -h    
