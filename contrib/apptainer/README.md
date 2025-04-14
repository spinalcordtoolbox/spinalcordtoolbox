# Minimal SCT Apptainer installation w/ Ubuntu 22.04

This directory contains a minimal Apptainer definition, creating a container with the most recent master commit of the Spinal Cord Toolbox. Due to the restrictions Apptainer (and most Compute Clusters) impose on tools installed within them, this comes with a few notable restrictions:

* `sct_deepseg task -install` will NOT work, nor will any command that requires networks access. If you need models installed, use the `sct_model_install.def`, modifying the `From` tag to point to your SCT `sif` file.
* We need to pin `openssh-client` and `dbus` to get around [this](https://github.com/apptainer/apptainer/issues/1822#issuecomment-2051581258) bug. This doesn't appear to impact the installation, but `openssh-client` in particular could break the installation of some SCT components. Caveat Emptor!


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
