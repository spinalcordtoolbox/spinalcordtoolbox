.. _apptainer_installation:

Installation with Apptainer
---------------------------

.. warning::
    This is currently experimental; while we have done our best to test and confirm all functions of SCT run as would be expected, we cannot guarantee that all functionalities of SCT function identically to a non-containerized installation.

Like Docker, `Apptainer <https://apptainer.org/docs/user/main/introduction.html>`_ (formerly Singularity) is a portable container platform. It was designed with a focus on being used in "shared system" contexts, where multiple users with different needs require access to the same hardware. If you need to run SCT in this context (and a native install is not possible, as is often the case in High Performance Computer (HPC) clusters), you should install SCT in this way.

Using Apptainer introduces a few caveats to using SCT, however:

- Apptainer containers will only work on Linux-based systems, and will not run on Windows or MacOS.
    - They can be run through Windows Subsystem for Linux (WSL) if needed, however.
- Due to containers being static post-creation, functions which install within or modify SCT (such as ``sct_deepseg -install``) will not work.
    - We have provided a workaround this this, should you need these functions: see :ref:`here <apptainer-task-install>` for details.

Installation
************

This method will install the latest SCT version available on the master branch.

#. Install Apptainer if you have not done so already (or activate the module which contains it, if on an shared resource system)

#. Download the requisite files from GitHub:

    .. code:: sh

        curl "raw.githubusercontent.com/spinalcordtoolbox/spinalcordtoolbox/refs/heads/master/contrib/apptainer/sct.def" -o "sct.def"
        curl "raw.githubusercontent.com/spinalcordtoolbox/spinalcordtoolbox/refs/heads/master/contrib/apptainer/install_sct_containered.sh" -o "install_sct_containered.sh"
        # The following may not be required; it enables execution permissions for the file if it doesn't already have it.
        chmod +x "install_sct_containered.sh"

#. Run the installation script. You may also provide provide a list of :ref:`sct_deepseg` tasks you want installed as well:

    Basic installation (without any ``sct_deepseg`` tasks)

    .. code:: sh

        ./install_sct_containered.sh

    Installing the ``spinalcord`` and ``tumor_t2`` tasks as well:

    .. code:: sh

        ./install_sct_containered.sh spinalcord tumor_t2


If installation ran to completion without error, a ``sct.sif`` file should now be present in the directory. This can be used to run any SCT command as if SCT were installed locally; just prepend ``apptainer exec sct.sif`` before it. For example, to run a spinal cord segmentation using DeepSeg:

.. code:: sh

    apptainer exec sct.sif sct_deepseg spinalcord -i example_T2w.nii.gz

.. _apptainer-task-install:

Installing DeepSeg Tasks Post-Install
*************************************

If you need to install a task after the initial ``sct.sif`` file was created, you can use the following instructions. Note, however, that each time you do this, the ``.sif`` file is rebuilt, which can take quite a while to do. To avoid this, try to determine which ``sct_deepseg`` models you'll need as early as possible!

#. Download the requisite files from GitHub:

    .. code:: sh

        curl "raw.githubusercontent.com/spinalcordtoolbox/spinalcordtoolbox/refs/heads/master/contrib/apptainer/sct_model_install.def" -o "sct_model_install.def"
        curl "raw.githubusercontent.com/spinalcordtoolbox/spinalcordtoolbox/refs/heads/master/contrib/apptainer/install_deepseg_task.sh" -o "install_deepseg_task.sh"
        # The following may not be required; it enables execution permissions for the file if it doesn't already have it.
        chmod +x "install_deepseg_task.sh"

#. Run the following command, replacing ``<task1> <task2>`` with the list of ``sct_deepseg`` task(s) you want to install (i.e. ``spinalcord t2_tumor``):

.. code:: sh

        ./install_deepseg_task.sh spinalcord t2_tumor

This will update the existing ``sct.sif`` file to one containing SCT with the requested models.