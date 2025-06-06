.. _apptainer_installation:

Installation with Apptainer
---------------------------

.. versionadded:: 7.1

.. warning::
    This is currently experimental; while we have done our best to test and confirm all functions of SCT run identically to a native installation, we cannot guarantee that all functionalities of SCT currently operate as expected in this context. If you run into any bugs, please report them `on the forum <https://forum.spinalcordmri.org/c/sct>`_.

Like Docker, `Apptainer <https://apptainer.org/docs/user/main/introduction.html>`_ (formerly Singularity) is a portable container platform. It was designed with a focus on being used in "shared system" contexts, where multiple users with different needs require access to the same hardware. If you need to run SCT in this context, and a native install is not possible (as is often the case in High Performance Computer (HPC) clusters), you should install SCT in this way.

Using Apptainer introduces a few caveats to using SCT, however:

- Apptainer containers will only work on Linux-based systems, and will not run on Windows or MacOS.
    - They can be run through Windows Subsystem for Linux (WSL) if needed, however.
- Due to containers being static post-creation, functions which install within or modify SCT (such as ``sct_deepseg -install``) will not work.
    - We have provided a workaround this this, should you need these functions: see :ref:`here <apptainer-task-install>` for details.

Installation
************

This method will install SCT within an Apptainer container, ready for portable use.

#. Install Apptainer if you have not done so already (or activate the module which contains it, if on an shared resource system)

#. Download the ``sct_apptainer_{sct_version}.tar.gz`` file for your desired SCT release from GitHub (the current release is available `here <https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/latest/>`_).

#. Move the file and unpack it. On most Linux machines, this can be done with the following command:

    .. code:: sh

        tar -xzf sct_apptainer_{sct_version}.tar.gz

    If done correctly, you should see four files in the directory; two ``.sh`` scripts (``install_sct_containered_{sct_verion}.sh`` and ``install_deepseg_task.sh``), and two ``.def`` Apptainer definition files (``sct_{sct_version}.def`` and ``sct_model_install.def``).

#. Run the installation script ``install_sct_containered_{sct_version}.sh``. You may optionally provide provide a list of :ref:`sct_deepseg` tasks you want installed as well:

    Basic installation (without any ``sct_deepseg`` tasks)

    .. code:: sh

        ./install_sct_containered_{sct_version}.sh

    Installing the ``spinalcord`` and ``tumor_t2`` tasks as well:

    .. code:: sh

        ./install_sct_containered_{sct_version}.sh spinalcord tumor_t2


If installation ran to completion without error, a ``sct.sif`` file should now be present in the same directory. This can be used to run any SCT command as if SCT were installed locally; just prepend ``apptainer exec sct.sif`` before it. For example, to run a spinal cord segmentation using DeepSeg's ``spinalcord`` task:

.. code:: sh

    apptainer exec sct.sif sct_deepseg spinalcord -i example_T2w.nii.gz

.. _apptainer-task-install:

Installing DeepSeg Tasks Post-Install
*************************************

If you need to install a task after the initial ``sct.sif`` file was created, you can use the following method to bypass the "static" nature of Apptainer containers. Note, however, that each time you do this, the ``.sif`` file is rebuilt, which can take quite a while. To avoid this, try to determine which ``sct_deepseg`` models you'll need as early as possible and install them all at once!

#. Navigate to the directory where you first installed SCT in the prior section.

#. Ensure the following files are still in the directory:
    * ``install_deepseg_task.sh``
    * ``sct_model_install.def``
    * The ``sct.sif`` file you generated in the prior section.

#. Run the following command, replacing ``<task1> <task2>`` with the list of ``sct_deepseg`` task(s) you want to install (i.e. ``spinalcord t2_tumor``):

    .. code:: sh

        ./install_deepseg_task.sh spinalcord t2_tumor

This will update the existing ``sct.sif`` file to one containing SCT with the requested models.
