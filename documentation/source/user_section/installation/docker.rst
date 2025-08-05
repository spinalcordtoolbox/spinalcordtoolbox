Installation with Docker
------------------------

`Docker <https://www.docker.com/what-container/>`__ is a portable (Linux, macOS, Windows) container platform.

In the context of SCT, it can be used to test SCT in a specific OS environment; this is much faster than running a fully fledged virtual machine.

Installing Docker itself
************************

The instructions for installing Docker itself depend on the host operating system (that is, the operating system used *outside* Docker):

- :ref:`Docker on MacOS <docker-install-macos>`
- :ref:`Docker on Linux <docker-install-linux>`
- :ref:`Docker on Windows <docker-install-windows>`

.. _docker-install-sct:

Installing SCT within Docker
****************************

Once you have installed Docker (and optionally enabled GUI scripts), you can either modify and use this `example Dockerfile for SCT <https://github.com/spinalcordtoolbox/spinalcordtoolbox/tree/master/contrib/docker>`__, or follow the instructions below to manually install SCT inside Docker.

These commands should be run in an interactive terminal within Docker:

.. code:: bash

    # Now inside Docker container, install SCT dependencies
    apt update
    apt install git curl bzip2 libglib2.0-0 libgl1-mesa-glx libxrender1 libxkbcommon-x11-0 libdbus-1-3 gcc
    # Note for above: libglib2.0-0, libgl1-mesa-glx, libxrender1, libxkbcommon-x11-0, libdbus-1-3 are required by PyQt
    # Install SCT
    git clone https://github.com/spinalcordtoolbox/spinalcordtoolbox.git sct
    cd sct
    ./install_sct -y
    source /root/.bashrc
    # Test SCT
    sct_testing

Once the installation is done, you should follow the host operating system-specific commands to save the container (:ref:`MacOS <docker-install-macos>`, :ref:`Linux <docker-install-linux>`, :ref:`Windows <docker-install-windows>`).
