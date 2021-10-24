.. _fsleyes_installation:

******************
Installing FSLeyes
******************

FSLeyes is a viewer for NIfTI images. SCT features a plugin script to make SCT functions integrated into
FSLeyes' graphical user interface. To benefit from this functionality, you will need to install FSLeyes.

Windows via WSL
---------------

Install the C/C++ compilers required to use wxPython:

.. code-block:: sh

    sudo apt-get install build-essential
    sudo apt-get install libgtk2.0-dev libgtk-3-dev libwebkitgtk-dev libwebkitgtk-3.0-dev
    sudo apt-get install libjpeg-turbo8-dev libtiff5-dev libsdl1.2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libnotify-dev freeglut3-dev

Activate SCT's conda environment (to run each time you wish to use FSLeyes):

.. code-block:: sh

    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Set the channel priority to strict (`as recommended by conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html#strict-channel-priority>`_), then install FSLeyes using conda-forge:

.. code-block:: sh

    conda config --set channel_priority strict
    conda install -y -c conda-forge fsleyes

To use FSLeyes, run Xming from your computer before entering the fsleyes command.

.. warning::

    Each time you wish to use FSLeyes, you first need to activate SCT's conda environment (see above).

MacOS
-----

You can either install ``FSLeyes`` directly using ``conda-forge``, or you can install the entire
``FSL`` package, which includes ``FSLeyes``.

Install from conda-forge
^^^^^^^^^^^^^^^^^^^^^^^^

First, activate the ``conda`` virtual environment:

.. code-block:: sh

    conda activate venv_sct

Next, install ``FSLeyes`` using ``conda-forge``:

.. code-block:: sh

    conda install -c conda-forge -y fsleyes


Install from FSL
^^^^^^^^^^^^^^^^

You can find instructions for installing ``FSL`` here:
`FSL Installation <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_.

1. Download the installer.
2. Make sure XQuartz is installed: https://www.xquartz.org/.
3. Run the install script using ``python 2`` (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/MacOsX).


MacOS Big Sur
^^^^^^^^^^^^^

Currently, MacOS Big Sur (10.16) is not fully supported by ``FSLeyes``. The best method seems to
be installing via the ``FSL`` system. When asked for which operating system you have, you will
not see ``Big Sur (10.16)`` listed, so just select ``Catalina (10.15)``.

If you are still having issues, you may need to edit one of the source files:

Use a text editor to open the ``ctypesloader.py`` file:

.. code-block:: sh

    atom ${FSLDIR}/fslpython/envs/fslpython/lib/python3.x/site-packages/OpenGL/platform/ctypesloader.py

Search for the following line:

.. code-block:: python

    fullName = util.find_library( name )

Comment this line out and add these 4 lines:

.. code-block:: python

    # fullName = util.find_library( name )
    if name == "OpenGL":
      fullName = "/System/Library/Frameworks/OpenGL.framework/OpenGL"
    elif name == "GLUT":
      fullName = "/System/Library/Frameworks/GLUT.framework/GLUT"
