@echo off
rem Installation script for SCT on native Windows platforms
rem
rem Copyright (c) 2022 Polytechnique Montreal <www.neuro.polymtl.ca>
rem License: see the file LICENSE
rem
rem Usage: install_sct.bat <version>
rem e.g.
rem        install_sct.bat 5.5

echo:
echo *******************************
echo * Welcome to SCT installation *
echo *******************************

rem This option is needed for expanding !git_ref!, which is set (*and expanded*!) inside the 'if' statement below.
rem See also https://stackoverflow.com/q/9102422 for a further description of this behavior.
setLocal EnableDelayedExpansion

set TMP_DIR=%temp%\tmp-%RANDOM%%RANDOM%
mkdir %TMP_DIR%

rem Try to ensure that Git is available on the PATH prior to invoking `git clone` to avoid 'command not found' errors
rem   - See also: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3912
rem NB: This *should* be handled by the git installer, and we even have reports of people running git --version
rem successfully and still getting an error. However, there are perhaps situations where someone has installed git but
rem hasn't refreshed their terminal. Manually modifying the PATH is a bit of a hacky workaround, especially if Git has
rem been installed somewhere else, but if this mitigates a user post on the forum, this will save us some dev time.
PATH=%PATH%;C:\Program Files\Git
git --version >nul 2>&1 || (
    echo ### git not found. Make sure that git is installed ^(and a fresh Command Prompt window has been opened^) before running the SCT installer.
    goto error
)

rem Default value: 'master', however this value is updated on stable release branches.
set git_ref=master

rem Check to see if the PWD contains the project source files (using `version.txt` as a proxy for the entire source dir)
rem If it exists, then we can reliably access source files (`version.txt`, `requirements-freeze.txt`) from the PWD.
if exist spinalcordtoolbox\version.txt (
  set SCT_SOURCE=%cd%
rem If version.txt isn't present, then the installation script is being run by itself (i.e. without source files).
rem So, we need to clone SCT to a TMPDIR to access the source files, and update SCT_SOURCE accordingly.
) else (
  set SCT_SOURCE=%TMP_DIR%\spinalcordtoolbox
  echo:
  echo ### Source files not present. Downloading source files ^(@ !git_ref!^) to !SCT_SOURCE!...
  git clone -b !git_ref! --single-branch --depth 1 https://github.com/spinalcordtoolbox/spinalcordtoolbox.git !SCT_SOURCE!
  rem Since we're git cloning into a TMPDIR, this can never be an "in-place" installation, so we force "package" instead.
  set SCT_INSTALL_TYPE=package
)

rem Get installation type if not already specified
if [%SCT_INSTALL_TYPE%]==[] (
  rem The file 'requirements-freeze.txt` only exists for stable releases
  if exist %SCT_SOURCE%\requirements-freeze.txt (
    set SCT_INSTALL_TYPE=package
  rem If it doesn't exist, then we can assume that a dev is performing an in-place installation from master
  ) else (
    set SCT_INSTALL_TYPE=in-place
  )
)

rem Fetch the version of SCT from the source file
for /F %%g IN (%SCT_SOURCE%\spinalcordtoolbox\version.txt) do (set SCT_VERSION=%%g)

echo:
echo ### SCT version ......... %SCT_VERSION%
echo ### Installation type ... %SCT_INSTALL_TYPE%

rem if installing from git folder, then becomes default installation folder
if %SCT_INSTALL_TYPE%==in-place (
  set SCT_DIR=%SCT_SOURCE%
) else (
  set SCT_DIR=%USERPROFILE%\sct_%SCT_VERSION%
)

rem Allow user to set a custom installation directory
:while_loop_sct_dir
  echo:
  echo ### SCT will be installed here: [%SCT_DIR%]
  set keep_default_path=yes
  :while_loop_path_agreement
    set /p keep_default_path="### Do you agree? [y]es/[n]o: "
    echo %keep_default_path% | findstr /b [YyNn] >nul 2>&1 || goto :while_loop_path_agreement
  :done_while_loop_path_agreement

  echo %keep_default_path% | findstr /b [Yy] >nul 2>&1
  if %errorlevel% EQU 0 (
    rem user accepts default path, so exit loop
    goto :done_while_loop_sct_dir
  )

  rem user enters new path
  echo:
  echo ### Choose install directory.
  set /p new_install="### Warning^! Give full path ^(e.g. C:\Users\username\sct_v3.0^): "

  rem Check user-selected path for spaces
  rem TODO: This may no longer be true as of a patch made to Mamba in Dec. 2024!
  if not "%new_install%"=="%new_install: =%" (
       echo ### WARNING: Install directory %new_install% contains spaces.
       echo ### SCT uses conda, which does not permit spaces in installation paths.
       echo ### More details can be found here: https://github.com/ContinuumIO/anaconda-issues/issues/716
       echo:
       goto :while_loop_sct_dir
  )

  rem Validate the user's choice of path
  if exist %new_install% (
    rem directory exists, so update SCT_DIR and exit loop
    echo ### WARNING: '%new_install%' already exists. Files will be overwritten.
    set SCT_DIR=%new_install%
    goto :done_while_loop_sct_dir
  ) else (
    if [%new_install%]==[]  (
      rem If no input, asking again, and again, and again
      goto :while_loop_sct_dir
    ) else (
      set SCT_DIR=%new_install%
      goto :done_while_loop_sct_dir
    )
  )
:done_while_loop_sct_dir

rem Create directory
if not exist %SCT_DIR% (
  mkdir %SCT_DIR% || goto error
)

rem Copy files to destination directory
echo:
if not %SCT_DIR%==%SCT_SOURCE% (
  echo ### Copying source files from %SCT_SOURCE% to %SCT_DIR%
  xcopy /s /e /q /y %SCT_SOURCE% %SCT_DIR% || goto error
) else (
  echo ### Skipping copy of source files ^(source and destination folders are the same^)
)

rem Clean old install setup in bin/ if existing
if exist %SCT_DIR%\bin\ (
  echo ### Removing sct and isct softlink inside the SCT directory...
  del %SCT_DIR%\bin\sct_* || goto error
  del %SCT_DIR%\bin\isct_* || goto error
)
rem Remove old python folder
if exist %SCT_DIR%\python\ (
  echo ### Removing existing 'python' folder inside the SCT directory...
  rmdir /s /q %SCT_DIR%\python\ || goto error
)
rem Remove old '.egg-info` folder created by editable installs
if exist %SCT_DIR%\spinalcordtoolbox.egg-info\ (
  echo ### Removing existing '.egg-info' folder inside the SCT directory...
  rmdir /s /q %SCT_DIR%\spinalcordtoolbox.egg-info\ || goto error
)

rem Move into the SCT installation directory
pushd %SCT_DIR% || goto error

rem Install portable miniforge instance. (Command source: https://github.com/conda/conda/issues/1977)
echo:
echo ### Downloading Miniforge installer...
curl -o %TMP_DIR%\miniconda.exe -L https://github.com/conda-forge/miniforge/releases/download/24.11.2-1/Miniforge3-Windows-x86_64.exe
echo:
echo ### Installing portable copy of Miniforge...
start /wait "" %TMP_DIR%\miniconda.exe /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /NoRegistry=1 /S /D=%cd%\python

rem Create and activate miniforge environment to install SCT into
echo:
echo ### Using Conda to create virtual environment...
python\Scripts\conda create -y -p python\envs\venv_sct python=3.10 || goto error
CALL python\Scripts\activate.bat python\envs\venv_sct || goto error
echo Virtual environment created and activated successfully!

rem Install SCT and its requirements
if exist requirements-freeze.txt (
  set requirements_file=requirements-freeze.txt || goto error
) else (
  set requirements_file=requirements.txt || goto error
)
echo:
echo ### Installing SCT and its dependencies from %requirements_file%...
rem Skip pip==21.2 to avoid dependency resolver issue (https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3593)
python\envs\venv_sct\python -m pip install -U "pip^!=21.2.*" || goto error
python\envs\venv_sct\Scripts\pip install -r %requirements_file% || goto error
python\envs\venv_sct\Scripts\pip install -e . --use-pep517 || goto error

rem Install external dependencies
echo:
echo ### Downloading model files and binaries...
python\envs\venv_sct\Scripts\sct_download_data -d binaries_win -k
python\envs\venv_sct\Scripts\sct_download_data -d default -k
python\envs\venv_sct\python -c "import spinalcordtoolbox.deepseg.models; spinalcordtoolbox.deepseg.models.install_default_models()"

rem Copying SCT scripts to an isolated folder (so we can add scripts to the PATH without adding the entire venv_sct)
echo:
echo ### Copying SCT's CLI scripts to %CD%\bin\
xcopy %CD%\python\envs\venv_sct\Scripts\*sct*.* %CD%\bin\ /v /y /q /i || goto error

echo ### Checking installation...
python\envs\venv_sct\Scripts\sct_check_dependencies

rem Give further instructions that the user add the Scripts directory to their PATH
echo:
echo ### Installation finished!
echo:
echo To use SCT's command-line scripts in Command Prompt, please follow these instructions:
echo:
echo 1. Open the Start Menu -^> Type 'edit environment' -^> Open 'Edit environment variables for your account'
echo 2. Click 'New', then enter 'SCT_DIR' for the variable name. For the value, copy and paste this directory:
echo:
echo    %CD%
echo:
echo 3. Click 'OK', then click on the 'Path' variable, then click the 'Edit...' button.
echo 4. Click 'New', then copy and paste this directory:
echo:
echo    %CD%\bin\
echo:
echo 5. Click 'OK' three times. You can now access SCT's scripts in the Command Prompt.
echo:
echo If you have any questions or concerns, feel free to create a new topic on SCT's forum:
echo   --^> https://forum.spinalcordmri.org/c/sct

rem Return to initial directory and deactivate the virtual environment
goto exit

:error
set cached_errorlevel=%errorlevel%
echo:
echo Installation failed with error code %cached_errorlevel%.
echo Please copy and paste the installation log in a new topic on SCT's forum:
echo   --^> https://forum.spinalcordmri.org/c/sct

:exit
if "%cached_errorlevel%"=="" set cached_errorlevel=0
popd
where deactivate >nul 2>&1
if %errorlevel% EQU 0 call conda deactivate
PAUSE
exit /b %cached_errorlevel%
