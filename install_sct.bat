@echo off
rem Installation script for SCT on native Windows platforms
rem Usage: install_sct.bat <version>
rem e.g.
rem        install_sct.bat 5.5

rem This option is needed for expanding !git_ref!, which is set (*and expanded*!) inside the 'if' statement below.
rem See also https://stackoverflow.com/q/9102422 for a further description of this behavior.
setLocal EnableDelayedExpansion

rem Try to ensure that Git is available on the PATH prior to invoking `git clone` to avoid 'command not found' errors
rem   - See also: https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3912
rem NB: This *should* be handled by the git installer, and we even have reports of people running git --version
rem successfully and still getting an error. However, there are perhaps situations where someone has installed git but
rem hasn't refreshed their terminal. Manually modifying the PATH is a bit of a hacky workaround, especially if Git has
rem been installed somewhere else, but if this mitigates a user post on the forum, this will save us some dev time.
PATH=%PATH%;C:\Program Files\Git
git --version >nul 2>&1 || (
    echo ### git not found. Make sure that git is installed ^(and a fresh Command Prompt window has been opened^) before running the SCT installer.
)

if exist .git\ (
  rem If install_sct.bat is being run from a git repository, we assume that this is a git clone of SCT
  rem So, stay in this folder, skip git clone, and assume that we want to install SCT from the current state of the repository
  pushd .
  echo ### Current working directory is a git repository. Installing SCT from current state of the repository:
  echo:
  git status
) else (
  rem Not an in-place install, so go to user's home directory
  pushd %HOMEPATH%

  rem Check to see if we're going to git clone into an existing installation of SCT
  if exist spinalcordtoolbox\ (
    echo ### Previous spinalcordtoolbox installation found at %HOMEPATH%\spinalcordtoolbox.
    rem NB: The rmdir command will output 'spinalcordtoolbox\, Are you sure (Y/N)?', so we don't need our own Y/N prompt
    rem     We also use "echo set /p=" here in order to make sure that Y/N text is output on the same line.
    echo|set /p="### Continuing will overwrite the existing installation directory "
    rmdir /s spinalcordtoolbox\ || goto error
    if exist spinalcordtoolbox\ (
      echo ### spinalcordtoolbox\ not removed. Quitting installation...
      goto exit
    )
  )

  rem Set git ref. If no git ref is specified when calling `install_sct.bat`, use a default instead.
  if [%1]==[] (
    set git_ref=master
  ) else (
    set git_ref=%1
  )

  rem Download SCT and check out the branch requested by the user
  echo:
  echo ### Downloading SCT source code ^(@ !git_ref!^) to %HOMEPATH%\spinalcordtoolbox...
  git clone -b !git_ref! --single-branch --depth 1 https://github.com/spinalcordtoolbox/spinalcordtoolbox.git || goto error
  cd spinalcordtoolbox
)

rem Create and activate virtual environment to install SCT into
echo:
echo ### Using Python to create virtual environment...
python -m venv venv_sct || goto error
call venv_sct\Scripts\activate.bat || goto error
echo Virtual environment created and activated successfully!

rem Install SCT and its requirements
if exist requirements-freeze.txt (
  set requirements_file=requirements-freeze.txt || goto error
) else (
  set requirements_file=requirements.txt || goto error
)
echo:
echo ### Installing SCT and its dependencies from %requirements_file%...
rem Pip needs to be upgraded because default p3.7 pip won't resolve dependency conflicts correctly
python -m pip install --upgrade pip || goto error
pip install -r %requirements_file% || goto error
pip install -e . || goto error

rem Install external dependencies
echo:
echo ### Downloading model files and binaries...
FOR %%D IN (PAM50 gm_model optic_models pmj_models deepseg_sc_models deepseg_gm_models deepseg_lesion_models c2c3_disc_models binaries_win deepreg_models) DO sct_download_data -d %%D -k || goto error

rem Copying SCT scripts to an isolated folder (so we can add scripts to the PATH without adding the entire venv_sct)
echo:
echo ### Copying SCT's CLI scripts to %CD%\bin\
xcopy %CD%\venv_sct\Scripts\*sct*.* %CD%\bin\ /v /y /q /i || goto error

rem Give further instructions that the user add the Scripts directory to their PATH
echo:
echo ### Installation finished!
echo:
echo To use SCT's command-line scripts in Command Prompt, please follow these instructions:
echo:
echo 1. Open the Start Menu -^> Type 'path' -^> Open 'Edit environment variables for your account'
echo 2. Under the section 'User variables for ____', highlight the 'Path' entry, then click the 'Edit...' button.
echo 3. Click 'New', then copy and paste this directory:
echo:
echo    %CD%\bin\
echo:
echo 4. Click 'OK' three times. You can now access SCT's scripts in the Command Prompt.

rem Return to initial directory and deactivate the virtual environment
goto exit

:error
set cached_errorlevel=%errorlevel%
echo Failed with error #%cached_errorlevel%.

:exit
if "%cached_errorlevel%"=="" set cached_errorlevel=0
popd
where deactivate >nul 2>&1
if %errorlevel% EQU 0 call deactivate
PAUSE
exit /b %cached_errorlevel%
