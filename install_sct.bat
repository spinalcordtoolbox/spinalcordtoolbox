:: Usage: install_sct.bat <version>
:: e.g.
::        install_sct.bat 5.5

:: Turn off command echoing for nicer output
@echo off

:: Fetch version number (should be git branch, git tag, etc.)
set git_ref=%1

:: Save present working directory to return to later
FOR /F "tokens=*" %%G IN ('cd') do (SET PWD=%%G)

:: Go to user's home directory
cd %HOMEPATH%

:: Download SCT and check out the branch requested by the user
echo:
echo ### Downloading SCT source code to %HOMEPATH%...
git clone -b %git_ref% --single-branch --depth 1 https://github.com/spinalcordtoolbox/spinalcordtoolbox.git || goto :error

:: Create and activate virtual environment to install SCT into
echo:
echo ### Using Python to create virtual environment...
cd spinalcordtoolbox
python -m venv venv_sct || goto :error
CALL venv_sct\Scripts\activate.bat || goto :error
echo Virtual environment created and activated successfully!

:: Install SCT and its requirements
echo:
echo ### Installing SCT and its dependencies...
:: Pip needs to be upgraded because default p3.7 pip won't resolve dependency conflicts correctly
python -m pip install --upgrade pip || goto :error
pip install -r requirements.txt || goto :error
pip install -e . || goto :error

:: Install external dependencies
echo:
echo ### Downloading model files and binaries...
FOR %%D IN (PAM50 gm_model optic_models pmj_models deepseg_sc_models deepseg_gm_models deepseg_lesion_models c2c3_disc_models binaries_win) DO sct_download_data -d %%D -k || goto :error

:: Deactivate virtual environment
deactivate

:: Give further instructions that the user add the Scripts directory to their PATH
echo:
echo ### Installation finished!
echo:
echo To use SCT's command-line scripts in Command Prompt, please follow these instructions:
echo:
echo 1. Open the ^Start Menu -^> Type 'path' -^> Open 'Edit the System Environment Variables'
echo 2. Click the 'Environment variables...' button
echo 3. Under the section 'User variables for ____', highlight the 'Path' entry, then click the 'Edit...' button.
echo 4. Click 'New', then copy and paste this directory:
echo:
echo    %CD%\%venv_sct\Scripts
echo:
echo: 5. Click 'OK' three times. You can now access SCT's scripts in the Command Prompt.

:: Return to initial directory
cd %PWD%

exit /b 0

:error
cd %PWD%
echo Failed with error #%errorlevel%.
exit /b %errorlevel%

