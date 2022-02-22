:: Usage: install_sct.bat <version>
:: e.g.
::        install_sct.bat 5.5

:: Fetch version number (should be git branch, git tag, etc.)
set git_ref=%1

:: Save present working directory to return to later
FOR /F "tokens=*" %%G IN ('cd') do (SET PWD=%%G)

:: Go to user's home directory
cd %HOMEPATH%

:: Download SCT and check out the branch requested by the user
git clone -b %git_ref% --single-branch --depth 1 https://github.com/spinalcordtoolbox/spinalcordtoolbox.git || goto :error

:: Create and activate virtual environment to install SCT into
cd spinalcordtoolbox
python -m venv venv_sct || goto :error
CALL venv_sct\Scripts\activate.bat || goto :error

:: Upgrade pip because default p3.7 pip won't resolve dependency conflicts correctly
python -m pip install --upgrade pip || goto :error

:: Install SCT and its requirements
pip install -r requirements.txt || goto :error
pip install -e . || goto :error

:: Install external dependencies
FOR %%D IN (PAM50 gm_model optic_models pmj_models deepseg_sc_models deepseg_gm_models deepseg_lesion_models c2c3_disc_models binaries_win) DO sct_download_data -d %%D -k || goto :error

:: Deactivate virtual environment
deactivate

:: Return to initial directory
cd %PWD%


:: Give further instructions that the user add the Scripts directory to their PATH

:error
cd %PWD%
echo Failed with error #%errorlevel%.
exit /b %errorlevel%

