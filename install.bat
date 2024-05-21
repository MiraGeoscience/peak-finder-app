:: Creates a dedicated Conda environment with this package installed and ready to run in that environment.
::
:: Usage: install.bat [-e]
::
:: Use the optional -e argument to install in editable mode. In that case, any
:: change in the source code will be immediately reflected at execution, and the source folder
:: must not be moved or deleted after installation.


@echo off
setlocal EnableDelayedExpansion

call "%~dp0get_conda_exec.bat"
if !errorlevel! neq 0 (
  pause
  exit /B !errorlevel!
)

set EXTRA_PIP_INSTALL_OPTIONS =

:: if provided and first arg equals "-e", install with pip editable mode
if "%1" == "-e" (
  set EXTRA_PIP_INSTALL_OPTIONS=-e
  shift
)

set PY_VER=3.10

set ENV_NAME=peak-finder-app
set MY_CONDA=!MY_CONDA_EXE:"=!
cd %~dp0
set PYTHONUTF8=1
set CONDA_CHANNEL_PRIORITY=strict

:: all dependencies are installed from conda
set PIP_NO_DEPS=1

set MY_CONDA_ENV_FILE=environments\py-%PY_VER%-win-64.conda.lock.yml
if not exist %MY_CONDA_ENV_FILE% (
  echo "** ERROR: Could not find the conda environment specification file '%MY_CONDA_ENV_FILE%' **"
  pause
  exit /B 1
)

call "!MY_CONDA!" activate base ^
  && call "!MY_CONDA!" env create -y --solver libmamba -n %ENV_NAME% --file %MY_CONDA_ENV_FILE% ^
  && call "!MY_CONDA!" run -n %ENV_NAME% pip install !EXTRA_PIP_INSTALL_OPTIONS! .[dash]

if !errorlevel! neq 0 (
  echo "** ERROR: Installation failed **"
  pause
  exit /B !errorlevel!
)

pause
cmd /k "!MY_CONDA!" activate %ENV_NAME%
