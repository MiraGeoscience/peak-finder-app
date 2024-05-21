:: Creates a dedicated environment for using this package under the Python environment location
:: used by Geoscience Analyst.
:: Assumes Analyst is installed under "%ProgramFiles%\Mira Geoscience\Geoscience ANALYST".
::
:: Usage: install_in_analyst.bat [-e] [--conda default] [--name <env-name>]
::  - To install for regular conda, call with argument: --conda default
::  - To specify a different environment name, call with argument: --name <env-name>
::
:: Use the optional -e argument (in first position) to install in editable mode. In that case, any
:: change in the source code will be immediately reflected at execution, and the source folder
:: must not be moved or deleted after installation.


@echo off
setlocal EnableDelayedExpansion

set "MIRA_CMD_RUNNER_DIR=%ProgramFiles%\Mira Geoscience\Geoscience ANALYST\CmdRunner"
set curdir=%~dp0

set EXTRA_PIP_INSTALL_OPTIONS =

:: if provided and first arg equals "-e", install with pip editable mode
if "%1" == "-e" (
  set EXTRA_PIP_INSTALL_OPTIONS=-e
  shift
)

"%MIRA_CMD_RUNNER_DIR%\MambaEnvRunner.exe" --name peak-finder-app --install "%curdir%environments\py-3.10-win-64.conda.lock.yml" ^
  --run "pip install !EXTRA_PIP_INSTALL_OPTIONS! .[dash]" %1 %2 %3 %4 %5 %6 %7 %8 %9

pause
