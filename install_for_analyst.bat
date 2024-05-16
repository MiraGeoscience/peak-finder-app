:: Creates an environment for my-app under the Python environment location used by Geoscience Analyst.
:: Assumes Analyst is installed under "%ProgramFiles%\Mira Geoscience\Geoscience ANALYST"
::
:: Usage: install_in_mirageo.bat [--conda default] [--name <env-name>]
::  - To install for regular conda, call with argument: --conda default
::  - To specify a different environment name, call with argument: --name <env-name>

set "MIRA_CMD_RUNNER_DIR=%ProgramFiles%\Mira Geoscience\Geoscience ANALYST\CmdRunner"
set curdir=%~dp0
"%MIRA_CMD_RUNNER_DIR%\MambaEnvRunner.exe" --name peak-finder-app --install "%curdir%environments\py-3.10-win-64.conda.lock.yml" ^
  --run "pip install -e .[dash]" %*
