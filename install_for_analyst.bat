:: Creates an environment for my-app under the Python environment location used by Geoscience Analyst.
:: Assumes Analyst is installed under "%ProgramFiles%\Mira Geoscience\Geoscience ANALYST"
::
:: Usage: install_in_mirageo.bat [--conda default] [--name <env-name>]
::  - To install for regular conda, call with argument: --conda default
::  - To specify a different environment name, call with argument: --name <env-name>

set "MIRA_CMD_RUNNER_DIR=%ProgramFiles%\Mira Geoscience\Geoscience ANALYST\CmdRunner"
set curdir=%~dp0
"%MIRA_CMD_RUNNER_DIR%\MambaEnvRunner.exe" --install "%curdir%\environments\conda-py-3.10-win-64.lock.yml" --with-pip-deps %* --run pip install -e .[dash] %*
