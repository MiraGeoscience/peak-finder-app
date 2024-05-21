@echo off

:: install with package in editable mode
call "%~dp0install_in_analyst.bat" -e %*
exit /B !errorlevel!
