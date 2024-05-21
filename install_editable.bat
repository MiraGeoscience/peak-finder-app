@echo off

:: install with package in editable mode
call "%~dp0install.bat" -e
exit /B !errorlevel!
