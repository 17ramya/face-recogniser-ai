@echo off
rem Start the face-recognition API (Flask) on http://127.0.0.1:3001
rem
rem The same as start-backend.ps1, but runnable from cmd.exe and without
rem changing the PowerShell execution policy on the machine.
rem   start-backend.cmd                 create the venv on first run, then serve
rem   start-backend.cmd -SkipInstall    just serve

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-backend.ps1" %*
