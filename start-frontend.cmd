@echo off
rem Start the React development server on http://localhost:3000
rem
rem The same as start-frontend.ps1, but runnable from cmd.exe and without
rem changing the PowerShell execution policy on the machine.
rem   start-frontend.cmd                npm install on first run, then serve
rem   start-frontend.cmd -SkipInstall   just serve

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-frontend.ps1" %*
