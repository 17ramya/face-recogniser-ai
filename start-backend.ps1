# Start the face-recognition API (Flask) on http://127.0.0.1:3001
#
#   .\start-backend.ps1                 # create the venv on first run, then serve
#   .\start-backend.ps1 -SkipInstall    # just serve (dependencies already present)
#
# Prefer doing it by hand? See README.md ("Quick start").

[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'

$backend = Join-Path $PSScriptRoot 'backend'
$venv = Join-Path $backend '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
    Write-Host "[start] creating a virtual environment in $venv"
    & $Python -m venv $venv
    if (-not (Test-Path $venvPython)) {
        throw "Could not create the virtual environment. Is Python 3.9+ on PATH? Got: $Python"
    }
}

if (-not $SkipInstall) {
    Write-Host '[start] installing backend requirements (quiet when already satisfied)'
    & $venvPython -m pip install --disable-pip-version-check --quiet --upgrade pip
    & $venvPython -m pip install --disable-pip-version-check --quiet -r (Join-Path $backend 'requirements.txt')
}

# Refuse to die with "OSError: [WinError 10048] address already in use" and
# silence the log - say who is holding the port instead.
function Get-PortOwner([int]$Port) {
    try {
        return @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop) |
            Select-Object -First 1
    } catch {
        return $null  # nothing listening, or the NetTCPIP module is unavailable
    }
}

$apiPort = if ($env:FACE_API_PORT) { [int]$env:FACE_API_PORT } else { 3001 }
$busy = Get-PortOwner $apiPort
if ($busy) {
    $owner = (Get-Process -Id $busy.OwningProcess -ErrorAction SilentlyContinue).ProcessName
    $who = if ($owner) { "PID $($busy.OwningProcess) ($owner)" } else { "PID $($busy.OwningProcess)" }
    throw ("Port $apiPort is already in use by $who. The API may already be running - check " +
           "http://127.0.0.1:$apiPort/api/health. If it is something else, close it (or set " +
           '$env:FACE_API_PORT = ''3002'' and point REACT_APP_API_BASE_URL at the same port).')
}

Write-Host "[start] API on http://127.0.0.1:$apiPort - press Ctrl+C to stop"
Push-Location $backend
try {
    & $venvPython app.py
}
finally {
    Pop-Location
}
