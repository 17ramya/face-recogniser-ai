# Start the React development server on http://localhost:3000
#
#   .\start-frontend.ps1                # npm install on first run, then serve
#   .\start-frontend.ps1 -SkipInstall   # just serve (node_modules already present)
#
# The API must be running too (see start-backend.ps1) or the page shows an
# "API offline" badge, which is exactly what it is there to tell you.

[CmdletBinding()]
param(
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# The dev server has to stay on 3000. src/api.js talks to the API on 3001, and
# Create React App quietly moves to the next free port when 3000 is taken - if
# it lands on 3001 the app starts talking to itself. So pin the port, and fail
# loudly here instead of confusing you in the browser.
$env:PORT = '3000'

function Get-PortOwner([int]$Port) {
    try {
        return @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop) |
            Select-Object -First 1
    } catch {
        return $null  # nothing listening, or the NetTCPIP module is unavailable
    }
}

$busy = Get-PortOwner 3000
if ($busy) {
    $owner = (Get-Process -Id $busy.OwningProcess -ErrorAction SilentlyContinue).ProcessName
    $who = if ($owner) { "PID $($busy.OwningProcess) ($owner)" } else { "PID $($busy.OwningProcess)" }
    throw ("Port 3000 is already in use by $who. The app may already be running - open " +
           'http://localhost:3000. If that is something else, close it and run this again.')
}

if (-not $SkipInstall -and -not (Test-Path (Join-Path $PSScriptRoot 'node_modules'))) {
    Write-Host '[start] installing npm packages - this happens once'
    npm install
}

Write-Host '[start] React dev server on http://localhost:3000 - press Ctrl+C to stop'
npm start
