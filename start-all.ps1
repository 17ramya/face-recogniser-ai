# Start both halves of the project in one go.
#
#   .\start-all.ps1
#
# The backend opens in its own PowerShell window (so its log stays visible);
# the front end runs in this window. Close both windows to stop everything.

[CmdletBinding()]
param(
    [int]$WaitSeconds = 3
)

$ErrorActionPreference = 'Stop'

# Start-Process joins -ArgumentList without quoting, so a path containing a
# space (like "C:\Users\RAMYA S\...") must be quoted by hand.
$backendScript = Join-Path $PSScriptRoot 'start-backend.ps1'
$quoted = '"' + $backendScript + '"'

Write-Host '[start] launching the API in a new window...'
Start-Process -FilePath 'powershell' -ArgumentList @(
    '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', $quoted
) | Out-Null

Start-Sleep -Seconds $WaitSeconds
Write-Host '[start] launching the React dev server...'
& (Join-Path $PSScriptRoot 'start-frontend.ps1')
