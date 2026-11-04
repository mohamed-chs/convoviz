# Install convoviz on Windows (PowerShell):
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/mohamed-chs/convoviz/main/install.ps1 | iex"
# It installs uv (https://docs.astral.sh/uv/) if needed, then convoviz as a uv tool.
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

uv tool install --upgrade convoviz
Write-Host ""
Write-Host "Done. Open a new terminal and run: convoviz"
