[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project virtual environment was not found: $python"
}

Push-Location $projectRoot
try {
    & $python -m pip install --upgrade pyinstaller
    if ($LASTEXITCODE -ne 0) {
        throw 'PyInstaller installation failed.'
    }

    Remove-Item -LiteralPath (Join-Path $projectRoot 'build') -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath (Join-Path $projectRoot 'dist') -Recurse -Force -ErrorAction SilentlyContinue

    & $python -m PyInstaller --noconfirm --clean .\SignLanguageTranslator.spec
    if ($LASTEXITCODE -ne 0) {
        throw 'PyInstaller build failed.'
    }

    $executable = Join-Path $projectRoot 'dist\SignLanguageTranslator\SignLanguageTranslator.exe'
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Build completed without creating $executable"
    }

    Write-Host "Build completed: $executable"
}
finally {
    Pop-Location
}
