<#
.SYNOPSIS
    Downloads required simulation cache files directly from Google Drive, bypassing GitHub LFS.
    
.DESCRIPTION
    This script is intended for users who encounter "LFS budget exceeded" errors or want to avoid
    using GitHub LFS bandwidth. It downloads the required .pkl files for the LUT (Look-Up Table) 
    cache directly into the .h2_plant/lut_cache directory.

.EXAMPLE
    .\scripts\setup_lfs_bypass.ps1
#>

$ErrorActionPreference = "Stop"

# Define the target directory relative to the repo root
# Assuming this script is run from the repo root or we can resolve it.
# We'll stick to relative path assuming user runs from root as is common in python projects.
$TargetDir = ".h2_plant\lut_cache"

# Ensure the directory exists
if (-not (Test-Path $TargetDir)) {
    Write-Host "Creating directory: $TargetDir" -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
}

# Dictionary of Filenames and their Google Drive File IDs
$Files = @{
    "lut_CH4_v1.pkl"              = "165hSFAbkBbGcDG7U05cfQQXj6ur8IG8s"
    "lut_CO2_v1.pkl"              = "1PJ90U4Yafe6tVrAR_INOHX7uvE9I9TN4"
    "lut_H2_v1.pkl"               = "1YS2bHl4VafUp0CpUHNIoDCC2aM2RulZ_"
    "lut_H2O_v1.pkl"              = "1U6lPrNaZuydiHMs49Ji5A_-HQcVMoH3t"
    "lut_N2_v1.pkl"               = "1axXcY_2M4A-znlXf_4pBSgLYBSIIDW1_"
    "lut_O2_v1.pkl"               = "1OOQsT5fDJPceQ5aVtkxQYAxWiLIOSPQr"
    "lut_water_saturation_v1.pkl" = "1NuOL8ItX_BWZ7f7ae91vlA82U-aK4vqK"
}

Write-Host "Starting manual download of LFS cache files..." -ForegroundColor Green

foreach ($kvp in $Files.GetEnumerator()) {
    $FileName = $kvp.Key
    $FileId = $kvp.Value
    $OutputPath = Join-Path $TargetDir $FileName
    
    # Construct the direct download URL
    $DownloadUrl = "https://drive.google.com/uc?export=download&id=$FileId"
    
    Write-Host "Downloading $FileName..." -NoNewline
    
    try {
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $OutputPath
        Write-Host " [OK]" -ForegroundColor Green
    }
    catch {
        Write-Host " [FAILED]" -ForegroundColor Red
        Write-Error "Failed to download $FileName. Error: $_"
    }
}


Write-Host "`nAll files downloaded successfully to $TargetDir" -ForegroundColor Green

# 3) (Optional) Re-enable git-lfs for other LFS assets if needed
# git lfs install --local

# 4) Create venv + install dependencies
Write-Host "`nSetting up Python environment..." -ForegroundColor Cyan

# Check if python is available
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $PythonExe = "python"
} elseif (Get-Command "python3" -ErrorAction SilentlyContinue) {
    $PythonExe = "python3"
} else {
    Write-Warning "Python not found! Skipping environment setup."
    exit
}

$VenvDir = ".venv"

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment in $VenvDir..."
    & $PythonExe -m venv $VenvDir
} else {
    Write-Host "Virtual environment already exists."
}

# Activate venv
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path $ActivateScript) {
    Write-Host "Activating virtual environment..."
    . $ActivateScript
} else {
    Write-Warning "Could not find activation script: $ActivateScript"
    # Fallback for non-Windows structure just in case (e.g. bin/activate)
    $ActivateScriptLinux = Join-Path $VenvDir "bin/activate"
     if (Test-Path $ActivateScriptLinux) {
         Write-Warning "Found Linux-style venv. This script is intended for PowerShell on Windows."
     }
}

# Install dependencies
Write-Host "Installing dependencies..."
& $PythonExe -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

Write-Host "`nSetup complete! You can now run the simulation." -ForegroundColor Green

