<#
.SYNOPSIS
    Downloads required simulation cache files and sets up the Python environment.
    
.DESCRIPTION
    1. Downloads LFS cache files directly from Google Drive to bypass GitHub LFS quotas.
    2. Creates a local virtual environment (.venv).
    3. Explicitly installs dependencies into that virtual environment.

.EXAMPLE
    .\scripts\setup_lfs_bypass.ps1
#>

$ErrorActionPreference = "Stop"

# --- PART 1: LFS CACHE DOWNLOAD ---

# Define the target directory relative to the repo root
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


# --- PART 2: PYTHON ENVIRONMENT SETUP ---

Write-Host "`nSetting up Python environment..." -ForegroundColor Cyan

# 1. Find System Python (only used to CREATE the venv)
if (Get-Command "python" -ErrorAction SilentlyContinue) {
    $SystemPython = "python"
} elseif (Get-Command "python3" -ErrorAction SilentlyContinue) {
    $SystemPython = "python3"
} else {
    Write-Warning "Python not found! Skipping environment setup."
    exit
}

$VenvDir = ".venv"

# 2. Create Virtual Environment
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment in $VenvDir using $SystemPython..."
    & $SystemPython -m venv $VenvDir
} else {
    Write-Host "Virtual environment already exists."
}

# 3. Define path to the VENV Python executable
# This ensures we are installing into the local .venv, not the global system
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Could not find python executable at $VenvPython. The virtual environment creation might have failed."
    exit
}

# 4. Install dependencies using the VENV python explicitly
Write-Host "Installing dependencies into .venv..."
# Upgrade pip inside the venv
& $VenvPython -m pip install --upgrade pip setuptools wheel
# Install requirements inside the venv
& $VenvPython -m pip install -r requirements.txt
# Install the local project in editable mode inside the venv
& $VenvPython -m pip install -e .

# --- CONCLUSION ---

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "-----------------------------------------------------"
Write-Host "IMPORTANT: Before running the simulation, you must activate the environment:" -ForegroundColor Yellow
Write-Host "    $VenvDir\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "-----------------------------------------------------"