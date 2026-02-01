
### Windows Cloning Issues
#### 1. "Filename too long" Error
This occurs because some legacy files exceed the Windows 260-character path limit.
**Solution:** Enable long paths in Windows or clone deeply nested files manually.
To enable long paths (Admin PowerShell):
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
Or configure git:
```bash
git config --system core.longpaths true
```

#### 2. "LFS budget exceeded" (Smudge Error)
This happens when the monthly LFS bandwidth is exhausted, or if you simply wish to avoid LFS entirely.

**Solution:** Use the provided bypass script to download the required cache files from Google Drive.

1. Clone the repository skipping the LFS download:
   ```bash
   git clone --config filter.lfs.smudge= --config filter.lfs.process= https://github.com/utczbr/h2_plant
   ```
2. Run the bypass script (PowerShell):
   ```powershell
   .\setup.ps1
   ```
   This will:
   - Download the necessary `.pkl` files into `.h2_plant\lut_cache`.
   - Create a Python virtual environment (`.venv`).
   - Install all dependencies.

3. Proceed with setting up your environment (venvs, pip install, etc.) as normal.


if you are having a distutil problem, run "python -m pip install --upgrade pip setuptools wheel"
