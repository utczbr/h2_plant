
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
This happens when the monthly LFS bandwidth is exhausted. You can skip downloading the large `.pkl` cache files. The system will automatically regenerate them locally on first run (takes ~1-2 min).

**Solution:**
1. Skip LFS download during clone:
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/utczbr/h2_plant
   ```
2. Proceed with installation as normal.
3. On first run, ignore "Physics parameters file not found" warnings; the `LUTManager` will regenerate the tables.


if you are having a distutil problem, run "python -m pip install --upgrade pip setuptools wheel"
