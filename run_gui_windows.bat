@echo off
python -m h2_plant.gui.main
set "exit_code=%ERRORLEVEL%"

if not "%exit_code%"=="0" (
    >&2 echo ERROR: python -m h2_plant.gui.main failed (exit %exit_code%)
)

exit /b %exit_code%
