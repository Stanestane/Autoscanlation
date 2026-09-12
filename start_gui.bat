@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0comic_translator_env\Scripts\pythonw.exe" (
    echo The project Python environment was not found.
    echo Follow the setup instructions in README.md, then try again.
    pause
    exit /b 1
)

start "" "%~dp0comic_translator_env\Scripts\pythonw.exe" "%~dp0gui.py"
