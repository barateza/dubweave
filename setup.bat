@echo off
chcp 65001 >nul
REM ── Dubweave — Setup ────────────────────────────────────────

echo.
echo  +------------------------------------------+
echo  ^|          DUBWEAVE -- SETUP               ^|
echo  ^|  YouTube -^> Portugues Brasileiro (GPU)   ^|
echo  +------------------------------------------+
echo.
echo [1/1] Running Dubweave Pixi Setup...
pixi run setup
echo.
echo  [OK] Setup complete!
echo  --^> Run start.bat to launch the app.
echo.
pause