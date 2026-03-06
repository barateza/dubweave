@echo off
chcp 65001 >nul
REM ── Dubweave — Launch ───────────────────────────────────────

echo.
echo  +------------------------------------------+
echo  ^|          DUBWEAVE -- START               ^|
echo  ^|  YouTube -^> Portugues Brasileiro (GPU)   ^|
echo  +------------------------------------------+
echo.
echo  Opening at http://localhost:7860
echo  Press Ctrl+C to stop.
echo.
pixi run start
pause