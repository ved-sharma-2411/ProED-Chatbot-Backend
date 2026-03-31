@echo off
setlocal

cd /d "%~dp0"
title Full eCFR Pipeline Runner
set "PYTHONUNBUFFERED=1"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo ============================================================
echo Running full eCFR pipeline with live logs...
echo Python: %PY%
echo ============================================================
echo.

%PY% run_full_pipeline.py --parts 600 674 675 676 668 682 685 686 690 --embedding-model sentence-transformers/all-MiniLM-L6-v2 %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
  echo Pipeline completed successfully.
) else (
  echo Pipeline failed with exit code: %EXIT_CODE%
)

echo.
echo Press any key to close this window...
pause >nul
exit /b %EXIT_CODE%
