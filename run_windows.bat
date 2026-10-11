@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title CIRRUS++
set "LOG=%CD%\cirruspp_startup.log"
echo CIRRUS++ start %DATE% %TIME% > "%LOG%"

echo ============================================================
echo CIRRUS++ 1.5.1 - One-click launcher
echo ============================================================
echo Folder: %CD%
echo.

rem ------------------------------------------------------------
rem If this app has already been started once, no system Python
rem lookup is needed: use the local virtual environment directly.
rem ------------------------------------------------------------
if exist ".venv\Scripts\python.exe" goto :VENV_READY

set "PY_EXE="
set "PY_ARGS="
set "INSTALL_ATTEMPTED=0"

:SEARCH_PYTHON
rem 1) Optional explicit path.
if exist "python_path.txt" (
    set /p PY_EXE=<"python_path.txt"
    if exist "!PY_EXE!" goto :PY_FOUND
    set "PY_EXE="
)

rem 2) PythonCore locations (same family used by the previous CIRRUS++ launcher).
for %%V in (3.13 3.12 3.14 3.11) do (
    set "CAND=%LOCALAPPDATA%\Python\pythoncore-%%V-64\python.exe"
    if exist "!CAND!" (
        set "PY_EXE=!CAND!"
        goto :PY_FOUND
    )
)

rem 3) Standard python.org per-user installations.
for %%V in (313 312 314 311) do (
    set "CAND=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
    if exist "!CAND!" (
        set "PY_EXE=!CAND!"
        goto :PY_FOUND
    )
)

rem 4) Common Conda installations.
for %%C in ("%USERPROFILE%\anaconda3\python.exe" "%USERPROFILE%\miniconda3\python.exe" "%LOCALAPPDATA%\anaconda3\python.exe" "%LOCALAPPDATA%\miniconda3\python.exe") do (
    if exist "%%~C" (
        set "PY_EXE=%%~C"
        goto :PY_FOUND
    )
)

rem 5) If Jupyter works, its Scripts directory often sits next to python.exe.
for /f "delims=" %%J in ('where jupyter 2^>nul') do (
    for %%Q in ("%%~dpJ..\python.exe") do (
        if exist "%%~fQ" (
            set "PY_EXE=%%~fQ"
            goto :PY_FOUND
        )
    )
)

rem 6) Windows Python launcher.
where py >nul 2>nul
if not errorlevel 1 (
    for %%V in (3.13 3.12 3.14 3.11) do (
        py -%%V -c "import sys; print(sys.executable)" >nul 2>nul
        if not errorlevel 1 (
            set "PY_EXE=py"
            set "PY_ARGS=-%%V"
            goto :PY_FOUND
        )
    )
)

rem 7) PATH commands.
where python >nul 2>nul
if not errorlevel 1 (
    set "PY_EXE=python"
    goto :PY_FOUND
)
where python3 >nul 2>nul
if not errorlevel 1 (
    set "PY_EXE=python3"
    goto :PY_FOUND
)

rem 8) Last-resort one-click install through Windows Package Manager.
if "%INSTALL_ATTEMPTED%"=="0" (
    where winget >nul 2>nul
    if not errorlevel 1 (
        set "INSTALL_ATTEMPTED=1"
        echo No usable Python was found. Installing Python 3.13 for the current user...
        winget install -e --id Python.Python.3.13 --scope user --silent --accept-package-agreements --accept-source-agreements >>"%LOG%" 2>&1
        if not errorlevel 1 (
            set "PY_EXE="
            set "PY_ARGS="
            goto :SEARCH_PYTHON
        )
    )
)

echo.
echo [ERROR] No usable Python installation could be found automatically.
echo.
echo If Jupyter is installed in an unusual environment, create a file named
 echo python_path.txt in this folder and put the full path to python.exe inside it.
echo Startup log: %LOG%
pause
exit /b 1

:PY_FOUND
echo [1/4] Python found: %PY_EXE% %PY_ARGS%
if /I "%PY_EXE%"=="py" (
    py %PY_ARGS% --version >>"%LOG%" 2>&1
    if errorlevel 1 goto :FAIL
    echo [2/4] Creating private CIRRUS++ environment...
    py %PY_ARGS% -m venv .venv >>"%LOG%" 2>&1
) else (
    "%PY_EXE%" %PY_ARGS% --version >>"%LOG%" 2>&1
    if errorlevel 1 goto :FAIL
    echo [2/4] Creating private CIRRUS++ environment...
    "%PY_EXE%" %PY_ARGS% -m venv .venv >>"%LOG%" 2>&1
)
if errorlevel 1 goto :FAIL

:VENV_READY
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" goto :FAIL

echo [1/4] Private CIRRUS++ Python ready.
echo [2/4] Virtual environment ready.

if not exist ".venv\.cirruspp_dependencies_1_5" (
    echo [3/4] Installing CIRRUS++ dependencies - first start can take a few minutes...
    "%VENV_PY%" -m pip install --upgrade pip >>"%LOG%" 2>&1
    if errorlevel 1 goto :FAIL
    "%VENV_PY%" -m pip install -r requirements.txt >>"%LOG%" 2>&1
    if errorlevel 1 goto :FAIL
    echo ready> ".venv\.cirruspp_dependencies_1_5"
) else (
    echo [3/4] Dependencies already installed.
)

echo.
echo [4/4] Starting CIRRUS++ at http://localhost:8501
echo Keep this window open while you use CIRRUS++.
echo Close it when you are finished.
echo.
"%VENV_PY%" -m streamlit run app.py --server.port 8501
if errorlevel 1 goto :FAIL
goto :END

:FAIL
echo.
echo ============================================================
echo CIRRUS++ could not start.
echo Startup log: %LOG%
echo ============================================================
pause
exit /b 1

:END
endlocal
