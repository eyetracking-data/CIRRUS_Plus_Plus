@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 goto :FAIL
title CIRRUS++
set "PYTHON="
set "PYTHON_ARGS="
set "INSTALL_ATTEMPTED=0"
set "CHECK_ONLY=0"
for %%A in (%*) do if /I "%%~A"=="--check" set "CHECK_ONLY=1"
set "BOOTSTRAP_LOG=%TEMP%\cirruspp_python_install.log"
>"%BOOTSTRAP_LOG%" echo CIRRUS++ Windows startup %DATE% %TIME%

:SEARCH_PYTHON
set "CANDIDATE_ARGS="
set "CANDIDATE=%CD%\.venv\Scripts\python.exe"
call :TRY_PYTHON
if defined PYTHON goto :LAUNCH

if not exist "python_path.txt" goto :KNOWN_LOCATIONS
set "CANDIDATE="
set /p "CANDIDATE="<"python_path.txt"
call :TRY_PYTHON
if defined PYTHON goto :LAUNCH

:KNOWN_LOCATIONS
for %%V in (3.13 3.12 3.14 3.11) do (
    set "CANDIDATE=%LOCALAPPDATA%\Python\pythoncore-%%V-64\python.exe"
    call :TRY_PYTHON
    if defined PYTHON goto :LAUNCH
)
for %%V in (313 312 314 311) do (
    set "CANDIDATE=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
    call :TRY_PYTHON
    if defined PYTHON goto :LAUNCH
)
for %%P in ("%USERPROFILE%\anaconda3\python.exe" "%USERPROFILE%\miniconda3\python.exe" "%LOCALAPPDATA%\anaconda3\python.exe" "%LOCALAPPDATA%\miniconda3\python.exe") do (
    set "CANDIDATE=%%~P"
    call :TRY_PYTHON
    if defined PYTHON goto :LAUNCH
)
where py >nul 2>nul
if errorlevel 1 goto :PATH_SEARCH
set "CANDIDATE=py"
for %%V in (3.13 3.12 3.14 3.11) do (
    set "CANDIDATE_ARGS=-%%V"
    call :TRY_PYTHON
    if defined PYTHON goto :LAUNCH
)

:PATH_SEARCH
set "CANDIDATE_ARGS="
for /f "delims=" %%P in ('where python python3 2^>nul') do (
    set "CANDIDATE=%%P"
    call :TRY_PYTHON
    if defined PYTHON goto :LAUNCH
)
if "%INSTALL_ATTEMPTED%"=="1" goto :NO_PYTHON
rem --check never installs software, including a system Python.
if "%CHECK_ONLY%"=="1" goto :NO_PYTHON
where winget >nul 2>nul
if errorlevel 1 goto :NO_PYTHON
set "INSTALL_ATTEMPTED=1"
echo Installing Python 3.13 for the current Windows user...
winget install -e --id Python.Python.3.13 --scope user --silent --accept-package-agreements --accept-source-agreements >>"%BOOTSTRAP_LOG%" 2>&1
if errorlevel 1 goto :NO_PYTHON
goto :SEARCH_PYTHON

:TRY_PYTHON
if not defined CANDIDATE exit /b 1
rem Do not activate the Microsoft Store aliases while looking for an interpreter.
if /I "%CANDIDATE%"=="%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" exit /b 1
if /I "%CANDIDATE%"=="%LOCALAPPDATA%\Microsoft\WindowsApps\python3.exe" exit /b 1
"%CANDIDATE%" %CANDIDATE_ARGS% -c "import sys; raise SystemExit(sys.version_info < (3, 11))" >nul 2>nul
if errorlevel 1 exit /b 1
set "PYTHON=%CANDIDATE%"
set "PYTHON_ARGS=%CANDIDATE_ARGS%"
exit /b 0

:LAUNCH
"%PYTHON%" %PYTHON_ARGS% "%~dp0launch_cirruspp.py" %*
if errorlevel 1 goto :FAIL
exit /b 0

:NO_PYTHON
echo Python 3.11 or newer could not be found or installed.
echo Install Python from https://www.python.org/downloads/windows/ and retry.
echo Alternatively, put the full path to python.exe in python_path.txt.
echo Python installation log: "%BOOTSTRAP_LOG%"

:FAIL
echo.
echo CIRRUS++ could not start. Check the messages above and the displayed startup-log path.
echo If port 8501 is busy, close the other server or use --port 8502.
pause
exit /b 1
