@echo off
setlocal ENABLEDELAYEDEXPANSION
echo ======================================================
echo [BUILD] Compiling hilbert_native.pyd via Conda Python
echo ======================================================

REM -----------------------------------------------------------------
REM 1. Detect active conda environment
REM -----------------------------------------------------------------
if not defined CONDA_PREFIX (
    echo [ERROR] No active conda environment detected.
    echo Run:  conda activate hilbert
    exit /b 1
)

set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python interpreter not found in: %CONDA_PREFIX%
    exit /b 1
)

echo [INFO] Using Conda Python: %PYTHON_EXE%
"%PYTHON_EXE%" -V

REM -----------------------------------------------------------------
REM 2. Query sysconfig safely (no heredoc)
REM -----------------------------------------------------------------
"%PYTHON_EXE%" -c "import sysconfig; print(sysconfig.get_path('include'))" > py_inc.txt
"%PYTHON_EXE%" -c "import sysconfig,sys,os; print(sysconfig.get_config_var('LIBDIR') or os.path.join(sys.base_prefix,'libs'))" > py_libdir.txt
"%PYTHON_EXE%" -c "import sys; print(f'python{sys.version_info[0]}{sys.version_info[1]}.lib')" > py_libfile.txt

set /p INC=<py_inc.txt
set /p LIBDIR=<py_libdir.txt
set /p LIBFILE=<py_libfile.txt

del py_inc.txt py_libdir.txt py_libfile.txt >nul 2>nul

echo [Python] Include: %INC%
echo [Python] LibDir:  %LIBDIR%
echo [Python] LibFile: %LIBFILE%

REM -----------------------------------------------------------------
REM 3. Load MSVC Build Tools via vswhere
REM -----------------------------------------------------------------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if not exist "%VSWHERE%" (
    echo [ERROR] vswhere.exe not found. Install VS Build Tools 2019/2022.
    exit /b 1
)

"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%TEMP%\vs_path.txt"
set /p VS_PATH=<"%TEMP%\vs_path.txt"
del "%TEMP%\vs_path.txt" >nul 2>nul

if not defined VS_PATH (
    echo [ERROR] Visual Studio Build Tools not detected.
    exit /b 1
)

call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat" >nul
echo [INFO] Using MSVC toolchain at: %VS_PATH%

REM -----------------------------------------------------------------
REM 4. Clean previous artifacts
REM -----------------------------------------------------------------


REM -----------------------------------------------------------------
REM 5. Compile
REM -----------------------------------------------------------------
echo [COMPILING] Building hilbert_native.pyd...

cl /nologo /LD ^
    /I"%INC%" ^
    hilbert_native.c ^
    hilbert_math.c ^
    hilbert_simulation.c ^
    hilbert_pybind.c ^
    /link /LIBPATH:"%LIBDIR%" "%LIBFILE%" ^
    /OUT:hilbert_native.pyd

if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo ======================================================
echo [SUCCESS] hilbert_native.pyd successfully built!
echo Output: %CD%\hilbert_native.pyd
echo ======================================================

endlocal
