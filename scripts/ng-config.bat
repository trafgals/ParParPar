@echo off
setlocal EnableDelayedExpansion
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" amd64 >NUL
if errorlevel 1 (
  echo ng-config: vcvars64 FAILED
  exit /b 1
)
echo ng-config: vcvars64 loaded.
echo ng-config: cwd = %CD%

REM Configure only.
echo ng-config: running node-gyp configure --release ...
node-gyp configure --release 2>&1
if errorlevel 1 (
  echo ng-config: CONFIGURE FAILED errorlevel=%errorlevel%
  exit /b 1
)
echo ng-config: CONFIGURE OK.
echo ng-config: Makefiles generated. List of build/:
dir /b build 2>&1 | findstr /R ".mk$ .gypi .Makefile node_gyp" | head -20
endlocal & exit /b 0
