@echo off
setlocal EnableDelayedExpansion
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" amd64 >NUL
if errorlevel 1 (
  echo ng-only-build: vcvars64 FAILED
  exit /b 1
)
echo ng-only-build: vcvars64 loaded.

REM Anchor in the project dir — cmd.exe inherits MSYS bash's cwd
cd /d "C:\code\trafgals\ParParPar"
echo ng-only-build: cwd = %CD%

echo ng-only-build: node-gyp build --release ...
node-gyp build --release 2>&1
echo ng-only-build: exit errorlevel=%errorlevel%
if exist build\Release (
  echo ng-only-build: build/Release/*.node listing:
  dir /b build\Release\*.node 2>&1
) else (
  echo ng-only-build: NO build\Release directory
)
endlocal & exit /b 0
