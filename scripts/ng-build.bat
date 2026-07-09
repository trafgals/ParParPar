@echo off
setlocal EnableDelayedExpansion
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" amd64 >NUL
if errorlevel 1 (
  echo ng-build: vcvars64 FAILED
  exit /b 1
)
echo ng-build: vcvars64 loaded.
cd /d "C:\code\trafgals\ParParPar"
echo ng-build: cwd = %CD%
echo ng-build: starting at %DATE% %TIME%

REM Clean prior build artifacts.
if exist build rmdir /s /q build
echo ng-build: prior build/ removed.

REM Configure + build as a SINGLE node-gyp invocation so the SLN ordering is right
REM (separate configure + build can leave the .sln referencing a stale config.gypi).
echo ng-build: configure + build (combined) ...
node-gyp rebuild --release > "C:\tmp\ng-build-%RANDOM%.log" 2>&1
set RC=%errorlevel%
echo ng-build: combined rebuild exit=%RC% at %DATE% %TIME%
echo ng-build: build/Release/* listings:
if exist build\Release (
  dir /b build\Release\*.node 2>&1
) else (
  echo ng-build: NO build\Release directory
)
exit /b %RC%
