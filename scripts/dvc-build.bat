@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" amd64 >NUL
if errorlevel 1 (
  echo dvc-build: vcvars64 FAILED errorlevel=%errorlevel%
  exit /b 1
)
echo dvc-build: vcvars64 loaded.
echo dvc-build: cl --version:
cl 2>&1 | findstr "Version"
echo dvc-build: link:
link 2>&1 | findstr "Version"
echo dvc-build: rc:
rc 2>&1 | findstr "version"
echo dvc-build: msbuild:
where msbuild 2>&1 | head -2
echo dvc-build: python (used by node-gyp):
where python 2>&1 | head -2
echo dvc-build: node:
where node 2>&1 | head -2
echo dvc-build: node-gyp:
where node-gyp 2>&1 | head -2
endlocal
exit /b 0
