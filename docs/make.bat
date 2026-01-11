@echo off
set SPHINXBUILD=sphinx-build
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR%
if errorlevel 1 exit /b 1
exit /b 0

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR%
