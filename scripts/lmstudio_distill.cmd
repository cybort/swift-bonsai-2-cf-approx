@echo off
setlocal
REM Windows CMD wrapper for lmstudio_distill.ps1
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0lmstudio_distill.ps1" %*
exit /b %ERRORLEVEL%
