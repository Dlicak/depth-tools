@echo off
setlocal
cd /d "%~dp0"
title Depth Tools - установка и запуск

echo.
echo === Depth Tools: автоустановка и запуск ===
echo.

rem ---------- 1. найти Python ----------
call :find_python
if %errorlevel% neq 0 (
    echo Python не найден. Устанавливаю Python 3.12.10...
    call :install_python
    if %errorlevel% neq 0 goto :err_py
    set "PYCMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    if not exist "%PYCMD%" goto :err_py
)

echo [1/5] Виртуальное окружение...
if not exist ".venv\Scripts\python.exe" (
    "%PYCMD%" -m venv .venv
    if %errorlevel% neq 0 goto :err
)

echo [2/5] Установка зависимостей...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if %errorlevel% neq 0 goto :err

echo [3/5] Проверка модели...
set "MODELDIR=%USERPROFILE%\Z-depth"
if not exist "%MODELDIR%\depth_anything_v2_small.onnx" (
    echo       Модель не найдена, скачиваю ~99 МБ...
    ".venv\Scripts\python.exe" download_model.py
    if %errorlevel% neq 0 goto :err_model
)

echo [4/5] Ярлык на рабочем столе...
powershell -NoProfile -Command "$ws=New-Object -ComObject WScript.Shell; $s=$ws.CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Карта глубины.lnk'); $s.TargetPath='%~dp0depth_ui.bat'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%~dp0depth_ui.bat,0'; $s.Save()" >nul 2>nul

echo [5/5] Запуск приложения...
echo.
".venv\Scripts\python.exe" depth_ui.py
exit /b 0

rem ---------- поиск Python ----------
:find_python
set "PYCMD="
where py >nul 2>nul
if %errorlevel%==0 (
    set "PYCMD=py -3"
    exit /b 0
)
where python >nul 2>nul
if %errorlevel% neq 0 exit /b 1
python -c "import sys" >nul 2>nul
if %errorlevel% neq 0 exit /b 1
set "PYCMD=python"
exit /b 0

rem ---------- тихая установка Python ----------
:install_python
set "PYVER=3.12.10"
set "PYURL=https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-amd64.exe"
set "PYINST=%TEMP%\python-%PYVER%-amd64.exe"
if not exist "%PYINST%" (
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('%PYURL%','%PYINST%')"
)
if not exist "%PYINST%" exit /b 1
"%PYINST%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1
exit /b 0

:err
echo.
echo Ошибка установки.
pause
exit /b 1

:err_py
echo.
echo Не удалось установить Python. Скачайте его вручную с https://www.python.org/downloads/
echo (галочка "Add Python to PATH"), затем запустите этот файл снова.
pause
exit /b 1

:err_model
echo.
echo Не удалось скачать модель. Положите depth_anything_v2_small.onnx в %MODELDIR%
echo или скачайте вручную:
echo   https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits_dynamic.onnx
pause
exit /b 1
