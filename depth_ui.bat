@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title Depth Tools

set "LOG=%~dp0depth_ui.log"
call :log "========================================"
call :log "Depth Tools: %date% %time%"
call :log "Рабочая папка: %CD%"

echo.
echo === Depth Tools ===
echo Лог установки: %LOG%
echo.

rem ---------- найти Python 3.11/3.12 ----------
call :find_python
if %errorlevel% neq 0 goto :install_python
call :log "Найден Python: %PYCMD%"
goto :main

rem ---------- установка Python 3.12.10 ----------
:install_python
call :log "Подходящий Python не найден. Ставлю Python 3.12.10..."
echo Python 3.11/3.12 не найден. Устанавливаю Python 3.12.10 (это займёт пару минут)...
set "PYVER=3.12.10"
set "PYURL=https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-amd64.exe"
set "PYINST=%TEMP%\python-%PYVER%-amd64.exe"
if not exist "%PYINST%" (
    call :log "Скачиваю %PYURL%"
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('%PYURL%','%PYINST%')"
    if not exist "%PYINST%" goto :err_py
)
call :log "Устанавливаю Python тихо..."
"%PYINST%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1
if not exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" goto :err_py
set "PYCMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
call :log "Python 3.12 установлен: %PYCMD%"
goto :main

rem ---------- основной процесс ----------
:main
echo [1/5] Виртуальное окружение...
if exist ".venv\Scripts\python.exe" goto :venv_ok
call :log "Создаю виртуальное окружение (%PYCMD% -m venv .venv)"
"%PYCMD%" -m venv .venv
if %errorlevel% neq 0 goto :err
:venv_ok
call :log "venv готово"

echo [2/5] Установка зависимостей (только готовые колёса)...
".venv\Scripts\python.exe" -m pip install --upgrade pip >>"%LOG%" 2>&1
call :log "pip обновлён"
echo       Устанавливаю пакеты, подождите...
".venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt >>"%LOG%" 2>&1
if %errorlevel% neq 0 goto :err_pip
call :log "Зависимости установлены"

echo [3/5] Проверка модели...
set "MODELDIR=%USERPROFILE%\Z-depth"
if exist "%MODELDIR%\depth_anything_v2_small.onnx" goto :model_ok
call :log "Модель не найдена, скачиваю ~99 МБ в %MODELDIR%"
echo       Модель не найдена, скачиваю ~99 МБ...
".venv\Scripts\python.exe" download_model.py >>"%LOG%" 2>&1
if %errorlevel% neq 0 goto :err_model
:model_ok
call :log "Модель на месте"

echo [4/5] Ярлык на рабочем столе...
powershell -NoProfile -Command "$ws=New-Object -ComObject WScript.Shell; $s=$ws.CreateShortcut([Environment]::GetFolderPath('Desktop')+'\Карта глубины.lnk'); $s.TargetPath='%~dp0depth_ui.bat'; $s.WorkingDirectory='%~dp0'; $s.Save()" >>"%LOG%" 2>&1
call :log "Ярлык создан"

echo [5/5] Запуск приложения...
echo.
call :log "Запуск depth_ui.py"
".venv\Scripts\python.exe" depth_ui.py
set "RC=%errorlevel%"
call :log "Приложение закрыто, код возврата: %RC%"
exit /b %RC%

rem ---------- поиск Python 3.11/3.12 ----------
:find_python
set "PYCMD="
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYCMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    exit /b 0
)
call :check_version py -3
if %errorlevel%==0 (
    set "PYCMD=py -3"
    exit /b 0
)
call :check_version python
if %errorlevel%==0 (
    set "PYCMD=python"
    exit /b 0
)
call :log "Python 3.11/3.12 не найден"
exit /b 1

rem ---------- проверка версии ----------
:check_version
%* -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)" >nul 2>nul
exit /b %errorlevel%

rem ---------- лог ----------
:log
echo %~1>>"%LOG%"
exit /b 0

:err
call :log "ОШИБКА: не удалось создать виртуальное окружение"
echo.
echo Ошибка создания окружения. Подробности в %LOG%
pause
exit /b 1

:err_pip
call :log "ОШИБКА: не удалось установить зависимости (см. выше в логе)"
echo.
echo Не удалось установить зависимости. Подробности выше в %LOG%
echo Причина: для вашего Python нет готовых колёс (чаще всего Python новее 3.12).
echo Запустите этот файл ещё раз - он сам поставит Python 3.12.
pause
exit /b 1

:err_py
call :log "ОШИБКА: не удалось скачать/установить Python 3.12"
echo.
echo Не удалось установить Python 3.12. Скачайте вручную:
echo   https://www.python.org/downloads/  (3.12.x, галочка "Add Python to PATH")
echo и запустите этот файл снова.
pause
exit /b 1

:err_model
call :log "ОШИБКА: не удалось скачать модель"
echo.
echo Не удалось скачать модель. Подробности в %LOG%
echo Скачайте вручную и положите как %MODELDIR%\depth_anything_v2_small.onnx:
echo   https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits_dynamic.onnx
pause
exit /b 1
