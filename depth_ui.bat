@echo off
setlocal
cd /d "%~dp0"
title Depth Tools - установка и запуск

echo.
echo === Depth Tools: автоустановка и запуск ===
echo.

rem ---------- найти Python 3.11/3.12 (для onnxruntime нужны готовые колёса) ----------
call :find_python
if %errorlevel% neq 0 goto :install_python
goto :main

rem ---------- тихая установка Python 3.12.10, если подходящего нет ----------
:install_python
echo Python 3.11/3.12 не найден (он нужен для onnxruntime). Устанавливаю Python 3.12.10...
set "PYVER=3.12.10"
set "PYURL=https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-amd64.exe"
set "PYINST=%TEMP%\python-%PYVER%-amd64.exe"
if not exist "%PYINST%" (
    powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; (New-Object Net.WebClient).DownloadFile('%PYURL%','%PYINST%')"
)
if not exist "%PYINST%" goto :err_py
"%PYINST%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1
if not exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" goto :err_py
set "PYCMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
goto :main

rem ---------- основной процесс установки ----------
:main
echo [1/5] Виртуальное окружение...
if not exist ".venv\Scripts\python.exe" (
    "%PYCMD%" -m venv .venv
    if %errorlevel% neq 0 goto :err
)

echo [2/5] Установка зависимостей (только готовые колёса)...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements.txt
if %errorlevel% neq 0 goto :err_pip

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

rem ---------- поиск Python 3.11/3.12 ----------
:find_python
set "PYCMD="
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYCMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    exit /b 0
)
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)" >nul 2>nul
    if %errorlevel%==0 (
        set "PYCMD=py -3"
        exit /b 0
    )
)
where python >nul 2>nul
if %errorlevel%==0 (
    python -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)" >nul 2>nul
    if %errorlevel%==0 (
        set "PYCMD=python"
        exit /b 0
    )
)
exit /b 1

:err
echo.
echo Ошибка создания окружения.
pause
exit /b 1

:err_pip
echo.
echo Не удалось установить зависимости.
echo Причина: для вашего Python нет готовых колёс (чаще всего Python новее 3.12,
echo а onnxruntime под него ещё не выпущен). Запустите этот файл ещё раз —
echo он сам установит Python 3.12 и всё настроит.
pause
exit /b 1

:err_py
echo.
echo Не удалось установить Python 3.12. Скачайте его вручную с
echo https://www.python.org/downloads/ (версия 3.12.x, галочка "Add Python to PATH")
echo и запустите этот файл снова.
pause
exit /b 1

:err_model
echo.
echo Не удалось скачать модель. Проверьте интернет или скачайте вручную:
echo   https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits_dynamic.onnx
echo и положите как %MODELDIR%\depth_anything_v2_small.onnx
pause
exit /b 1
