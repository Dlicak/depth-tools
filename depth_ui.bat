@echo off
setlocal
cd /d "%~dp0"

rem ---------- виртуальное окружение ----------
if not exist ".venv\Scripts\python.exe" (
    echo Создание виртуального окружения...
    py -3 -m venv .venv
    if errorlevel 1 goto :err
)

rem ---------- зависимости ----------
echo Установка зависимостей...
.venv\Scripts\python -m pip install --upgrade pip >nul
.venv\Scripts\pip install -r requirements.txt
if errorlevel 1 goto :err

rem ---------- модель ----------
set "MODELDIR=%USERPROFILE%\Z-depth"
if not exist "%MODELDIR%\depth_anything_v2_small.onnx" (
    echo.
    echo ВАЖНО: положите файл depth_anything_v2_small.onnx ^(~99 МБ^) в:
    echo   %MODELDIR%
    echo.
    pause
    exit /b 1
)

rem ---------- запуск ----------
.venv\Scripts\python depth_ui.py
exit /b 0

:err
echo.
echo Ошибка установки. Проверьте, что установлен Python 3 с 'py' в PATH.
pause
exit /b 1
