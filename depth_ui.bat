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
    echo Модель не найдена, скачиваю ~99 МБ...
    .venv\Scripts\python download_model.py
    if errorlevel 1 goto :err_model
)

rem ---------- запуск ----------
.venv\Scripts\python depth_ui.py
exit /b 0

:err
echo.
echo Ошибка установки. Проверьте, что установлен Python 3 с 'py' в PATH.
pause
exit /b 1

:err_model
echo.
echo Не удалось скачать модель. Скачайте вручную и положите в %MODELDIR%:
echo   https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits_dynamic.onnx
echo (переименуйте в depth_anything_v2_small.onnx)
pause
exit /b 1
