#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=== Depth Tools: автоустановка ==="
echo

# 1. python3 + venv
if ! command -v python3 >/dev/null 2>&1; then
    echo "[1/4] Устанавливаю python3..."
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-venv python3-pip
elif ! python3 -c "import venv" >/dev/null 2>&1; then
    echo "[1/4] Устанавливаю python3-venv..."
    sudo apt-get install -y python3-venv
else
    echo "[1/4] python3 найден."
fi

# 2. виртуальное окружение
echo "[2/4] Виртуальное окружение..."
if [ ! -x ".venv/bin/python" ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip >/dev/null
.venv/bin/pip install --only-binary=:all: -r requirements.txt

# 2b. GPU (NVIDIA): ускорение инференса в 5-10 раз
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "      NVIDIA GPU найден — ставлю onnxruntime-gpu + CUDA-библиотеки..."
    .venv/bin/pip install --only-binary=:all: "onnxruntime-gpu==1.26.0" \
        nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
        || echo "      Не удалось, остаёмся на CPU."
fi

# 3. модель
echo "[3/4] Проверка модели..."
if [ ! -f "$HOME/Z-depth/depth_anything_v2_small.onnx" ]; then
    echo "      Модель не найдена, скачиваю ~99 МБ..."
    .venv/bin/python download_model.py
fi

# 4. запуск
echo "[4/4] Запуск приложения..."
echo
.venv/bin/python depth_ui.py
