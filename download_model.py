import os
import sys
import urllib.request

import common

# Динамический ONNX-экспорт Depth Anything V2 (вход любого размера, кратного 14).
MODELS = {
    "small": ("depth_anything_v2_vits_dynamic.onnx", "depth_anything_v2_small.onnx", "~99 МБ"),
    "base": ("depth_anything_v2_vitb_dynamic.onnx", "depth_anything_v2_base.onnx", "~390 МБ"),
    "large": ("depth_anything_v2_vitl_dynamic.onnx", "depth_anything_v2_large.onnx", "~1.3 ГБ"),
}
BASE_URL = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/"


def download(url, dst):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.exists(dst):
        print(f"Модель уже есть: {dst}")
        return
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "depth-tools"})
    print("Скачивание модели...")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                print(f"\r  {done // (1024 * 1024)}/{total // (1024 * 1024)} МБ ({pct}%)", end="", flush=True)
    print()
    os.replace(tmp, dst)
    print(f"Готово: {dst}")


def main():
    key = sys.argv[1].lower() if len(sys.argv) > 1 else "small"
    if key not in MODELS:
        print(f"Неизвестная модель: {key}. Доступны: {', '.join(MODELS)}")
        return
    remote, local, size = MODELS[key]
    download(BASE_URL + remote, os.path.join(common.model_dir(), local))


if __name__ == "__main__":
    main()
