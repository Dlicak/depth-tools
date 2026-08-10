import os
import urllib.request

import common

# Динамический ONNX-экспорт Depth Anything V2 Small (вход любого размера, кратного 14).
URL = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits_dynamic.onnx"


def download(url, dst):
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.exists(dst):
        print(f"Модель уже есть: {dst}")
        return
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "depth-tools"})
    print(f"Скачивание модели (~99 МБ)...")
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
    download(URL, os.path.join(common.model_dir(), "depth_anything_v2_small.onnx"))


if __name__ == "__main__":
    main()
