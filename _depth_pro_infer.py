"""Depth Pro (apple/DepthPro-hf) в отдельном процессе.

PyTorch и ~1 ГБ весов не должны оставаться в памяти окна после обработки,
поэтому модель живёт только здесь. На вход — PNG, на выход — .npy с
нормализованной обратной глубиной (1.0 = близко, как у остальных моделей)
и .json с углом обзора: Depth Pro сам оценивает фокусное расстояние,
и для PLY/GLB это точнее зашитых 60°.
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import DepthProForDepthEstimation
from transformers.utils import logging as hf_logging

# без полосы «Loading weights» в журнале запуска: она забивает его на сотни строк
hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()


def main():
    model_dir, nw, nh, side, img_path, out_npy, out_meta = sys.argv[1:8]
    nw, nh, side = int(nw), int(nh), int(side)
    # на 4-ядерном ноутбуке больше потоков только мешает
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))

    dtype = torch.bfloat16 if os.environ.get("DEPTH_PRO_DTYPE") == "bf16" else torch.float32

    img = Image.open(img_path).convert("RGB").resize((side, side), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))[None].to(dtype)

    model = DepthProForDepthEstimation.from_pretrained(
        model_dir, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    with torch.inference_mode():
        out = model(pixel_values=x)
        inv = out.predicted_depth[0]
        fov_raw = out.field_of_view
        fov_raw = None if fov_raw is None else float(torch.as_tensor(fov_raw).reshape(-1)[0])
        # HF-порт отдаёт угол обзора в 10 раз меньше: без поправки метрическая
        # глубина тоже выходит в 10 раз больше (голова собаки — 23 м вместо ~2 м).
        # Поправка проверена по масштабу глубины на двух разных фото.
        fov_x = 60.0 if fov_raw is None else fov_raw * 10.0
        if not (5.0 <= fov_x <= 150.0):
            fov_x = 60.0
        # перевод в обратную метрическую глубину — как в apple/ml-depth-pro:
        # f_px = 0.5 * W / tan(fov/2), inverse_depth = canonical * (W / f_px)
        f_px = 0.5 * side / float(np.tan(np.deg2rad(fov_x) / 2.0))
        inv = inv * (side / f_px)
        inv = torch.nn.functional.interpolate(
            inv[None, None], size=(nh, nw), mode="bicubic", align_corners=False)[0, 0]
        d = inv.double().cpu().numpy()
        # метрическая глубина (м) — для журнала: по ней видно, вменяем ли масштаб
        with np.errstate(divide="ignore", invalid="ignore"):
            met = np.where(d > 1e-9, 1.0 / d, np.nan)

    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(d, 1.0), np.percentile(d, 99.0)
    d = np.clip((d - lo) / (hi - lo + 1e-9), 0.0, 1.0).astype(np.float32)
    np.save(out_npy, d)

    # fov_x — по ширине кадра; PLY строит проекцию по вертикали, переводим
    fov_y = float(np.degrees(2.0 * np.arctan(
        np.tan(np.deg2rad(fov_x) / 2.0) * float(nh) / float(nw))))
    with open(out_meta, "w") as f:
        json.dump({"fov_x": fov_x, "fov_y": fov_y, "side": side, "fov_raw": fov_raw,
                   "m_min": float(np.nanmin(met)), "m_med": float(np.nanmedian(met)),
                   "m_max": float(np.nanmax(met))}, f)


if __name__ == "__main__":
    main()
