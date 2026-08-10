import json
import os
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import onnxruntime as ort

import common

CONFIG = os.path.join(common.data_dir(), "settings.json")
SRC = common.default_src()
OUT = common.data_dir()

DEFAULTS = {
    "model": "large",
    "input_size": 1024,
    "unsharp_radius": 3,
    "unsharp_percent": 120,
    "unsharp_thresh": 2,
    "depth_contrast": 1.0,
    "overlay_alpha": 0.45,
    "blur_strength": 6,
    "out_size": "300x300",
}

cfg = DEFAULTS.copy()
if os.path.exists(CONFIG):
    try:
        cfg.update(json.load(open(CONFIG)))
    except Exception:
        pass

print("=== Настройки глубины (Enter = оставить как есть) ===")

def ask(label, key, fmt=None):
    val = cfg.get(key, DEFAULTS[key])
    cur = fmt(val) if fmt else str(val)
    try:
        txt = input(f"{label} [{cur}]: ").strip()
    except EOFError:
        txt = ""
    if txt:
        val = fmt(txt)
    cfg[key] = val

ask("Модель (large/small)", "model", str)
ask("Разрешение входа (518/1024/2048)", "input_size", int)
ask("Unsharp радиус (2-6)", "unsharp_radius", float)
ask("Unsharp сила (50-300)", "unsharp_percent", float)
ask("Unsharp порог (0-10)", "unsharp_thresh", float)
ask("Контраст карты (1.0-2.0)", "depth_contrast", float)
ask("Прозрачность оверлея (0-1)", "overlay_alpha", float)
ask("Сила размытия DoF (2-12)", "blur_strength", float)
ask("Размер выхода (300x300 / 1024x768)", "out_size", str)

json.dump(cfg, open(CONFIG, "w"), indent=2)

w, h = [int(x) for x in cfg["out_size"].split("x")]

MODEL_PATH = common.find_model(f"depth_anything_v2_{cfg['model']}.onnx")
if not os.path.exists(MODEL_PATH):
    print("Нет модели:", MODEL_PATH)
    raise SystemExit(1)

img = Image.open(SRC).convert("RGB")
orig = img

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
inp = img.resize((cfg["input_size"], cfg["input_size"]), Image.LANCZOS)
arr = np.asarray(inp, dtype=np.float32) / 255.0
arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
arr = arr.transpose(2, 0, 1)[None]
name = sess.get_inputs()[0].name
pred = sess.run(None, {name: arr})[0]
d = pred[0]
d = d - d.min()
d = d / (d.max() + 1e-8)
d = (d * 255).astype(np.uint8)
depth_small = Image.fromarray(d).resize(orig.size, Image.BILINEAR)
depth_small = ImageOps.autocontrast(depth_small)

depth_small = depth_small.filter(ImageFilter.UnsharpMask(
    radius=max(1, int(cfg["unsharp_radius"])), percent=int(cfg["unsharp_percent"]), threshold=int(cfg["unsharp_thresh"])))
depth_small = ImageOps.autocontrast(depth_small)
if cfg["depth_contrast"] != 1.0:
    depth_small = ImageEnhance.Contrast(depth_small).enhance(cfg["depth_contrast"])

depth_small.save(f"{OUT}/photo_depth.png")

overlay = Image.blend(orig, depth_small.convert("RGB"), cfg["overlay_alpha"])
overlay.save(f"{OUT}/photo_depth_overlay.png")

blurred = orig.filter(ImageFilter.GaussianBlur(cfg["blur_strength"]))
depth_mask = depth_small.point(lambda p: 255 - p)
Image.composite(blurred, orig, depth_mask).save(f"{OUT}/photo_dof.png")

a = orig.convert("RGB").resize((w, h), Image.LANCZOS)
b = depth_small.convert("RGB").resize((w, h), Image.LANCZOS)
side = Image.new("RGB", (w * 2 + 10, h), (40, 40, 40))
side.paste(a, (0, 0))
side.paste(b, (w + 10, 0))
side.save(f"{OUT}/photo_color_plus_depth.png")

print("Готово. Настройки сохранены в", CONFIG)
