import os
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import onnxruntime as ort

import common

# ===================== НАСТРОЙКИ (крути тут) =====================

# Исходное фото (по умолчанию — свежее изображение в ~/Downloads) и куда сохранять.
# Можно задать DEPTH_TOOLS_SRC и DEPTH_TOOLS_HOME.
SRC = common.default_src() or "путь/к/фото.png"
OUT = common.data_dir()

# Модель: large = максимальное качество/детали, small = быстрее и грубее.
# Положите ONNX в ~/Z-depth (или DEPTH_TOOLS_MODELS).
MODEL = common.find_model("depth_anything_v2_large.onnx")

# Разрешение входа модели (разрешение карты глубины ~ этому значению).
# 518 = быстро и сносно, 1024 = детальнее, 2048 = максимум деталей, но медленно.
INPUT_SIZE = 1024

# Ресайз-фильтр для входного изображения (LANCZOS = самый чёткий)
INPUT_RESAMPLE = Image.LANCZOS

# Усиление деталей карты глубины (unsharp mask):
#   UNSUARP_RADIUS  - радиус (2-6), больше = более крупные детали
#   UNSHARP_PERCENT - сила (50-300), больше = резче контраст деталей
#   UNSHARP_THRESH  - порог (0-10), выше = усиливаются только сильные перепады
UNSUARP_RADIUS = 3
UNSHARP_PERCENT = 120
UNSHARP_THRESH = 2

# Дополнительное усиление контраста карты глубины (1.0 = без изменений, 1.3-2.0 = сильнее)
DEPTH_CONTRAST = 1.0

# --- Настройки совмещений ---

# Оверлей: насколько сильно depth-градиент накладывается на фото (0.0-1.0)
OVERLAY_ALPHA = 0.45

# Depth-of-field: сила размытия дальних объектов (пиксели)
BLUR_STRENGTH = 6

# Размер выходных изображений (width, height). 0 = как у исходного фото
OUT_SIZE = (300, 300)

# ================================================================


def load_model(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def predict_depth(session, img, input_size):
    inp = img.resize((input_size, input_size), INPUT_RESAMPLE)
    arr = np.asarray(inp, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    arr = arr.transpose(2, 0, 1)[None]
    name = session.get_inputs()[0].name
    pred = session.run(None, {name: arr})[0]
    d = pred[0]
    d = d - d.min()
    d = d / (d.max() + 1e-8)
    d = (d * 255).astype(np.uint8)
    return Image.fromarray(d).resize(img.size, Image.BILINEAR)


img = Image.open(SRC).convert("RGB")
orig = img

sess = load_model(MODEL)
depth_small = predict_depth(sess, orig, INPUT_SIZE)
depth_small = ImageOps.autocontrast(depth_small)

depth_small = depth_small.filter(ImageFilter.UnsharpMask(
    radius=UNSUARP_RADIUS, percent=UNSHARP_PERCENT, threshold=UNSHARP_THRESH))
depth_small = ImageOps.autocontrast(depth_small)
if DEPTH_CONTRAST != 1.0:
    depth_small = ImageEnhance.Contrast(depth_small).enhance(DEPTH_CONTRAST)

depth_small.save(f"{OUT}/photo_depth.png")

depth_color = depth_small.convert("RGB")

overlay = Image.blend(orig, depth_color, OVERLAY_ALPHA)
overlay.save(f"{OUT}/photo_depth_overlay.png")

blurred = depth_small.convert("RGB").filter(ImageFilter.GaussianBlur(BLUR_STRENGTH))
depth_mask = depth_small.point(lambda p: 255 - p)
dof = Image.composite(blurred, depth_small.convert("RGB"), depth_mask)
dof.save(f"{OUT}/photo_dof.png")

size = OUT_SIZE if OUT_SIZE and OUT_SIZE[0] else orig.size
a = orig.convert("RGB").resize(size, Image.LANCZOS)
b = depth_small.convert("RGB").resize(size, Image.LANCZOS)
side = Image.new("RGB", (size[0] * 2 + 10, size[1]), (40, 40, 40))
side.paste(a, (0, 0))
side.paste(b, (size[0] + 10, 0))
side.save(f"{OUT}/photo_color_plus_depth.png")

print("done. input:", INPUT_SIZE, "model:", MODEL.split("/")[-1])
