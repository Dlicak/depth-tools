import json
import math
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageFilter, ImageTk
import onnxruntime as ort

import common

CONFIG = os.path.join(common.data_dir(), "settings.json")
OUT = common.data_dir()
SRC = common.default_src()

# Вертикальный угол обзора для облака точек и GLB. Раньше его крутили
# ползунком «FOV PLY»; оставили одно разумное значение — как у обычной камеры.
DEFAULT_FOV_Y = 60.0

# Палитра интерфейса
BG = "#15171c"
CARD = "#1e2229"
CARD2 = "#262b34"
FG = "#e9ecf1"
MUTED = "#98a0ad"
ACCENT = "#4c8dff"
BORDER = "#2c323c"

DEFAULTS = {
    "model": "small",
    "depth_gamma": 1.0,
    "depth_lo": 0.0,
    "depth_hi": 100.0,
    "depth_contrast": 1.0,
    "out_size": "300x300",
    "invert": False,
    "render2": False,
    "input_mult": 6.0,
    "depth16": False,
    "depth32": False,
    "depth_exr": False,
    "cmap_exr": False,
    "normal_exr": False,
    "ply_out": False,
    "glb_out": False,
    "gen": "none",
    "gen_amp": 1.0,
    "gen_freq": 2.0,
    "src_div": "1x",
    "guided": 45.0,
    "denoise": 0.0,
}

cfg = DEFAULTS.copy()
_raw = {}
if os.path.exists(CONFIG):
    try:
        _raw = json.load(open(CONFIG))
        cfg.update(_raw)
    except Exception:
        _raw = {}

# Миграция на единый «Множитель входа» (раньше вход = размер фото × множитель,
# а «Эконом ОЗУ» давал 200 px × множитель). Число пересчитываем так, чтобы
# прошлый вход модели остался прежним: mult = in_w / 200.
if cfg.get("_input_scheme") != "200x":
    if not cfg.get("compress"):
        try:
            _long = max(float(cfg.get("in_w") or 0), float(cfg.get("in_h") or 0))
        except Exception:
            _long = 0.0
        if _long > 0:
            cfg["input_mult"] = max(1.0, min(10.0, round(_long / 200.0)))
    cfg["_input_scheme"] = "200x"

# «Карта нормалей» в старых настройках называлась normal_png
if "normal_exr" not in _raw and "normal_png" in _raw:
    cfg["normal_exr"] = bool(_raw["normal_png"])

# Убранные из интерфейса ключи: иначе они тянутся из старого settings.json
for _stale in ("ply_fov", "ply_scale", "compress", "overlay_alpha", "focus_enable",
               "focus_x", "focus_y", "focus_radius", "dof_strength", "normal_png",
               "gen", "gen_amp", "gen_freq", "fx_strength",
               "relief_radius", "relief_strength"):
    cfg.pop(_stale, None)

STATE = {"busy": False}


MODEL_URLS = {
    "small": "depth_anything_v2_vits_dynamic.onnx",
    "base": "depth_anything_v2_vitb_dynamic.onnx",
    "large": "depth_anything_v2_vitl_dynamic.onnx",
    "base_in": "depth_anything_v2_vitb_indoor_dynamic.onnx",
    "base_out": "depth_anything_v2_vitb_outdoor_dynamic.onnx",
    "large_in": "depth_anything_v2_vitl_indoor_dynamic.onnx",
    "large_out": "depth_anything_v2_vitl_outdoor_dynamic.onnx",
}

METRIC_MODELS = {"zoe", "base_in", "base_out", "large_in", "large_out", "large_mix"}


def download_model(key, dst):
    import urllib.request
    url = ("https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/"
           + MODEL_URLS[key])
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "depth-tools"})
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
                print(f"\r{done // (1024 * 1024)}/{total // (1024 * 1024)} МБ ({pct}%)", end="", flush=True)
    os.replace(tmp, dst)


# --- Depth Pro (apple/DepthPro-hf) ---
# Тяжёлая модель на PyTorch: зависимости и ~1,9 ГБ весов ставятся только при
# первом выборе, чтобы обычная установка «в один клик» оставалась лёгкой.
DEPTH_PRO_DIRNAME = "DepthPro"
DEPTH_PRO_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
_DEPTH_PRO_MODULES = None


def depth_pro_dir():
    return os.path.join(common.model_dir(), DEPTH_PRO_DIRNAME)


def depth_pro_weights_ok():
    d = depth_pro_dir()
    return all(os.path.exists(os.path.join(d, f)) for f in DEPTH_PRO_FILES)


def depth_pro_modules_ok():
    global _DEPTH_PRO_MODULES
    if _DEPTH_PRO_MODULES is None:
        import subprocess
        import sys
        r = subprocess.run([sys.executable, "-c", "import torch, transformers, safetensors"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _DEPTH_PRO_MODULES = r.returncode == 0
    return _DEPTH_PRO_MODULES


def download_hf_file(name, dst_dir, on_progress=None):
    import urllib.request
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name)
    url = f"https://huggingface.co/apple/DepthPro-hf/resolve/main/{name}"
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "depth-tools"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if on_progress is not None and total:
                on_progress(done / float(total))
    os.replace(tmp, dst)


def install_depth_pro_deps(on_progress=None):
    """torch (CPU-сборка), transformers и safetensors — в то же окружение,
    где запущено приложение. Журнал пишем в каталог данных: если pip упадёт,
    в ошибке будет путь к файлу."""
    import subprocess
    import sys
    import time
    log_path = os.path.join(OUT, "depth_pro_install.log")
    cmds = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "--no-input",
         "--disable-pip-version-check", "--index-url",
         "https://download.pytorch.org/whl/cpu", "torch"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "--no-input",
         "--disable-pip-version-check", "transformers", "safetensors"],
    ]
    with open(log_path, "w") as log:
        for i, cmd in enumerate(cmds):
            t0 = time.monotonic()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                time.sleep(0.3)
                if on_progress is not None:
                    frac = 1.0 - math.exp(-(time.monotonic() - t0) / 20.0)
                    on_progress((i + 0.95 * frac) / len(cmds))
            if proc.returncode != 0:
                raise RuntimeError(
                    "Не удалось установить PyTorch/transformers для Depth Pro.\n"
                    f"Подробности в файле:\n{log_path}")
    global _DEPTH_PRO_MODULES
    _DEPTH_PRO_MODULES = None


def ensure_depth_pro(on_progress=None):
    """Доставить зависимости и веса — вызывается только при первом выборе."""
    def part(lo, hi):
        def f(frac):
            if on_progress is not None:
                on_progress(lo + (hi - lo) * float(frac))
        return f

    if not depth_pro_modules_ok():
        install_depth_pro_deps(part(0.0, 0.5))
    d = depth_pro_dir()
    todo = [f for f in DEPTH_PRO_FILES if not os.path.exists(os.path.join(d, f))]
    for i, name in enumerate(todo):
        download_hf_file(name, d,
                         part(0.5 + 0.5 * i / len(todo),
                              0.5 + 0.5 * (i + 1) / len(todo)))


def _depth_pro_side():
    """Сторона квадрата для сети. Меньше 1536 нельзя: патчи по 384 точки
    должны делиться на самом мелком масштабе (0.25), иначе модель падает."""
    env = os.environ.get("DEPTH_TOOLS_DEPTHPRO_SIDE")
    try:
        return max(1536, int(env)) if env else 1536
    except ValueError:
        return 1536


def _infer_depth_pro(model_dir, img, nw, nh, on_tick=None):
    import subprocess
    import sys
    import tempfile
    import time
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_depth_pro_infer.py")
    side = _depth_pro_side()
    av = avail_ram_mb()
    if av is not None and av < 4200:
        raise MemoryError(
            "Depth Pro нужно ~6 ГБ свободной ОЗУ.\n"
            "Закройте лишние программы или выберите модель поменьше.")
    tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tf, "PNG")
    tf.close()
    tmp_in = tf.name
    tmp_out = tempfile.mktemp(suffix=".npy")
    tmp_meta = tmp_out + ".json"
    log_path = os.path.join(OUT, "depth_pro_run.log")
    fov_y = DEFAULT_FOV_Y
    try:
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                [sys.executable, worker, model_dir, str(nw), str(nh), str(side),
                 tmp_in, tmp_out, tmp_meta],
                stdout=log, stderr=subprocess.STDOUT)
            t0 = time.monotonic()
            while proc.poll() is None:
                time.sleep(0.25)
                if on_tick is not None:
                    on_tick(min(0.99, 1.0 - math.exp(-(time.monotonic() - t0) / 25.0)))
                av = avail_ram_mb()
                if av is not None and av < 400:
                    proc.kill()
                    raise MemoryError("Depth Pro остановлен: свободная ОЗУ закончилась.")
        if proc.returncode != 0:
            if proc.returncode < 0:
                raise MemoryError(
                    "Depth Pro остановлен: не хватило ОЗУ.\n"
                    "Закройте лишние программы или выберите модель поменьше.")
            tail = ""
            try:
                with open(log_path) as f:
                    tail = "".join(f.readlines()[-3:]).strip()
            except Exception:
                pass
            raise RuntimeError("Depth Pro не запустилась" + (f":\n{tail}" if tail else "."))
        if not os.path.exists(tmp_out):
            raise RuntimeError("Depth Pro не вернула карту глубины.")
        d = np.load(tmp_out).astype(np.float32)
        try:
            with open(tmp_meta) as f:
                meta = json.load(f)
            fov_x = float(meta.get("fov_x") or 0.0)
            if 5.0 <= fov_x <= 150.0:
                # вертикальный угол считаем по пропорциям самого фото, а не по
                # размеру входа сети: он округляется до кратного 14 и врёт на 5–6 %
                fov_y = float(np.degrees(2.0 * np.arctan(
                    np.tan(np.deg2rad(fov_x) / 2.0) * float(img.height) / float(img.width))))
        except Exception:
            pass
    finally:
        for t in (tmp_in, tmp_out, tmp_meta):
            try:
                if t:
                    os.remove(t)
            except Exception:
                pass
    return d, fov_y


def open_path(path):
    try:
        if os.name == "nt":
            os.startfile(path)
        else:
            os.system(f'xdg-open "{path}" &')
    except Exception:
        pass


def avail_ram_mb():
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullAvailPhys // (1024 * 1024))
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return None
    return None


def total_ram_mb():
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                            ("ullAvailExtendedVirtual", ctypes.c_uint64)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullTotalPhys // (1024 * 1024))
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return None
    return None


def colormap_rgb(t):
    stops = [
        (0.0, (0.0, 0.0, 0.016)),
        (0.1, (0.086, 0.043, 0.224)),
        (0.2, (0.259, 0.039, 0.408)),
        (0.3, (0.416, 0.090, 0.431)),
        (0.4, (0.576, 0.149, 0.404)),
        (0.5, (0.733, 0.216, 0.329)),
        (0.6, (0.867, 0.318, 0.227)),
        (0.7, (0.953, 0.467, 0.098)),
        (0.8, (0.980, 0.678, 0.110)),
        (0.9, (0.988, 0.933, 0.125)),
        (1.0, (0.988, 1.0, 0.643)),
    ]
    r = np.interp(t, [s[0] for s in stops], [s[1][0] for s in stops])
    g = np.interp(t, [s[0] for s in stops], [s[1][1] for s in stops])
    b = np.interp(t, [s[0] for s in stops], [s[1][2] for s in stops])
    return np.stack([r, g, b], axis=-1)


def _srgb_to_linear(a):
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def write_exr(path, img):
    import struct as _s
    a = np.nan_to_num(np.asarray(img, dtype=np.float32), nan=0.0)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    h, w = a.shape[:2]
    has_a = a.shape[2] == 4

    def attr(name, typ, data):
        return name.encode() + b"\0" + typ.encode() + b"\0" + _s.pack("<i", len(data)) + data

    # каналы обязаны идти в алфавитном порядке: A, B, G, R
    chans = b""
    order = ([("A", 3)] if has_a else []) + [("B", 2), ("G", 1), ("R", 0)]
    for nm, src in order:
        chans += nm.encode() + b"\0" + _s.pack("<i", 2) + _s.pack("<B", 0) + b"\0\0\0" + _s.pack("<ii", 1, 1)
    chans += b"\0"
    hdr = b"\x76\x2f\x31\x01" + _s.pack("<I", 2)
    hdr += attr("channels", "chlist", chans)
    hdr += attr("compression", "compression", _s.pack("<B", 0))
    hdr += attr("dataWindow", "box2i", _s.pack("<iiii", 0, 0, w - 1, h - 1))
    hdr += attr("displayWindow", "box2i", _s.pack("<iiii", 0, 0, w - 1, h - 1))
    hdr += attr("lineOrder", "lineOrder", _s.pack("<B", 0))
    hdr += attr("pixelAspectRatio", "float", _s.pack("<f", 1.0))
    hdr += attr("screenWindowCenter", "v2f", _s.pack("<ff", 0.0, 0.0))
    hdr += attr("screenWindowWidth", "float", _s.pack("<f", 1.0))
    hdr += b"\0"
    planes = ([a[..., 3]] if has_a else []) + [a[..., 2], a[..., 1], a[..., 0]]
    planes32 = [p.astype(np.float32) for p in planes]
    npl = len(planes)
    offsets = []
    pos = len(hdr) + 8 * h
    chunks = []
    # стандартный порядок: чанк y = строка y (сверху вниз), lineOrder=INCREASING_Y
    for y in range(h):
        offsets.append(pos)
        data = b"".join(p[y].tobytes() for p in planes32)
        chunks.append(_s.pack("<ii", y, len(data)) + data)
        pos += 8 + len(data)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(_s.pack("<" + "Q" * h, *offsets))
        for ch in chunks:
            f.write(ch)


def _exr_planes(path):
    # чтение несжатых scanline-EXR (формат этого инструмента) -> {канал: float2D}; иначе None
    try:
        import struct as _st
        d = open(path, "rb").read()
        if d[:4] != b"\x76\x2f\x31\x01":
            return None
        pos = 8
        attrs = {}
        while d[pos] != 0:
            n = d.index(0, pos); name = d[pos:n].decode(); pos = n+1
            t = d.index(0, pos); typ = d[pos:t].decode(); pos = t+1
            size = _st.unpack("<i", d[pos:pos+4])[0]
            attrs[name] = (typ, d[pos+4:pos+4+size]); pos += 4+size
        pos += 1
        if attrs.get("compression", ("", b"\xff"))[1][0] != 0:
            return None
        x1, y1, x2, y2 = _st.unpack("<iiii", attrs["dataWindow"][1])
        w, h = x2-x1+1, y2-y1+1
        chans = []
        blob = attrs["channels"][1]
        p = 0
        types = []
        while blob[p] != 0:
            e = blob.index(0, p); nm = blob[p:e].decode(); p = e+1
            types.append(_st.unpack("<i", blob[p:p+4])[0]); p += 16
            chans.append(nm)
        if types and any(t not in (1, 2) for t in types):
            return None  # только half (1) и float32 (2)
        dec = attrs.get("lineOrder", ("", b"\x00"))[1][0] == 1
        offs = _st.unpack("<" + "Q"*h, d[pos:pos+8*h])
        planes = {}
        for ci, nm in enumerate(chans):
            dt = "<f2" if types[ci] == 1 else "<f4"
            bps = 2 if types[ci] == 1 else 4
            arr = np.empty((h, w), dtype=np.float32)
            for y in range(h):
                o = offs[y] + 8 + ci*w*bps
                row = np.frombuffer(d[o:o+w*bps], dtype=dt).astype(np.float32)
                # DECREASING_Y: чанк 0 = нижняя строка файла
                arr[h-1-y if dec else y] = row
            planes[nm] = arr
        return planes
    except Exception:
        return None


def _exr_load_rgb(path):
    # полное чтение несжатых scanline-EXR -> PIL RGB; иначе None
    planes = _exr_planes(path)
    if not planes or "R" not in planes:
        return None
    rgb = np.stack([planes.get("R"), planes.get("G"), planes.get("B")], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    lin = np.clip(rgb, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin*12.92,
                    1.055*np.power(np.clip(lin, 1e-8, None), 1/2.4) - 0.055)
    from PIL import Image as _I
    return _I.fromarray((np.clip(srgb, 0.0, 1.0)*255).astype(np.uint8)).convert("RGB")


def _linear_to_srgb(a):
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.0031308, a*12.92,
                    1.055*np.power(np.clip(a, 1e-8, None), 1/2.4) - 0.055)


def _exr_thumb(path, max_side=150):
    im = _exr_load_rgb(path)
    if im is None:
        return None
    im.thumbnail((max_side, max_side), Image.BILINEAR)
    return im


def png_write_rgb16(path, rgb01):
    import zlib
    import struct
    h, w = rgb01.shape[:2]
    v = np.clip(rgb01, 0.0, 1.0)
    # PNG хранит 16-битные отсчёты в big-endian, а .astype(uint16) на x86 даёт
    # little-endian — без ">u2" байты меняются местами и цвета читаются мусором
    data = np.round(v * 65535.0).astype(">u2")
    raw = b"".join(b"\x00" + data[y].tobytes() for y in range(h))

    def chunk(tag, payload):
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _depth01(depth):
    """Приводит карту к float32 0..1 (1.0 = близко к камере)."""
    if isinstance(depth, Image.Image):
        return np.asarray(depth.convert("L"), dtype=np.float32) / 255.0
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    return np.clip(np.nan_to_num(d), 0.0, 1.0)


def _slope_field(depth, radius=6.0, strength=1.5):
    """Наклоны карты глубины (dh/dx, dh/dy) в единицах, не зависящих от размера.

    Без авто-масштаба наклоны в 0..1-карте порядка 0.001 на пиксель: края
    выжигаются, а середина остаётся плоской. Масштаб берём по 90-му
    процентилю, а всё вместе нормируем на «1000 px», чтобы картинка
    выглядела одинаково при выходе 300 px и 3000 px.
    """
    d = _depth01(depth)
    h, w = d.shape
    k = max(1.0, max(h, w) / 1000.0)
    d = _blur_f(d, max(0.0, float(radius)) * k)
    gy, gx = np.gradient(d)
    gx, gy = gx * k, gy * k
    amp = float(np.percentile(np.hypot(gx, gy), 90)) or 1.0
    gain = max(0.0, float(strength)) / max(amp, 1e-6)
    return gx * gain, gy * gain


def normal_map(depth, radius=6.0, strength=2.5):
    """Карта нормалей (tangent space, RGB 0..1) из карты глубины."""
    gx, gy = _slope_field(depth, radius, strength)
    nx, ny, nz = -gx, -gy, np.ones_like(gx)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / ln, ny / ln, nz / ln], axis=-1).astype(np.float32) * 0.5 + 0.5


def _points_field(dfloat, img_rgb, fov_y=DEFAULT_FOV_Y, scale=1.0):
    d = np.clip(np.asarray(dfloat, dtype=np.float32), 0.0, 1.0)
    h, w = d.shape
    z = np.clip(d, 0.05, 1.0) * float(scale)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    fy = (h / 2.0) / math.tan(math.radians(max(1.0, float(fov_y)) / 2.0))
    ys, xs = np.mgrid[0:h, 0:w]
    # float32: при выходе 4x точек ~17 млн, на float64 это лишние сотни МБ
    k = z.astype(np.float32) / np.float32(fy)
    X = ((xs - cx) * k).astype(np.float32)
    Y = (-(ys - cy) * k).astype(np.float32)
    rgb = np.asarray(img_rgb.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
    return X.reshape(-1), Y.reshape(-1), z.reshape(-1).astype(np.float32), rgb


def _meshgrid(h, w):
    ys, xs = np.mgrid[0:h, 0:w]
    return xs / (w - 1) * 2 - 1, ys / (h - 1) * 2 - 1


def _fbm(h, w, seed, octaves=5, scale=2.6):
    rng = np.random.RandomState(seed)
    xs, ys = np.mgrid[0:h, 0:w].astype(np.float32)
    total = np.zeros((h, w), dtype=np.float32)
    norm = 0.0
    amp = 1.0
    per = max(2.0, float(scale))
    for _ in range(octaves):
        nx = int(np.ceil(w / per)) + 2
        ny = int(np.ceil(h / per)) + 2
        g = rng.rand(ny + 1, nx + 1)
        xx = xs / per
        yy = ys / per
        gx = np.clip(xx.astype(int), 0, nx - 1)
        gy = np.clip(yy.astype(int), 0, ny - 1)
        wx = xx - xx.astype(int)
        wy = yy - yy.astype(int)
        v = (g[gy, gx] * (1 - wx) * (1 - wy)
             + g[gy, gx + 1] * wx * (1 - wy)
             + g[gy + 1, gx] * (1 - wx) * wy
             + g[gy + 1, gx + 1] * wx * wy)
        total += v * amp
        norm += amp
        amp *= 0.5
        per *= 1.9
    return total / norm


def gen_depth_map(kind, h, w, amp=1.0, freq=2.0):
    amp = max(0.05, float(amp))
    f = max(0.01, float(freq))
    kind = str(kind or "waves")
    if kind == "waves":
        X, Y = _meshgrid(h, w)
        z = 0.5 + 0.5 * np.sin(X * np.pi * f) * np.sin(Y * np.pi * f)
        z = z * 0.75 + 0.25 * np.sin((X + Y) * np.pi * f * 1.7)
        z = z - z.min()
        z = z / (z.max() + 1e-8)
    elif kind == "bumps":
        z = np.zeros((h, w), dtype=np.float32)
        X, Y = _meshgrid(h, w)
        rng = np.random.RandomState(11)
        for _ in range(int(round(5 * amp))):
            cx, cy = rng.uniform(-1.1, 1.1, 2)
            r = rng.uniform(0.25, 0.75)
            z += rng.uniform(0.5, 1.0) * np.exp(-(((X - cx) ** 2 + (Y - cy) ** 2) / (2 * r * r)))
        z = z - z.min()
        z = z / (z.max() + 1e-8)
    elif kind == "craters":
        z = np.zeros((h, w), dtype=np.float32)
        X, Y = _meshgrid(h, w)
        rng = np.random.RandomState(21)
        for _ in range(int(round(3 * amp))):
            cx, cy = rng.uniform(-1.0, 1.0, 2)
            rc = rng.uniform(0.3, 0.8)
            r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            z += rng.uniform(0.5, 1.0) * (np.exp(-((r - rc) ** 2) / 0.02) * (1 - np.exp(-((r + 0.05) * 8) ** 2)))
        z = z - z.min()
        z = z / (z.max() + 1e-8)
    elif kind == "spiral":
        X, Y = _meshgrid(h, w)
        a = np.arctan2(Y, X) / (2 * np.pi)
        r = np.sqrt(X * X + Y * Y)
        z = 0.5 + 0.5 * np.sin((a * f + r * 2.2) * 2 * np.pi)
        z = z * (1 - r * 0.4)
        z = z - z.min()
        z = z / (z.max() + 1e-8)
    elif kind == "noise":
        z = _fbm(h, w, int(round(f * 10 % 997)), octaves=5, scale=2.6 * f)
        z = z - z.min()
        z = z / (z.max() + 1e-8)
    elif kind == "ridges":
        z = _fbm(h, w, int(round(f * 10 % 991)), octaves=6, scale=2.2 * f)
        z = 1.0 - np.abs(z * 2 - 1)
        z = np.power(np.clip(z, 0.0, 1.0), 1.4)
        z = z - z.min()
        z = z / (z.max() + 1e-8)
    elif kind in ("valley", "peaks", "dunes", "caldera", "canyon"):
        _b = (np.linspace(0.0, 1.0, h)[:, None] ** 1.3).astype(np.float32)
        if kind == "valley":
            Y = _meshgrid(h, w)[1]
            z = _b * (0.45 + 0.55 * np.abs(Y))
            z = z + (0.5 - 0.5 * np.abs(Y)) * 0.35
        elif kind == "peaks":
            z = _fbm(h, w, 13, octaves=6, scale=2.8 * f)
            z = 1.0 - np.abs(z * 2 - 1)
            z = np.power(np.clip(z, 0.0, 1.0), 1.5)
            z = _b * 0.35 + z * 0.65
        elif kind == "dunes":
            Y, X = _meshgrid(h, w)[1], _meshgrid(h, w)[0]
            z = _b + 0.45 * np.sin(X * np.pi * f * 0.6 + Y * 3.1) * np.abs(np.sin(Y * 2.2 + 1.3))
            z = z + 0.15 * (_fbm(h, w, 17, octaves=4, scale=2.0) - 0.5)
        elif kind == "caldera":
            Y, X = _meshgrid(h, w)[1], _meshgrid(h, w)[0]
            cx, cy = 0.0, -0.45
            r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            z = np.exp(-((r - 0.55) ** 2) / 0.04) * (1 + Y) * 0.8
            z = z + np.exp(-(r ** 2) / 0.35) * 0.3
            z = z * (1.0 - np.exp(-((r + 0.4) * 6) ** 2))
            z = _b * 0.25 + z
        elif kind == "canyon":
            z = _fbm(h, w, int(round(f * 10 % 977)), octaves=6, scale=2.4 * f)
            z = (1.0 - np.abs(z * 2 - 1))
            z = np.power(np.clip(z, 0.0, 1.0), 0.8)
            z = _b * (0.3 + 0.7 * z)
        z = z - z.min()
        z = z / (z.max() + 1e-8)
    else:
        X, Y = _meshgrid(h, w)
        z = 0.5 + 0.5 * np.sin(X * np.pi * f) * np.sin(Y * np.pi * f)
        z = z - z.min()
        z = z / (z.max() + 1e-8)
    z = np.asarray(z, dtype=np.float32)
    return np.clip(z * (float(amp) ** 0.6), 0.0, 1.0)


def gen_mesh(kind, size=1.0, freq=2.0, res=96):
    size = max(0.05, float(size))
    turns = max(0.1, float(freq))
    res = int(res)
    kind = str(kind or "torus")
    u = np.linspace(0, 2 * np.pi, res, dtype=np.float32)
    v = np.linspace(0, 2 * np.pi, res, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing="ij")
    if kind == "sphere":
        x = (size * np.sin(V) * np.cos(U)).reshape(-1)
        y = (size * np.sin(V) * np.sin(U)).reshape(-1)
        z = (size * np.cos(V)).reshape(-1)
    elif kind == "helix":
        nw = int(round(turns * res))
        uu = np.linspace(0, 2 * np.pi * turns, nw, dtype=np.float32)
        UU, VV = np.meshgrid(uu, v, indexing="ij")
        R = size * 0.6
        r = size * 0.16
        c = np.cos(UU)
        s = np.sin(UU)
        x = ((R + r * np.cos(VV)) * c).reshape(-1)
        y = ((R + r * np.cos(VV)) * s).reshape(-1)
        z = (size * 0.9 * UU / (2 * np.pi * turns) + r * np.sin(VV)).reshape(-1)
        U = UU
        V = VV
    elif kind == "mobius":
        tw = int(round(turns))
        ww = np.linspace(0, 2 * np.pi * tw, res, dtype=np.float32)
        WW, VV = np.meshgrid(ww, np.linspace(-1, 1, res, dtype=np.float32), indexing="ij")
        hw = WW * 0.5 * tw
        x = ((1 + VV * 0.4 * np.cos(hw + WW * (tw % 2) * 0)) * np.cos(WW)).reshape(-1)
        y = ((1 + VV * 0.4 * np.cos(hw + WW * (tw % 2) * 0)) * np.sin(WW)).reshape(-1)
        z = (VV * 0.4 * np.sin(hw + WW * (tw % 2) * 0)).reshape(-1)
        x = (x * size * 0.6).reshape(-1)
        y = (y * size * 0.6).reshape(-1)
        z = (z * size * 0.6).reshape(-1)
        U = WW
        V = VV
    elif kind == "superell":
        eu = max(0.3, (turns - 0.5) / 1.5)
        ev = eu
        x = (size * np.sign(np.cos(U)) * np.abs(np.cos(U)) ** eu
             * np.sign(np.cos(V)) * np.abs(np.cos(V)) ** ev).reshape(-1)
        y = (size * np.sign(np.cos(U)) * np.abs(np.cos(U)) ** eu
             * np.sign(np.sin(V)) * np.abs(np.sin(V)) ** ev).reshape(-1)
        z = (size * np.sign(np.sin(U)) * np.abs(np.sin(U)) ** eu).reshape(-1)
    else:  # torus
        R = size
        r = size * 0.35
        x = ((R + r * np.cos(V)) * np.cos(U)).reshape(-1)
        y = ((R + r * np.cos(V)) * np.sin(U)).reshape(-1)
        z = (r * np.sin(V)).reshape(-1)
    nx, ny = U.shape
    verts = np.stack([x, y, z], axis=-1)
    mx = float(np.abs(verts).max())
    if mx > 0:
        verts = verts / mx * (float(size))
    cols = np.stack([
        (0.5 + 0.5 * np.cos(U) * np.cos(V)).reshape(-1),
        (0.5 + 0.5 * np.sin(U)).reshape(-1),
        (0.5 + 0.5 * np.sin(V)).reshape(-1),
    ], axis=-1)
    cols = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            faces.append((a, a + 1, a + ny + 1))
            faces.append((a, a + ny + 1, a + ny))
    return verts, cols, np.asarray(faces, dtype=np.uint32)


def _write_mesh_ply(path, verts, cols, faces):
    n = verts.shape[0]
    with open(path, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\nelement vertex %d\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "element face %d\nproperty list uchar int vertex_indices\n"
            "end_header\n" % (n, n_faces)
        )
        np.savetxt(f, np.column_stack([verts, cols]).astype(np.float32),
                   fmt="%.6f %.6f %.6f %d %d %d")
        for fc in faces:
            f.write("3 %d %d %d\n" % (fc[0], fc[1], fc[2]))


def _write_mesh_glb(path, verts, cols, faces):
    import struct
    n = verts.shape[0]
    pos = verts.astype(np.float32)
    lcol = ((cols / 255.0) ** 2.2).astype(np.float32)
    col = np.concatenate([lcol, np.ones((n, 1), dtype=np.float32)], axis=-1)
    idx = faces.astype(np.uint32).reshape(-1)
    v0, v1, v2 = pos[idx[0::3]], pos[idx[1::3]], pos[idx[2::3]]
    nrm = np.cross(v1 - v0, v2 - v0)
    nl = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = (nrm / np.maximum(nl, 1e-12)).astype(np.float32)
    vn = np.zeros_like(pos)
    np.add.at(vn, idx[0::3], nrm)
    np.add.at(vn, idx[1::3], nrm)
    np.add.at(vn, idx[2::3], nrm)
    nl2 = np.linalg.norm(vn, axis=-1, keepdims=True)
    vn = (vn / np.maximum(nl2, 1e-12)).astype(np.float32)
    ib = idx.tobytes()
    pb = pos.tobytes()
    nb = vn.tobytes()
    cb = col.tobytes()
    ib = ib + b"\x00" * (-len(ib) % 4)
    bin_ = ib + pb + nb + cb
    i_off, p_off, n_off, c_off = 0, len(ib), len(ib) + len(pb), len(ib) + len(pb) + len(nb)
    acc = [
        {"bufferView": 0, "componentType": 5125, "count": idx.size, "type": "SCALAR"},
        {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3",
         "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
        {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC3"},
        {"bufferView": 3, "componentType": 5126, "count": n, "type": "VEC4"},
    ]
    bv = [
        {"buffer": 0, "byteOffset": i_off, "byteLength": len(ib), "target": 34963},
        {"buffer": 0, "byteOffset": p_off, "byteLength": len(pb), "target": 34962},
        {"buffer": 0, "byteOffset": n_off, "byteLength": len(nb), "target": 34962},
        {"buffer": 0, "byteOffset": c_off, "byteLength": len(cb), "target": 34962},
    ]
    doc = {
        "asset": {"version": "2.0", "generator": "depth-tools"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "photo_points"}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 1, "NORMAL": 2, "COLOR_0": 3},
            "indices": 0, "mode": 4}]}],
        "buffers": [{"byteLength": len(bin_)}],
        "bufferViews": bv,
        "accessors": acc,
    }
    json_ = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    json_ = json_ + b"\x20" * (-len(json_) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_) + 8 + len(bin_)))
        f.write(struct.pack("<II", len(json_), 0x4E4F534A))
        f.write(json_)
        f.write(struct.pack("<II", len(bin_), 0x004E4942))
        f.write(bin_)


def _infer_depth(model_path, img, nw, nh, metric=False, on_tick=None):
    import subprocess
    import sys
    import tempfile
    import time
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_depth_infer.py")
    tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tf, "PNG")
    tf.close()
    tmp_in = tf.name
    tf = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    tf.close()
    tmp_out = tf.name
    try:
        proc = subprocess.Popen([sys.executable, worker, model_path, str(nw), str(nh), tmp_in, tmp_out])
        total = total_ram_mb()
        floor = max(400.0, (total or 0.0) * 0.08)
        t0 = time.monotonic()
        while proc.poll() is None:
            time.sleep(0.2)
            if on_tick is not None:
                # длительность заранее неизвестна: прогресс плавно подтягивается
                # к 90% шага и ждёт фактического окончания
                on_tick(min(0.9, 1.0 - math.exp(-(time.monotonic() - t0) / 3.5)))
            av = avail_ram_mb()
            if av is not None and av < floor:
                proc.kill()
                raise MemoryError("Обработка остановлена: свободная ОЗУ закончилась.\n"
                                  "Уменьшите «Множитель входа» или «Сжать оригинал до».")
        if proc.returncode != 0:
            raise RuntimeError(f"Ошибка вычисления глубины (код {proc.returncode}).")
        d = np.load(tmp_out).astype(np.float32)
    finally:
        for t in (tmp_in, tmp_out):
            try:
                if t:
                    os.remove(t)
            except Exception:
                pass
    d = d - d.min()
    d = d / (d.max() + 1e-8)
    if metric:
        d = 1.0 - d
    return d


def write_ply(path, dfloat, img_rgb, fov_y=DEFAULT_FOV_Y, scale=1.0, on_progress=None):
    X, Y, z, rgb = _points_field(dfloat, img_rgb, fov_y=fov_y, scale=scale)
    h, w = np.asarray(dfloat).shape[:2]
    n = X.size
    n_faces = (h - 1) * (w - 1)
    with open(path, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\nelement vertex %d\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "element face %d\nproperty list uchar int vertex_indices\n"
            "end_header\n" % (n, n_faces)
        )
        # пишем полосами по строкам: на выходе 4x склеивание всех точек в один
        # массив съедало больше гигабайта ОЗУ ради текстового файла
        Xr, Yr, zr = X.reshape(h, w), Y.reshape(h, w), z.reshape(h, w)
        rgbr = rgb.reshape(h, w, 3)
        step = max(1, 2000000 // max(1, w))
        for i0 in range(0, h, step):
            i1 = min(h, i0 + step)
            blk = np.column_stack([Xr[i0:i1].ravel(), Yr[i0:i1].ravel(), zr[i0:i1].ravel(),
                                   rgbr[i0:i1].reshape(-1, 3)])
            np.savetxt(f, blk, fmt="%.6f %.6f %.6f %d %d %d")
            if on_progress is not None:
                on_progress(i1 / float(h))
        # грани — против часовой стрелки со стороны камеры (a, a+w, a+w+1, a+1),
        # иначе в Blender и просмотрщиках модель выглядит вывернутой наизнанку
        for i in range(h - 1):
            base = i * w
            f.writelines("4 %d %d %d %d\n" % (base + j, base + j + w, base + j + w + 1, base + j + 1)
                         for j in range(w - 1))


def write_glb(path, dfloat, img_rgb, fov_y=DEFAULT_FOV_Y, scale=1.0):
    import struct
    X, Y, z, rgb = _points_field(dfloat, img_rgb, fov_y=fov_y, scale=scale)
    h, w = dfloat.shape[0], dfloat.shape[1]
    n = X.size
    pos = np.stack([X, Y, z], axis=-1).astype(np.float32)
    lcol = ((rgb / 255.0) ** 2.2).astype(np.float32)
    col = np.concatenate([lcol, np.ones((n, 1), dtype=np.float32)], axis=-1)
    # индексы треугольников считаем сразу массивом: питоновский список на
    # 2 млн граней — это сотни мегабайт и лишние секунды
    a = (np.arange(h - 1, dtype=np.uint32)[:, None] * w
         + np.arange(w - 1, dtype=np.uint32)[None, :]).ravel()
    idx = np.empty(a.size * 6, dtype=np.uint32)
    # обход против часовой стрелки со стороны камеры
    idx[0::6] = a
    idx[1::6] = a + w
    idx[2::6] = a + w + 1
    idx[3::6] = a
    idx[4::6] = a + w + 1
    idx[5::6] = a + 1
    v0, v1, v2 = pos[idx[0::3]], pos[idx[1::3]], pos[idx[2::3]]
    nrm = np.cross(v1 - v0, v2 - v0)
    nl = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = (nrm / np.maximum(nl, 1e-12)).astype(np.float32)
    vn = np.zeros_like(pos)
    np.add.at(vn, idx[0::3], nrm)
    np.add.at(vn, idx[1::3], nrm)
    np.add.at(vn, idx[2::3], nrm)
    nl2 = np.linalg.norm(vn, axis=-1, keepdims=True)
    vn = (vn / np.maximum(nl2, 1e-12)).astype(np.float32)
    ib = idx.tobytes()
    pb = pos.tobytes()
    nb = vn.tobytes()
    cb = col.tobytes()
    ib = ib + b"\x00" * (-len(ib) % 4)
    bin_ = ib + pb + nb + cb
    i_off, p_off, n_off, c_off = 0, len(ib), len(ib) + len(pb), len(ib) + len(pb) + len(nb)
    acc = [
        {"bufferView": 0, "componentType": 5125, "count": idx.size, "type": "SCALAR"},
        {"bufferView": 1, "componentType": 5126, "count": n, "type": "VEC3",
         "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
        {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC3"},
        {"bufferView": 3, "componentType": 5126, "count": n, "type": "VEC4"},
    ]
    bv = [
        {"buffer": 0, "byteOffset": i_off, "byteLength": len(ib), "target": 34963},
        {"buffer": 0, "byteOffset": p_off, "byteLength": len(pb), "target": 34962},
        {"buffer": 0, "byteOffset": n_off, "byteLength": len(nb), "target": 34962},
        {"buffer": 0, "byteOffset": c_off, "byteLength": len(cb), "target": 34962},
    ]
    doc = {
        "asset": {"version": "2.0", "generator": "depth-tools"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "photo_points"}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 1, "NORMAL": 2, "COLOR_0": 3},
            "indices": 0, "mode": 4}]}],
        "buffers": [{"byteLength": len(bin_), "uri": "data:..."}],
        "bufferViews": bv,
        "accessors": acc,
    }
    doc["buffers"][0].pop("uri")
    json_ = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    json_ = json_ + b"\x20" * (-len(json_) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_) + 8 + len(bin_)))
        f.write(struct.pack("<II", len(json_), 0x4E4F534A))
        f.write(json_)
        f.write(struct.pack("<II", len(bin_), 0x004E4942))
        f.write(bin_)


def _box_f(a, r):
    ri = max(0, int(round(r)))
    a = np.ascontiguousarray(a, dtype=np.float64)
    if ri == 0:
        return a.astype(np.float32)
    h, w = a.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.clip(ys - ri, 0, h)
    y1 = np.clip(ys + ri + 1, 0, h)
    x0 = np.clip(xs - ri, 0, w)
    x1 = np.clip(xs + ri + 1, 0, w)
    s = ii[y1][:, x1] - ii[y0][:, x1] - ii[y1][:, x0] + ii[y0][:, x0]
    area = (y1 - y0)[:, None].astype(np.float64) * (x1 - x0)[None, :]
    return (s / area).astype(np.float32)


def guided_filter(guide, src, radius, eps=0.01):
    mean_i = _box_f(guide, radius)
    mean_p = _box_f(src, radius)
    corr_ip = _box_f(guide * src, radius)
    corr_ii = _box_f(guide * guide, radius)
    var_i = corr_ii - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box_f(a, radius) * guide + _box_f(b, radius)


def _blur_f(a, r):
    out = a
    rr = max(1, int(round(float(r) * 0.6)))
    for _ in range(3):
        out = _box_f(out, rr)
    return out


class NiceSlider(tk.Canvas):
    """Ползунок на Canvas: ровный желоб, заливка до бегунка и круглая ручка.

    Штатный ttk.Scale в теме clam рисует тонкую полоску с квадратиком, который
    на тёмном фоне почти не видно, поэтому рисуем сами.
    """

    H = 26
    PAD = 10      # отступ ручки от краёв, чтобы она не вылезала за виджет
    KNOB = 6      # радиус ручки

    def __init__(self, master, variable, from_=0.0, to=100.0, command=None, **kw):
        super().__init__(master, height=self.H, bg=CARD, highlightthickness=0, bd=0, **kw)
        self.var = variable
        self.lo = float(from_)
        self.hi = float(to)
        self.cmd = command
        self._drag = False
        self.bind("<Configure>", lambda _e: self._draw())
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Enter>", lambda _e: self.configure(cursor="hand2"))
        self.bind("<Leave>", lambda _e: self.configure(cursor=""))
        for seq in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
            self.bind(seq, self._wheel)
        self.var.trace_add("write", lambda *_: self._draw())

    def _geom(self):
        w = max(1, self.winfo_width())
        span = (self.hi - self.lo) or 1.0
        t = (float(self.var.get()) - self.lo) / span
        t = min(1.0, max(0.0, t))
        x0 = float(self.PAD)
        x1 = max(x0 + 1.0, w - self.PAD)
        return x0, x1, x0 + (x1 - x0) * t

    def _draw(self):
        self.delete("all")
        x0, x1, x = self._geom()
        cy = self.H / 2.0
        self.create_rectangle(x0, cy - 2, x1, cy + 2, fill=CARD2, outline="")
        if x > x0:
            self.create_rectangle(x0, cy - 2, x, cy + 2, fill=ACCENT, outline="")
        self.create_oval(x - self.KNOB, cy - self.KNOB, x + self.KNOB, cy + self.KNOB,
                         fill="#ffffff", outline=ACCENT, width=2)

    def _set_from_x(self, px):
        x0, x1, _ = self._geom()
        t = (float(px) - x0) / max(1.0, x1 - x0)
        t = min(1.0, max(0.0, t))
        self.var.set(round(self.lo + (self.hi - self.lo) * t, 4))
        if self.cmd:
            self.cmd(self.var.get())

    def _wheel(self, e):
        # колесо над ползунком двигает его влево/вправо, а не прокручивает панель
        up = getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0
        step = (self.hi - self.lo) / 100.0 * (1.0 if up else -1.0)
        self.var.set(round(min(self.hi, max(self.lo, float(self.var.get()) + step)), 4))
        if self.cmd:
            self.cmd(self.var.get())
        return "break"

    def _press(self, e):
        self._drag = True
        self._set_from_x(e.x)

    def _move(self, e):
        if self._drag:
            self._set_from_x(e.x)

    def _release(self, _e):
        self._drag = False


class NiceCheck(tk.Canvas):
    """Флажок с квадратиком и подписью: в теме clam флажки рисуются светлыми
    квадратами, которые на тёмном фоне выглядят как дырки, поэтому рисуем сами."""

    SIZE = 16
    GAP = 8
    H = 22

    def __init__(self, master, text, variable, bg=CARD, fg=FG, **kw):
        import tkinter.font as tkfont
        f = tkfont.nametofont("TkDefaultFont")
        super().__init__(master, width=self.SIZE + self.GAP + f.measure(text) + 4,
                         height=self.H, bg=bg, highlightthickness=0, bd=0,
                         cursor="hand2", **kw)
        self.var = variable
        self.text = text
        self._fg = fg
        self._hover = False
        self.bind("<Button-1>", lambda _e: self.var.set(not self.var.get()))
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.var.trace_add("write", lambda *_: self._draw())
        self._draw()

    def _on_enter(self, _e):
        self._hover = True
        self._draw()

    def _on_leave(self, _e):
        self._hover = False
        self._draw()

    def _draw(self):
        self.delete("all")
        s = self.SIZE
        y0 = (self.H - s) / 2.0
        on = bool(self.var.get())
        if on:
            fill, outline = ACCENT, ACCENT
        else:
            fill = "#2f3540" if self._hover else CARD2
            outline = ACCENT if self._hover else BORDER
        self.create_rectangle(0.5, y0 + 0.5, s - 0.5, y0 + s - 0.5,
                              fill=fill, outline=outline, width=1)
        if on:
            self.create_line(4, y0 + 8, 7, y0 + 11, 12, y0 + 5, fill="#ffffff",
                             width=2, capstyle="round", joinstyle="round")
        self.create_text(s + self.GAP, self.H / 2.0, text=self.text, anchor="w",
                         fill=self._fg, font="TkDefaultFont")


class ApplyButton(tk.Canvas):
    """Кнопка «Применить» во всю ширину. Пока идёт обработка, она же
    показывает процент: синяя заливка доходит до края, надпись меняется на «42%»."""

    H = 42

    def __init__(self, master, text="Применить", command=None, bg=BG, font=None, **kw):
        super().__init__(master, height=self.H, bg=bg, highlightthickness=0, bd=0,
                         cursor="hand2", **kw)
        self._text = text
        self._cmd = command
        self._font = font
        self._pct = None
        self._hover = False
        self._enabled = True
        self.bind("<Button-1>", self._click)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<Configure>", lambda _e: self._draw())
        self._draw()

    def set_busy(self, on):
        self._pct = 0.0 if on else None
        self._enabled = not on
        self.configure(cursor="" if on else "hand2")
        self._draw()

    def set_progress(self, pct):
        if self._pct is None:
            return
        self._pct = max(0.0, min(100.0, float(pct)))
        self._draw()

    def _click(self, _e):
        if self._enabled and self._cmd is not None:
            self._cmd()

    def _enter(self, _e):
        self._hover = True
        self._draw()

    def _leave(self, _e):
        self._hover = False
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = max(1, self.winfo_width()), self.H
        busy = self._pct is not None
        if busy:
            self.create_rectangle(0, 0, w, h, fill=CARD2, outline="")
            if self._pct > 0:
                self.create_rectangle(0, 0, w * self._pct / 100.0, h, fill=ACCENT, outline="")
            label = f"{int(round(self._pct))}%"
        else:
            self.create_rectangle(0, 0, w, h,
                                  fill="#6aa1ff" if self._hover else ACCENT, outline="")
            label = self._text
        self.create_text(w / 2.0, h / 2.0, text=label, fill="#ffffff", font=self._font)


class DepthUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Карта глубины")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d" % (max(1120, min(1360, sw - 80)),
                                 max(700, min(900, sh - 90))))
        self.minsize(1120, 700)
        self.configure(bg=BG)
        self._init_style()

        self.vars = {}

        outer = ttk.Frame(self, style="Root.TFrame", padding=(16, 14, 16, 12))
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        # --- шапка ---
        head = ttk.Frame(outer, style="Root.TFrame")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(head, text="Карта глубины", style="Title.TLabel").pack(side="left")
        ttk.Label(head, text="глубина → PNG · TIFF · EXR · PLY · GLB",
                  style="Muted.TLabel").pack(side="left", padx=(12, 0), pady=(7, 0))

        body = ttk.Frame(outer, style="Root.TFrame")
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0, minsize=440)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Левая колонка прокручивается: на ноутбуке с низким экраном настройки
        # не обрезаются, а «Результаты» остаются доступными
        side_wrap = ttk.Frame(body, style="Root.TFrame")
        side_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        side_canvas = tk.Canvas(side_wrap, bg=CARD, highlightthickness=0, bd=0)
        side_bar = ttk.Scrollbar(side_wrap, orient="vertical", command=side_canvas.yview,
                                 style="Card.Vertical.TScrollbar")
        side_canvas.configure(yscrollcommand=side_bar.set)
        side_canvas.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(side_canvas, style="Card.TFrame", padding=(16, 14))
        _side_id = side_canvas.create_window((0, 0), window=side, anchor="nw")

        def _side_scroll(_e=None):
            side_canvas.configure(scrollregion=side_canvas.bbox("all"))
            need = side.winfo_reqheight() > side_canvas.winfo_height() + 1
            if need and not side_bar.winfo_ismapped():
                side_bar.pack(side="right", fill="y")
            elif not need and side_bar.winfo_ismapped():
                side_bar.pack_forget()

        side.bind("<Configure>", _side_scroll)

        def _fit_side():
            # ширина окна в canvas ставится только при изменении значения,
            # а на первом показе она «не прилипала»: колонка оставалась шириной
            # по содержимому и правый край настроек обрезался. Сброс в 0
            # (по размеру содержимого) и повторная установка чинят это.
            side_canvas.itemconfigure(_side_id, width=0)
            side_canvas.itemconfigure(_side_id, width=max(1, side_canvas.winfo_width()))
            _side_scroll()

        side_canvas.bind("<Configure>", lambda _e: side_canvas.after_idle(_fit_side))

        def _wheel(e):
            step = -1 if (getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0) else 1
            side_canvas.yview_scroll(step, "units")

        def _bind_wheel(w):
            # у ползунков своё колесо (меняет значение), его не перебиваем
            if not isinstance(w, NiceSlider):
                for seq in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
                    w.bind(seq, _wheel)
            for child in w.winfo_children():
                _bind_wheel(child)

        view = ttk.Frame(body, style="Card.TFrame", padding=(12, 10))
        view.grid(row=0, column=1, sticky="nsew")

        def section(text, first=False):
            ttk.Label(side, text=text, style="Section.TLabel").pack(
                anchor="w", pady=(0 if first else 16, 7))

        # --- фото ---
        # «Открыть папку» стоит рядом с фото: внизу у кнопки «Применить» она
        # выглядела непонятно — что именно открывается
        head = ttk.Frame(side, style="Card.TFrame")
        head.pack(fill="x", pady=(0, 7))
        ttk.Label(head, text="ФОТО", style="Section.TLabel").pack(side="left")
        self.btn_folder = ttk.Button(head, text="Открыть папку", command=self.open_folder)
        self.btn_folder.pack(side="right")

        row = ttk.Frame(side, style="Card.TFrame")
        row.pack(fill="x")
        self.var_src = tk.StringVar(value=SRC)
        ttk.Entry(row, textvariable=self.var_src).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Выбрать", command=self.pick_src).pack(side="left", padx=(8, 0))

        # --- модель ---
        section("МОДЕЛЬ")
        _m = str(cfg.get("model", "small")).lower()
        if _m not in ("small", "base", "large", "midas", "zoe",
                      "base_in", "base_out", "large_in", "large_out", "large_mix",
                      "depthpro"):
            _m = "small"
        _mdisp = {"small": "Small", "base": "Base", "large": "Large", "midas": "MiDaS",
                  "zoe": "ZoeDepth", "base_in": "Base Indoor", "base_out": "Base Outdoor",
                  "large_in": "Large Indoor", "large_out": "Large Outdoor",
                  # large_mix в этом окне не выбирается, но старый settings.json
                  # мог его сохранить — иначе KeyError при запуске
                  "large_mix": "Large", "depthpro": "Depth Pro"}
        self.vars["model"] = tk.StringVar(value=_mdisp[_m])
        ttk.Combobox(side, textvariable=self.vars["model"],
                     values=["Small", "Base", "Large", "MiDaS", "ZoeDepth",
                             "Base Indoor", "Base Outdoor", "Large Indoor", "Large Outdoor",
                             "Depth Pro"],
                     state="readonly").pack(fill="x")
        ttk.Label(side, text="Base и Large детальнее, но медленнее",
                  style="Hint.TLabel").pack(anchor="w", pady=(5, 0))
        ttk.Label(side, text="Depth Pro точнее и в метрах, но ~минута на фото",
                  style="Hint.TLabel").pack(anchor="w", pady=(2, 0))

        # --- глубина ---
        section("ГЛУБИНА")

        def slider(label, key, lo, hi, fmt):
            row = ttk.Frame(side, style="Card.TFrame")
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, style="Card.TLabel", width=16,
                      anchor="w").pack(side="left")
            var = tk.DoubleVar(value=float(cfg[key]))
            val = ttk.Label(row, text=fmt(var.get()), style="Card.TLabel", width=6, anchor="e")
            val.pack(side="right")
            NiceSlider(row, var, lo, hi,
                       command=lambda _v: val.configure(text=fmt(var.get()))).pack(
                side="left", fill="x", expand=True, padx=(10, 8))
            self.vars[key] = var

        slider("Глубина от %", "depth_lo", 0, 90, lambda v: f"{v:.0f}")
        slider("Глубина до %", "depth_hi", 10, 100, lambda v: f"{v:.0f}")
        slider("Гамма", "depth_gamma", 0.2, 5.0, lambda v: f"{v:.2f}")
        slider("Контраст", "depth_contrast", 0.0, 4.0, lambda v: f"{v:.2f}")
        slider("Сглаживание", "guided", 0, 100, lambda v: f"{v:.0f}")
        slider("Шумодав", "denoise", 0, 100, lambda v: f"{v:.0f}")

        # --- разрешение ---
        section("РАЗРЕШЕНИЕ")
        row = ttk.Frame(side, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="Множитель входа", style="Card.TLabel", width=16,
                  anchor="w").pack(side="left")
        self.vars["input_mult"] = tk.DoubleVar(value=float(cfg.get("input_mult", 6)))
        self._input_lbl = ttk.Label(row, text="", style="Card.TLabel", width=4, anchor="e")
        self._input_lbl.pack(side="right")
        self._input_scale = NiceSlider(row, self.vars["input_mult"], 1, 10,
                                       command=lambda _v: self._update_input_hint())
        self._input_scale.pack(side="left", fill="x", expand=True, padx=(10, 8))
        self._ram_hint = ttk.Label(side, text="", style="Hint.TLabel")
        self._ram_hint.pack(anchor="w", pady=(5, 0))
        self._update_input_hint()

        row = ttk.Frame(side, style="Card.TFrame")
        row.pack(fill="x", pady=(10, 0))
        ttk.Label(row, text="Сжать оригинал", style="Card.TLabel").pack(side="left")
        _sd = str(cfg.get("src_div", "1x"))
        if _sd not in ("1x", "2x", "3x", "4x"):
            _sd = "1x"
        self.vars["src_max"] = tk.StringVar(value=_sd)
        ttk.Combobox(row, textvariable=self.vars["src_max"], values=["1x", "2x", "3x", "4x"],
                     width=5, state="readonly").pack(side="left", padx=(8, 18))
        ttk.Label(row, text="Выход", style="Card.TLabel").pack(side="left")
        self.vars["out_mult"] = tk.StringVar(value=str(cfg.get("out_mult", "1x")))
        ttk.Combobox(row, textvariable=self.vars["out_mult"], values=["1x", "2x", "3x", "4x"],
                     width=5, state="readonly").pack(side="left", padx=(8, 0))

        # --- результаты ---
        section("РЕЗУЛЬТАТЫ")
        grid = ttk.Frame(side, style="Card.TFrame")
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        def check(text, key, r, c):
            var = tk.BooleanVar(value=bool(cfg.get(key, False)))
            NiceCheck(grid, text, var).grid(row=r, column=c, sticky="w", pady=3, padx=(0, 6))
            self.vars[key] = var

        check("16-бит PNG", "depth16", 0, 0)
        check("32-бит TIFF", "depth32", 0, 1)
        check("Глубина EXR", "depth_exr", 1, 0)
        check("Цветная карта EXR", "cmap_exr", 1, 1)
        check("Нормали EXR", "normal_exr", 2, 0)
        check("XYZ-облако PLY", "ply_out", 2, 1)
        check("Модель GLB", "glb_out", 3, 0)
        check("Инверсия", "invert", 3, 1)
        check("Гибрид (2 прохода)", "render2", 4, 0)

        # --- нижняя панель: кнопка во всю ширину ---
        foot = ttk.Frame(outer, style="Root.TFrame")
        foot.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self.btn_run = ApplyButton(foot, font=self._f_btn, command=self.run)
        self.btn_run.pack(fill="x")

        # --- предпросмотр ---
        self.TYPE_NAMES = ["Глубина", "Цветная карта", "Нормаль", "Цветное фото"]
        self.type_idx = 0

        top = ttk.Frame(view, style="Card.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text="ПРЕДПРОСМОТР", style="Section.TLabel").pack(side="left")
        self.lbl_size = ttk.Label(top, text="—", style="Hint.TLabel")
        self.lbl_size.pack(side="right")

        bar = ttk.Frame(view, style="Card.TFrame")
        bar.pack(fill="x", pady=(9, 9))
        self.var_show_prev = tk.BooleanVar(value=False)
        NiceCheck(bar, "Предыдущий", self.var_show_prev).pack(side="left", padx=(0, 10))
        self.var_show_prev.trace_add("write", lambda *_: self.show_preview())
        seg = ttk.Frame(bar, style="Card.TFrame")
        seg.pack(side="left")
        self._seg_btns = []
        for i, name in enumerate(self.TYPE_NAMES):
            b = ttk.Button(seg, text=name, style="Seg.TButton",
                           command=lambda i=i: self.set_type(i))
            b.pack(side="left", padx=(0, 4))
            self._seg_btns.append(b)
        holder = tk.Frame(view, bg=BORDER)
        holder.pack(fill="both", expand=True)
        self.cur_view = tk.Label(holder, bg="#0d0f13", fg=MUTED, text="Обработайте фото", bd=0)
        self.cur_view.pack(fill="both", expand=True, padx=1, pady=1)
        self._preview_holder = holder
        self._preview_card = view
        for w in (holder, self.cur_view):
            for seq in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
                w.bind(seq, self._wheel_type)
        # запасной путь: колесо над предпросмотром листает типы, даже если
        # событие пришло не в сам label (после показа фото он остаётся тем же,
        # но у части окон/драйверов колесо уходит выше)
        for seq in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
            self.bind(seq, self._wheel_any, add="+")


        self._prev_img = None
        self._cur_img = None
        self._prev_col = None
        self._cur_col = None
        self._prev_nrm = None
        self._cur_nrm = None
        self._src_img = None
        self._prev_src = None

        # «Цветная карта» и «Нормаль» нужны, только когда их файлы включены
        # в «Результатах»: иначе предпросмотр предлагал типы, которых не будет
        for _k in ("cmap_exr", "normal_exr"):
            self.vars[_k].trace_add("write", lambda *_: self._refresh_types())
        self._refresh_types()
        _bind_wheel(side)
        self._ui_q = queue.Queue()
        self.after(60, self._poll_ui)
        self.bind("<Configure>", lambda _e: self._schedule_resize())

    def _ui(self, fn, *args):
        """Выполнить fn(*args) в главном потоке: с Python 3.14 Tkinter больше
        не принимает after() из рабочего потока, поэтому обновления идут через очередь."""
        self._ui_q.put((fn, args))

    def _poll_ui(self):
        try:
            while True:
                fn, args = self._ui_q.get_nowait()
                fn(*args)
        except queue.Empty:
            pass
        self.after(60, self._poll_ui)

    # ---------- оформление ----------
    def _init_style(self):
        import tkinter.font as tkfont
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass

        base = tkfont.nametofont("TkDefaultFont")
        base.configure(size=10)
        self._f_title = base.copy()
        self._f_title.configure(size=17, weight="bold")
        self._f_section = base.copy()
        self._f_section.configure(size=8, weight="bold")
        self._f_hint = base.copy()
        self._f_hint.configure(size=9)
        self._f_btn = base.copy()
        self._f_btn.configure(weight="bold")

        st.configure(".", background=BG, foreground=FG)
        st.configure("Root.TFrame", background=BG)
        st.configure("Card.TFrame", background=CARD)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Card.TLabel", background=CARD, foreground=FG)
        st.configure("Muted.TLabel", background=BG, foreground=MUTED)
        st.configure("Hint.TLabel", background=CARD, foreground=MUTED, font=self._f_hint)
        st.configure("Warn.TLabel", background=CARD, foreground="#ffa06b", font=self._f_hint)
        st.configure("Title.TLabel", background=BG, foreground="#ffffff", font=self._f_title)
        st.configure("Section.TLabel", background=CARD, foreground="#7f8794",
                     font=self._f_section)

        st.configure("TButton", background=CARD2, foreground=FG, bordercolor=BORDER,
                     lightcolor=CARD2, darkcolor=CARD2, focuscolor=CARD2, padding=(12, 6))
        st.map("TButton", background=[("active", "#333a46"), ("pressed", "#333a46")],
               foreground=[("disabled", "#6b7280")])

        st.configure("Accent.TButton", background=ACCENT, foreground="#ffffff",
                     bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
                     focuscolor=ACCENT, padding=(22, 8), font=self._f_btn)
        st.map("Accent.TButton",
               background=[("active", "#6aa1ff"), ("pressed", "#3f7ae0"),
                           ("disabled", "#2a3a57")],
               foreground=[("disabled", "#8f9bb0")])

        st.configure("Seg.TButton", background=CARD2, foreground=MUTED, bordercolor=CARD2,
                     lightcolor=CARD2, darkcolor=CARD2, focuscolor=CARD2, padding=(8, 5))
        st.map("Seg.TButton", background=[("active", "#333a46")],
               foreground=[("active", FG)])

        st.configure("SegOn.TButton", background=ACCENT, foreground="#ffffff",
                     bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
                     focuscolor=ACCENT, padding=(10, 5))
        st.map("SegOn.TButton", background=[("active", "#6aa1ff")],
               foreground=[("active", "#ffffff")])

        st.configure("TCombobox", fieldbackground=CARD2, background=CARD2, foreground=FG,
                     arrowcolor=MUTED, bordercolor=BORDER, lightcolor=BORDER,
                     darkcolor=BORDER, padding=(8, 5))
        st.map("TCombobox", fieldbackground=[("readonly", CARD2)],
               foreground=[("readonly", FG)], arrowcolor=[("active", FG)],
               bordercolor=[("focus", ACCENT)])

        st.configure("TEntry", fieldbackground=CARD2, foreground=FG, insertcolor=FG,
                     bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, padding=(8, 6))
        st.map("TEntry", bordercolor=[("focus", ACCENT)])

        self.option_add("*TCombobox*Listbox.background", CARD2)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        st.configure("Card.Vertical.TScrollbar", background="#3a414d", troughcolor=CARD,
                     bordercolor=CARD, arrowcolor=MUTED, lightcolor="#3a414d",
                     darkcolor="#3a414d", gripcount=0, arrowsize=12, width=11)
        st.map("Card.Vertical.TScrollbar",
               background=[("active", "#4a5260"), ("pressed", "#4a5260")])

        self.option_add("*TCombobox*Listbox.borderWidth", 0)

    # ---------- предпросмотр: тип ----------
    def set_type(self, i):
        self.type_idx = int(i) % len(self.TYPE_NAMES)
        self._set_seg_styles()
        self.show_preview()

    def _type_visible(self):
        """Видимость типов предпросмотра: «Глубина» и «Цветное фото» — всегда,
        «Цветная карта» и «Нормаль» — по галочкам «Цветная карта EXR» и
        «Нормали EXR»."""
        return [True,
                bool(self.vars["cmap_exr"].get()),
                bool(self.vars["normal_exr"].get()),
                True]

    def _refresh_types(self):
        vis = self._type_visible()
        if not vis[self.type_idx]:
            self.type_idx = vis.index(True)
        for b in self._seg_btns:
            b.pack_forget()
        for i, b in enumerate(self._seg_btns):
            if vis[i]:
                b.pack(side="left", padx=(0, 4))
        self._set_seg_styles()
        self.show_preview()

    def _step_type(self, step):
        """Следующий/предыдущий видимый тип — скрытые пропускаются."""
        vis = self._type_visible()
        n = len(vis)
        for k in range(1, n + 1):
            j = (self.type_idx + step * k) % n
            if vis[j]:
                self.set_type(j)
                return

    def _set_seg_styles(self):
        for i, b in enumerate(getattr(self, "_seg_btns", [])):
            b.configure(style="SegOn.TButton" if i == self.type_idx else "Seg.TButton")


    def pick_src(self):
        import subprocess
        start = self.var_src.get().strip()
        initdir = os.path.dirname(start)
        f = None
        if os.name == "nt":
            f = filedialog.askopenfilename(
                parent=self,
                title="Выберите фото",
                initialdir=initdir if os.path.isdir(initdir) else common.home_dir(),
                filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff *.exr"),
                           ("All Files", "*.*")],
            )
        else:
            cmd = ["zenity", "--file-selection", "--title=Выбор фото",
                   "--file-filter=Изображения | *.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff *.exr",
                   "--file-filter=Все файлы | *"]
            if start and os.path.isfile(start):
                cmd.append(f"--filename={start}")
            elif initdir and os.path.isdir(initdir):
                cmd.append(f"--filename={initdir}/")
            else:
                cmd.append(f"--filename={os.path.join(common.home_dir(), 'Pictures')}/")
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if r.returncode == 0:
                    f = r.stdout.strip()
            except FileNotFoundError:
                f = filedialog.askopenfilename(
                    parent=self,
                    title="Выберите фото",
                    initialdir=initdir if os.path.isdir(initdir) else common.home_dir(),
                    filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff *.exr"),
                               ("All Files", "*.*")],
                )
            except Exception:
                f = None
        if f:
            self.var_src.set(f)

    def browse_photos(self):
        win = tk.Toplevel(self)
        win.title("Поиск фото")
        win.geometry("900x600")
        win.configure(bg="#1e1e1e")

        folders = {
            "Downloads": os.path.join(common.home_dir(), "Downloads"),
            "Pictures": os.path.join(common.home_dir(), "Pictures"),
            "Desktop": os.path.join(common.home_dir(), "Desktop"),
            "Z-depth": common.data_dir(),
        }
        self._browse_cache = {}

        top = tk.Frame(win, bg="#1e1e1e")
        top.pack(fill="x", padx=8, pady=6)
        folder_names = list(folders.keys()) + ["Другая папка..."]
        var_fold = tk.StringVar(value="Pictures")
        ttk.Combobox(top, textvariable=var_fold, values=folder_names, width=16, state="readonly").pack(side="left")
        var_folder_lbl = tk.StringVar()
        ttk.Label(top, textvariable=var_folder_lbl, foreground="#aaa").pack(side="left", padx=(4, 0))
        var_search = tk.StringVar()
        ttk.Entry(top, textvariable=var_search).pack(side="left", fill="x", expand=True, padx=8)

        def toggle_maximize(_e=None):
            if win.state() == "zoomed":
                win.state("normal")
            else:
                win.state("zoomed")

        top.bind("<Double-Button-1>", toggle_maximize)

        canvas = tk.Canvas(win, bg="#1e1e1e", highlightthickness=0)
        scroll = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg="#1e1e1e")
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        self._inner_item = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scroll.pack(side="right", fill="y", pady=8)

        def on_wheel(e):
            canvas.yview_scroll(-1 * (e.delta // 120) if e.delta else (-1 if e.num == 4 else 1), "units")

        win.bind_all("<MouseWheel>", on_wheel)
        win.bind_all("<Button-4>", on_wheel)
        win.bind_all("<Button-5>", on_wheel)

        pan_state = {"y": 0}

        def mid_start(e):
            pan_state["y"] = e.y

        def mid_move(e):
            dy = e.y - pan_state["y"]
            pan_state["y"] = e.y
            canvas.yview_scroll(-dy, "pixels")

        canvas.bind("<ButtonPress-2>", mid_start)
        canvas.bind("<B2-Motion>", mid_move)
        inner.bind("<ButtonPress-2>", mid_start)
        inner.bind("<B2-Motion>", mid_move)

        tiles = []
        tile_refs = {}
        load_gen = [0]

        def close_browser():
            load_gen[0] = 0
            tile_refs.clear()
            win.unbind_all("<MouseWheel>")
            win.unbind_all("<Button-4>")
            win.unbind_all("<Button-5>")
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", close_browser)
        win.bind("<Escape>", lambda _e: close_browser())

        def relayout(*_a):
            cols = max(1, canvas.winfo_width() // 170)
            for i, tile in enumerate(tiles):
                tile.grid(row=i // cols, column=i % cols, padx=4, pady=4)

        canvas.bind("<Configure>", relayout)

        def load_thumbs(paths):
            import hashlib
            cdir = os.path.join(common.data_dir(), ".thumbs")
            try:
                os.makedirs(cdir, exist_ok=True)
            except Exception:
                cdir = None
            for path in paths:
                if load_gen[0] == 0 or path not in tile_refs:
                    return
                low = path.lower()
                im = None
                cp = None
                if cdir:
                    try:
                        key = hashlib.md5(f"{path}|{os.path.getmtime(path)}".encode()).hexdigest()[:16]
                        cp = os.path.join(cdir, key + ".png")
                        if os.path.exists(cp):
                            im = Image.open(cp).convert("RGB")
                    except Exception:
                        cp = im = None
                if im is None:
                    try:
                        im = Image.open(path)
                        if low.endswith((".jpg", ".jpeg")):
                            im.draft("RGB", (300, 300))
                        im.thumbnail((150, 150), Image.BILINEAR)
                        im = im.convert("RGB")
                    except Exception:
                        im = None
                    if im is None and low.endswith(".exr"):
                        im = _exr_thumb(path)
                    if im is None:
                        im = Image.new("RGB", (140, 100), (38, 38, 38))
                    elif cp:
                        try:
                            im.save(cp)
                        except Exception:
                            pass
                self.after(0, lambda p=path, im=im: self._set_thumb(p, im, tile_refs, load_gen))

        custom_folder = {"path": None}

        def refresh(*_a):
            load_gen[0] += 1
            gen = load_gen[0]
            for w in inner.winfo_children():
                w.destroy()
            tiles.clear()
            tile_refs.clear()
            chosen = var_fold.get()
            if chosen == "Другая папка...":
                folder = custom_folder["path"]
                if not folder or not os.path.isdir(folder):
                    import tkinter.filedialog as fd
                    folder = fd.askdirectory(initialdir=common.home_dir(), title="Выберите папку с фото")
                    if not folder:
                        var_fold.set("Pictures")
                        return
                    custom_folder["path"] = folder
                    var_folder_lbl.set(os.path.basename(folder))
            else:
                folder = folders[chosen]
            win.title(f"Поиск фото — {os.path.basename(folder)}")
            q = var_search.get().strip().lower()
            try:
                files = sorted(os.listdir(folder), reverse=True)
            except Exception:
                files = []
            exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".exr", ".tif", ".tiff")
            all_paths = []
            for name in files:
                if len(all_paths) >= 300:
                    break
                if not name.lower().endswith(exts):
                    continue
                if q and q not in name.lower():
                    continue
                all_paths.append(os.path.join(folder, name))

            def add_tile(path):
                name = os.path.basename(path)
                tile = ttk.Frame(inner, padding=4)
                lbl = tk.Label(tile, text="…", bg="#2a2a2a", fg="#666", cursor="hand2", width=20, height=10)
                lbl.pack()
                tname = name if len(name) <= 22 else name[:19] + "..."
                lbln = tk.Label(tile, text=tname, bg="#1e1e1e", fg="#ccc")
                lbln.pack()
                lbl.bind("<Button-1>", lambda _e, p=path: self._pick_photo(p, win))
                lbln.bind("<Button-1>", lambda _e, p=path: self._pick_photo(p, win))
                lbl.bind("<ButtonPress-2>", mid_start)
                lbl.bind("<B2-Motion>", mid_move)
                lbln.bind("<ButtonPress-2>", mid_start)
                lbln.bind("<B2-Motion>", mid_move)
                tiles.append(tile)
                tile_refs[path] = lbl

            def build_batch():
                nonlocal all_paths
                if load_gen[0] != gen:
                    return
                batch, all_paths = all_paths[:40], all_paths[40:]
                for path in batch:
                    add_tile(path)
                relayout()
                if all_paths:
                    win.after(1, build_batch)
                else:
                    threading.Thread(target=load_thumbs, args=(list(tile_refs),), daemon=True).start()

            win.after_idle(build_batch)

        var_fold.trace_add("write", refresh)
        var_search.trace_add("write", refresh)
        refresh()

    def _set_thumb(self, path, im, tile_refs, load_gen):
        if load_gen[0] == 0 or path not in tile_refs:
            return
        lbl = tile_refs[path]
        try:
            if not lbl.winfo_exists():
                return
            photo = ImageTk.PhotoImage(im)
        except Exception:
            return
        self._browse_cache[path] = photo
        try:
            lbl.configure(image=photo, text="", width=0, height=0)
        except tk.TclError:
            return
        lbl.image = photo

    def _pick_photo(self, path, win):
        self.var_src.set(path)
        try:
            win.master.unbind_all("<MouseWheel>")
            win.master.unbind_all("<Button-4>")
            win.master.unbind_all("<Button-5>")
        except Exception:
            pass
        win.destroy()
        self.var_src.set(path)

    def _update_input_hint(self):
        v = max(1, min(10, int(round(float(self.vars["input_mult"].get())))))
        # множитель целый: вход модели = 200 px × v, дробные значения только мешают
        if abs(float(self.vars["input_mult"].get()) - v) > 1e-6:
            self.vars["input_mult"].set(v)
        px = 200 * v
        self._input_lbl.configure(text=f"{v}×")
        # обычный режим — приглушённый текст, большой вход — предупреждение
        if v >= 8:
            self._ram_hint.configure(text=f"Осторожно: вход {px} px по длинной стороне — это много ОЗУ",
                                     style="Warn.TLabel")
        elif v >= 6:
            self._ram_hint.configure(text=f"Вход {px} px — обычный режим для фото 1000–2000 px",
                                     style="Hint.TLabel")
        elif v >= 3:
            self._ram_hint.configure(text=f"Вход {px} px — быстрый черновой режим",
                                     style="Hint.TLabel")
        else:
            self._ram_hint.configure(text=f"Вход {px} px — очень грубо, только чтобы прикинуть",
                                     style="Hint.TLabel")

    def collect(self):
        c = dict(cfg)
        c["model"] = {"small": "small", "base": "base", "large": "large",
                      "midas": "midas", "zoedepth": "zoe", "zoe": "zoe",
                      "base indoor": "base_in", "base outdoor": "base_out",
                      "large indoor": "large_in",
                      "large outdoor": "large_out",
                      "depth pro": "depthpro"}.get(self.vars["model"].get().lower(), "small")
        c["src"] = self.var_src.get()
        mult = max(1, int(round(float(self.vars["input_mult"].get()))))
        if not str(c["src"]).strip():
            raise ValueError("Не выбрано фото: укажите путь или нажмите «Поиск фото».")
        if not os.path.exists(str(c["src"])):
            raise ValueError(f"Файл не найден:\n{c['src']}")
        if str(c["src"]).lower().endswith(".exr"):
            src_img = _exr_load_rgb(c["src"])
            if src_img is None:
                raise ValueError(f"Не удалось прочитать EXR: {c['src']}")
        else:
            src_img = Image.open(c["src"]).convert("RGB")
        div = int("".join(ch for ch in str(self.vars["src_max"].get()) if ch.isdigit()) or 1)
        c["src_div"] = div
        if div > 1:
            src_img = src_img.resize((max(1, src_img.width // div), max(1, src_img.height // div)), Image.LANCZOS)
        orig_w, orig_h = src_img.width, src_img.height
        # вход модели: 200 px × множитель по длинной стороне (бывший «Эконом ОЗУ»);
        # выше 4x от оригинала не растягиваем — деталей это не добавит, только ОЗУ
        scale = min((200.0 * mult) / max(1, max(src_img.width, src_img.height)), 4.0)
        c["in_w"] = max(14, int(src_img.width * scale))
        c["in_h"] = max(14, int(src_img.height * scale))
        out_mult = int(self.vars["out_mult"].get().replace("x", ""))
        c["out_size"] = f"{out_mult * orig_w}x{out_mult * orig_h}"
        for k, var in self.vars.items():
            if isinstance(var, tk.DoubleVar):
                c[k] = float(var.get())
        c["invert"] = bool(self.vars["invert"].get())
        c["render2"] = bool(self.vars["render2"].get())
        c["depth16"] = bool(self.vars["depth16"].get())
        c["depth32"] = bool(self.vars["depth32"].get())
        c["depth_exr"] = bool(self.vars["depth_exr"].get())
        c["cmap_exr"] = bool(self.vars["cmap_exr"].get())
        c["normal_exr"] = bool(self.vars["normal_exr"].get())
        c["ply_out"] = bool(self.vars["ply_out"].get())
        c["glb_out"] = bool(self.vars["glb_out"].get())
        return c

    def _set_progress(self, pct, _text=None):
        self.btn_run.set_progress(pct)

    def run(self):
        if STATE["busy"]:
            return
        try:
            c = self.collect()
        except Exception as e:
            # файл не выбран/не читается — иначе ошибка уходила только в консоль
            messagebox.showerror("Ошибка", str(e))
            return
        if c["model"] == "depthpro" and not (depth_pro_modules_ok() and depth_pro_weights_ok()):
            if not messagebox.askyesno(
                    "Depth Pro",
                    "Для Depth Pro один раз нужно скачать PyTorch (~250 МБ) "
                    "и веса модели (~1,9 ГБ).\n\nПродолжить?"):
                return
        json.dump(c, open(CONFIG, "w"), indent=2)
        STATE["busy"] = True
        self.btn_run.set_busy(True)
        self._set_progress(0)
        threading.Thread(target=self.work, args=(c,), daemon=True).start()

    def work(self, c):
        # прогресс 0..100 на весь проход; у «Гибрида» два прохода делят шкалу
        base = float(c.get("_pct_base", 0.0))
        span = float(c.get("_pct_span", 100.0))
        dp_fov = None

        def stage(rel):
            self._ui(self._set_progress, base + span * float(rel) / 100.0)

        stage(2)
        try:
            gen = str(c.get("gen") or "none")
            if gen.startswith("shape:"):
                kind = gen.split(":", 1)[1]
                size = float(c.get("gen_amp", 1.0) or 1.0)
                freq = float(c.get("gen_freq", 2.0) or 2.0)
                verts, cols, faces = gen_mesh(kind, size=size, freq=freq, res=96)
                if c.get("ply_out"):
                    _write_mesh_ply(f"{OUT}/photo_points.ply", verts, cols, faces)
                if c.get("glb_out"):
                    _write_mesh_glb(f"{OUT}/photo_points.glb", verts, cols, faces)
                stage(100)
                return
            if c["model"] == "midas":
                model_paths = [common.find_model("midas_v21_small_256.onnx")]
                if not os.path.exists(model_paths[0]):
                    raise ValueError("MiDaS: нет файла midas_v21_small_256.onnx в папке Z-depth")
            elif c["model"] == "zoe":
                model_paths = [common.find_model("zoedepth_nk_fp16.onnx")]
                if not os.path.exists(model_paths[0]):
                    raise ValueError("ZoeDepth: нет файла zoedepth_nk_fp16.onnx в папке Z-depth")
            elif c["model"] == "depthpro":
                model_paths = [depth_pro_dir()]
                if not (depth_pro_modules_ok() and depth_pro_weights_ok()):
                    # первый выбор: torch/transformers и веса — с прогрессом в кнопке
                    # (отдельная широкая полоса 2–24 %, иначе загрузка на 3 минуты
                    # выглядела бы как «застряло на 5 %»)
                    ensure_depth_pro(lambda f: stage(2 + 22 * f))
                stage(24)
            elif c["model"] == "large_mix":
                labels = ("Large", "MiDaS", "Zoe")
                model_paths = [common.find_model("depth_anything_v2_large.onnx"),
                               common.find_model("midas_v21_small_256.onnx"),
                               common.find_model("zoedepth_nk_fp16.onnx")]
                for _lb, _p in zip(labels, model_paths):
                    if not os.path.exists(_p):
                        stage(4)
                        if _lb == "Large":
                            download_model("large", _p)
                        else:
                            raise ValueError(f"{_lb}: нет файла {_p}")
                stage(8)
            else:
                model_paths = [common.find_model(f"depth_anything_v2_{c['model']}.onnx")]
                if not os.path.exists(model_paths[0]):
                    stage(4)
                    download_model(c["model"], model_paths[0])
                    stage(8)
            gen = str(c.get("gen") or "none")
            if False and gen != "none":
                if str(c["src"]).lower().endswith(".exr"):
                    img = _exr_load_rgb(c["src"])
                    if img is None:
                        raise ValueError(f"Не удалось прочитать EXR: {c['src']}")
                else:
                    img = Image.open(c["src"]).convert("RGB")
                out_size = str(c["out_size"]).replace("х", "x").replace("Х", "x").replace(" ", "")
                try:
                    w0, h0 = [int(x) for x in out_size.split("x")]
                except ValueError:
                    w0, h0 = 512, 512
                if w0 < 2 or h0 < 2:
                    w0, h0 = 512, 512
                if gen == "photo":
                    g = np.asarray(img.convert("L").resize((w0, h0), Image.BICUBIC),
                                   dtype=np.float32) / 255.0
                    r = max(3, int(round(float(c.get("gen_freq", 2.0) or 2.0) * 6)))
                    v = _box_f(g * g, r) - _box_f(g, r) ** 2
                    dfull = _box_f(np.sqrt(np.clip(v, 0.0, None)), max(2, r // 2))
                    dfull = np.nan_to_num(np.asarray(dfull, dtype=np.float32), nan=0.0)
                elif gen == "shading":
                    g = np.asarray(img.convert("L").resize((w0, h0), Image.BICUBIC),
                                   dtype=np.float32) / 255.0
                    r = max(3, int(round(float(c.get("gen_freq", 2.0) or 2.0) * 3)))
                    s = _box_f(g, r)
                    grad = np.asarray(np.gradient(s), dtype=np.float32)
                    h = np.cumsum(grad[1], axis=1) * 0.5 + np.cumsum(grad[0], axis=0) * 0.5
                    dfull = s * 0.55 + (h - h.min()) / (h.max() - h.min() + 1e-8) * 0.45
                    dfull = np.nan_to_num(np.asarray(dfull, dtype=np.float32), nan=0.0)
                else:
                    dfull = np.asarray(gen_depth_map(gen, h0, w0,
                                                     float(c.get("gen_amp", 1.0) or 1.0),
                                                     float(c.get("gen_freq", 2.0) or 2.0)),
                                       dtype=np.float32)
                    k = float(c.get("fx_strength", 0.6) or 0.0)
                    if k > 0:
                        m = np.asarray(img.convert("L").resize((w0, h0), Image.BICUBIC),
                                       dtype=np.float32) / 255.0
                        dfull = dfull * ((1.0 - k) + k * m)
                img = img.convert("RGB").resize((w0, h0), Image.LANCZOS)
                d = None
            else:
                            if str(c["src"]).lower().endswith(".exr"):
                                img = _exr_load_rgb(c["src"])
                                if img is None:
                                    raise ValueError(f"Не удалось прочитать EXR: {c['src']}")
                            else:
                                img = Image.open(c["src"]).convert("RGB")
                            div = int("".join(ch for ch in str(c.get("src_div", 1)) if ch.isdigit()) or 1)
                            if div > 1:
                                img = img.resize((max(1, img.width // div), max(1, img.height // div)), Image.LANCZOS)
                            dn = float(c.get("denoise", 0) or 0)
                            if dn > 0:
                                arr = np.asarray(img, dtype=np.float32) / 255.0
                                r = int(1 + dn * 0.05)
                                eps = 10.0 ** (-4.0 + dn * 0.02)
                                arr = np.stack([guided_filter(_box_f(arr[..., k], r), arr[..., k], r, eps)
                                                for k in range(3)], axis=2)
                                img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
                            nw = int(c["in_w"]) - int(c["in_w"]) % 14
                            nh = int(c["in_h"]) - int(c["in_h"]) % 14
                            if c["model"] == "midas":
                                nw = nh = 256
                            elif c["model"] == "zoe":
                                nw, nh = 512, 384

                            # Depth Pro уже показала 2–24 % на установке/загрузке,
                            # поэтому общий шаг «10 %» для неё не повторяем
                            stage(10 if c["model"] != "depthpro" else 24)
                            ds = []
                            dp_fov = None
                            if c["model"] == "depthpro":
                                _dp_d, dp_fov = _infer_depth_pro(
                                    model_paths[0], img, nw, nh,
                                    on_tick=lambda f: stage(25 + 23 * f))
                                ds.append(_dp_d)
                            elif c["model"] == "large_mix":
                                sizes = ((nw, nh), (256, 256), (512, 384))
                                for _n, _mp, (_nw, _nh) in zip(
                                        ("large", "midas", "zoe"), model_paths, sizes):
                                    _di = _infer_depth(_mp, img, _nw, _nh, metric=_n == "zoe",
                                                       on_tick=lambda f: stage(10 + 38 * f))
                                    if _di.shape != (nh, nw):
                                        _di = np.asarray(Image.fromarray(_di).resize(
                                            (nw, nh), Image.BICUBIC), dtype=np.float32)
                                    ds.append(_di)
                            else:
                                for _mp in model_paths:
                                    ds.append(_infer_depth(_mp, img, nw, nh,
                                                           metric=c["model"] in METRIC_MODELS,
                                                           on_tick=lambda f: stage(10 + 38 * f)))
                            if c["model"] == "large_mix":
                                self.__dict__["_mix_ds"] = [np.asarray(_d, dtype=np.float32) for _d in ds]
                            d = np.mean(ds, axis=0)
                            dfull = np.asarray(Image.fromarray(d).resize(img.size, Image.BICUBIC), dtype=np.float32)
            stage(50)
            g_strength = float(c.get("guided", 0) or 0)
            if g_strength > 0:
                guide = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
                # Сглаживание: сила = и радиус окна, и eps. Если увеличивать только радиус
                # (eps фиксированный), фильтр начинает тянуть текстуру фото в карту глубины —
                # чем больше ползунок, тем "шумнее" результат (сглаживание работало наоборот).
                radius = 2.0 + g_strength * 0.15
                eps = 1e-4 * (1.0 + 10.0 * g_strength)
                dfull = guided_filter(guide, dfull, radius, eps)
                mn, mx = float(dfull.min()), float(dfull.max())
                dfull = (dfull - mn) / (mx - mn + 1e-8)
            dfloat = np.clip(dfull, 0.0, 1.0)
            mn, mx = float(dfloat.min()), float(dfloat.max())
            dfloat = (dfloat - mn) / (mx - mn + 1e-8)
            if c["depth_contrast"] != 1.0:
                dmean = float(dfloat.mean())
                dfloat = np.clip((dfloat - dmean) * float(c["depth_contrast"]) + dmean, 0.0, 1.0)
            if c["invert"]:
                dfloat = 1.0 - dfloat
            # «Глубина от/до %» — это проценты расстояния от камеры: 0% = ближайшее,
            # 100% = самое далёкое. В карте глубины наоборот: 1.0 = близко, 0.0 = далеко,
            # поэтому границы зеркалятся, иначе «от» и «до» работают наоборот.
            lo = float(c.get("depth_lo", 0.0) or 0.0) / 100.0
            hi = float(c.get("depth_hi", 100.0) or 100.0) / 100.0
            if hi > lo + 1e-6:
                dfloat = np.clip((dfloat - (1.0 - hi)) / (hi - lo), 0.0, 1.0)
            g = float(c.get("depth_gamma", 1.0) or 1.0)
            if g != 1.0:
                dfloat = np.clip(dfloat, 0.0, 1.0) ** (1.0 / g)

            out_size = str(c["out_size"]).replace("х", "x").replace("Х", "x").replace(" ", "")
            w, h = [int(x) for x in out_size.split("x")]
            dfloat = np.asarray(Image.fromarray(dfloat).resize((w, h), Image.LANCZOS), dtype=np.float32)
            stage(62)

            hp = self.__dict__.get("_hybrid_prev")
            if hp is not None and c.get("hybrid"):
                # поход 3: глубина = первый + второй рендеры вместе
                a1 = hp["a"]
                if a1.shape != dfloat.shape:
                    a1 = np.asarray(Image.fromarray(a1).resize((w, h), Image.LANCZOS), dtype=np.float32)
                self.__dict__["_hyb_d2"] = np.clip(dfloat, 0.0, 1.0).astype(np.float32)
                dfloat = np.clip(dfloat * 0.5 + np.clip(a1, 0.0, 1.0) * 0.5, 0.0, 1.0)

            depth_out = Image.fromarray((np.clip(dfloat, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB")
            img_out = img.convert("RGB").resize((w, h), Image.LANCZOS)

            scale_ratio = max(w, h) / max(1, max(int(c["in_w"]) - int(c["in_w"]) % 14, int(c["in_h"]) - int(c["in_h"]) % 14))
            smooth = min(3.0, max(0.0, (scale_ratio - 1) * 0.35))
            if smooth > 0:
                depth_out = depth_out.filter(ImageFilter.GaussianBlur(smooth))
                dfloat = _blur_f(dfloat, smooth)

            if c.get("depth_exr"):
                # бело-чёрная глубина 0..1 без гаммы — то, что читают ноды в Blender
                write_exr(f"{OUT}/photo_depth.exr", np.clip(dfloat, 0.0, 1.0).astype(np.float32))
            # исходное фото всегда сохраняем в EXR (float linear RGB) — для цвета меша
            write_exr(f"{OUT}/photo_src.exr",
                      _srgb_to_linear(np.asarray(img_out, dtype=np.float32) / 255.0))
            if c["depth16"]:
                Image.fromarray((np.clip(dfloat, 0.0, 1.0) * 65535).astype(np.uint16)).save(f"{OUT}/photo_depth_16.png")
            if c.get("depth32"):
                Image.fromarray(np.clip(dfloat, 0.0, 1.0)).save(f"{OUT}/photo_depth_32.tif")
            stage(68)

            crgb = colormap_rgb(np.clip(dfloat, 0.0, 1.0))
            _mix_ds = self.__dict__.get("_mix_ds")
            if _mix_ds:
                for _tag, _di in zip(("large", "midas", "zoe"), _mix_ds):
                    di = np.asarray(Image.fromarray(_di).resize((w, h), Image.LANCZOS), dtype=np.float32)
                    mn, mx = float(di.min()), float(di.max())
                    di = np.clip((di - mn) / (mx - mn + 1e-8), 0.0, 1.0)
                    Image.fromarray((di * 255).astype(np.uint8)).convert("RGB").save(f"{OUT}/photo_depth_{_tag}.png")
                    cdi = colormap_rgb(di)
                    write_exr(f"{OUT}/photo_colormap_{_tag}.exr", np.concatenate([
                        _srgb_to_linear(cdi).astype(np.float32),
                        di[..., None].astype(np.float32)], axis=-1))
                    Image.fromarray((cdi * 255.0).round().astype(np.uint8)).save(f"{OUT}/photo_colormap_{_tag}.png")
                self.__dict__["_mix_ds"] = None
            hp = self.__dict__.get("_hybrid_prev")
            if hp is not None and c.get("hybrid") and hp.get("rgb") is not None:
                # второй рендер не пропадает: сохраняем его отдельно
                d2 = self.__dict__.pop("_hyb_d2", None)
                if d2 is None:
                    d2 = np.clip(dfloat, 0.0, 1.0).astype(np.float32)
                try:
                    write_exr(f"{OUT}/photo_colormap_2.exr", np.concatenate([
                        _srgb_to_linear(crgb).astype(np.float32),
                        d2[..., None].astype(np.float32)], axis=-1))
                    Image.fromarray((crgb * 255.0).round().astype(np.uint8)).save(f"{OUT}/photo_colormap_2.png")
                    Image.fromarray((d2 * 255).astype(np.uint8)).convert("RGB").save(
                        f"{OUT}/photo_depth_2.png")
                except Exception:
                    pass
                rgb1 = hp["rgb"]
                if rgb1.shape[:2] != crgb.shape[:2]:
                    rgb1 = np.asarray(Image.fromarray(
                        (rgb1 * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS),
                        dtype=np.float32) / 255.0
                # гибрид: цвета первого + второго рендера вместе
                crgb = np.clip(crgb * 0.5 + rgb1 * 0.5, 0.0, 1.0)
                self._hybrid_prev = None
            Image.fromarray((crgb * 255.0).round().astype(np.uint8)).save(f"{OUT}/photo_colormap.png")
            if c["depth16"]:
                png_write_rgb16(f"{OUT}/photo_colormap_16.png", crgb)
            if c.get("cmap_exr"):
                # цвет конвертируем sRGB -> linear, чтобы EXR выглядел как PNG;
                # альфа (глубина) остаётся сырой 0-1 для нод
                rgba = np.concatenate([
                    _srgb_to_linear(crgb).astype(np.float32),
                    np.clip(dfloat, 0.0, 1.0)[..., None].astype(np.float32)], axis=-1)
                write_exr(f"{OUT}/photo_colormap.exr", rgba)

            # нормали в tangent space: X=R, Y=G, Z=B (плоскость лицом к камере = +Z=синий).
            # Считаем один раз: этот же массив идёт в предпросмотр «Нормаль»,
            # поэтому файл и превью всегда совпадают
            _nrm = normal_map(dfloat) if c.get("normal_exr") else None
            if _nrm is not None:
                write_exr(f"{OUT}/photo_normal.exr", _nrm)
            stage(80)

            if c.get("ply_out") or c.get("glb_out"):
                try:
                    if c.get("ply_out"):
                        write_ply(f"{OUT}/photo_points.ply", dfloat, img_out,
                                  fov_y=dp_fov or DEFAULT_FOV_Y,
                                  on_progress=lambda f: stage(82 + 12 * f))
                    if c.get("glb_out"):
                        stage(96)
                        write_glb(f"{OUT}/photo_points.glb", dfloat, img_out,
                                  fov_y=dp_fov or DEFAULT_FOV_Y)
                        stage(98)
                except Exception as e:
                    print("ply/glb error:", e)

            if c.get("render2") and os.path.realpath(str(c["src"])) != os.path.realpath(f"{OUT}/photo_colormap.exr"):
                # первый рендер не пропадает: сохраняем его отдельно
                prev_exr = f"{OUT}/photo_colormap_1.exr"
                try:
                    write_exr(prev_exr, np.concatenate([
                        _srgb_to_linear(crgb).astype(np.float32),
                        np.clip(dfloat, 0.0, 1.0)[..., None].astype(np.float32)], axis=-1))
                    Image.fromarray((crgb * 255.0).round().astype(np.uint8)).save(f"{OUT}/photo_colormap_1.png")
                    Image.fromarray((np.clip(dfloat, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB").save(
                        f"{OUT}/photo_depth_1.png")
                except Exception as e:
                    print("render1 save error:", e)
                if not os.path.exists(prev_exr):
                    # без этой карты второй проход не стартует, поэтому пишем повторно
                    write_exr(prev_exr, np.concatenate([
                        _srgb_to_linear(crgb).astype(np.float32),
                        np.clip(dfloat, 0.0, 1.0)[..., None].astype(np.float32)], axis=-1))
                # второй проход: карта первого рендера сама становится источником;
                # результат гибридизируется с первым (50/50).
                # отдельный файл, чтобы «Гибрид» не требовал галочку «Цветная карта (EXR)»
                self._hybrid_prev = {
                    "a": np.clip(dfloat, 0.0, 1.0).astype(np.float32),
                    "rgb": np.clip(crgb, 0.0, 1.0).astype(np.float32),
                    "size": (w, h),
                }
                c2 = dict(c)
                c2["src"] = prev_exr
                c2["render2"] = False
                c2["hybrid"] = True
                c2["_pct_base"] = 48
                c2["_pct_span"] = 52
                stage(48)
                return self.work(c2)


            depth_p = depth_out.copy()
            col_p = Image.fromarray((crgb * 255.0).round().astype(np.uint8))
            if _nrm is None:
                _nrm = normal_map(dfloat)
            nrm_p = Image.fromarray((_nrm * 255.0).astype(np.uint8))
            src_p = img_out.copy()

            self._ui(self.done, depth_p.copy(), col_p.copy(), nrm_p.copy(), src_p.copy())
        except Exception as e:
            msg = str(e)
            import traceback
            with open(os.path.join(common.data_dir(), "ui.log"), "a") as f:
                f.write(traceback.format_exc() + "\n")
            self._ui(self.fail, msg)

    def cycle_type(self):
        self._step_type(1)

    def _wheel_type(self, e):
        up = getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0
        self._step_type(-1 if up else 1)
        return "break"

    def _wheel_any(self, e):
        # вся карточка предпросмотра — зона колеса: картинка, кнопки типов и
        # галочка «Предыдущий»; иначе над ними колесо просто «не реагировало»
        h = self.__dict__.get("_preview_card")
        if h is None or not h.winfo_ismapped():
            return None
        x, y = getattr(e, "x_root", 0), getattr(e, "y_root", 0)
        if not (h.winfo_rootx() <= x < h.winfo_rootx() + h.winfo_width()
                and h.winfo_rooty() <= y < h.winfo_rooty() + h.winfo_height()):
            return None
        return self._wheel_type(e)

    def _schedule_resize(self):
        if hasattr(self, "_resize_after"):
            self.after_cancel(self._resize_after)
        self._resize_after = self.after(80, self._on_resized)

    def _on_resized(self):
        if hasattr(self, "_resize_after"):
            del self._resize_after
        if self._cur_img is not None:
            self.show_preview()

    def _all_images(self):
        """Текущая и предыдущая картинка выбранного типа."""
        if self._cur_img is None:
            return None, None
        if self.type_idx == 0:
            return self._cur_img, self._prev_img
        if self.type_idx == 1:
            return self._cur_col, self._prev_col
        if self.type_idx == 2:
            return self._cur_nrm, self._prev_nrm
        return self._src_img, self._prev_src

    def done(self, depth, col, nrm, src):
        STATE["busy"] = False
        self.btn_run.set_busy(False)
        if self._cur_img is not None:
            self._prev_img = self._cur_img
            self._prev_col = self._cur_col
            self._prev_nrm = self._cur_nrm
            self._prev_src = self._src_img
        self._cur_img = depth.copy()
        self._cur_col = col.copy()
        self._cur_nrm = nrm.copy()
        self._src_img = src.copy()
        self.show_preview()

    def show_preview(self):
        cur, prev = self._all_images()
        if self.var_show_prev.get() and prev is not None:
            cur = prev
        self.lbl_size.configure(text=f"{cur.width}×{cur.height}" if cur else "—")
        self._set_view(self.cur_view, cur, "Обработайте фото")

    def _set_view(self, widget, img, placeholder):
        if img is not None:
            scale = 1.0
            pw, ph = widget.winfo_width(), widget.winfo_height()
            if pw > 10 and ph > 10:
                scale = min(pw / img.width, ph / img.height)
            w, h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
            photo = ImageTk.PhotoImage(img.convert("RGB").resize((w, h), Image.LANCZOS))
            widget._photo = photo
            widget.configure(image=photo, text="")
        else:
            widget.configure(image="", text=placeholder)

    def fail(self, msg):
        STATE["busy"] = False
        self.btn_run.set_busy(False)
        messagebox.showerror("Ошибка", msg)

    def open_folder(self):
        open_path(OUT)


if __name__ == "__main__":
    app = DepthUI()
    app.mainloop()
