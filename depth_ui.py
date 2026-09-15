import json
import math
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageTk, ImageDraw
import onnxruntime as ort

import common

CONFIG = os.path.join(common.data_dir(), "settings.json")
OUT = common.data_dir()
SRC = common.default_src()

DEFAULTS = {
    "model": "small",
    "depth_gamma": 1.0,
    "depth_lo": 0.0,
    "depth_hi": 100.0,
    "depth_contrast": 1.0,
    "overlay_alpha": 0.45,
    "blur_strength": 6.0,
    "out_size": "300x300",
    "invert": False,
    "render2": False,
    "dof_near": False,
    "compress": False,
    "focus_enable": False,
    "focus_x": -1,
    "focus_y": -1,
    "focus_width": 25.0,
    "depth16": False,
    "depth32": False,
    "depth_exr": False,
    "cmap_exr": False,
    "normal_png": False,
    "ply_out": False,
    "glb_out": False,
    "ply_fov": 90,
    "ply_scale": 1.0,
    "gen": "none",
    "gen_amp": 1.0,
    "gen_freq": 2.0,
    "src_div": "1x",
    "guided": 45.0,
    "denoise": 0.0,
}

cfg = DEFAULTS.copy()
if os.path.exists(CONFIG):
    try:
        cfg.update(json.load(open(CONFIG)))
    except Exception:
        pass

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
        (0.0, (0.0, 0.0, 0.5)),
        (0.2, (0.0, 0.5, 1.0)),
        (0.4, (0.0, 1.0, 1.0)),
        (0.6, (0.5, 1.0, 0.0)),
        (0.8, (1.0, 1.0, 0.0)),
        (1.0, (1.0, 0.0, 0.0)),
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
        chans += nm.encode() + b"\0" + _s.pack("<i", 1) + _s.pack("<B", 0) + b"\0\0\0" + _s.pack("<ii", 1, 1)
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
    planes16 = [p.astype(np.float16) for p in planes]
    npl = len(planes)
    offsets = []
    pos = len(hdr) + 8 * h
    chunks = []
    # стандартный порядок: чанк y = строка y (сверху вниз), lineOrder=INCREASING_Y
    for y in range(h):
        offsets.append(pos)
        data = b"".join(p[y].tobytes() for p in planes16)
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
        if types and any(t != 1 for t in types):
            return None  # только half-float
        dec = attrs.get("lineOrder", ("", b"\x00"))[1][0] == 1
        offs = _st.unpack("<" + "Q"*h, d[pos:pos+8*h])
        planes = {}
        for ci, nm in enumerate(chans):
            arr = np.empty((h, w), dtype=np.float32)
            for y in range(h):
                o = offs[y] + 8 + ci*w*2
                row = np.frombuffer(d[o:o+w*2], dtype="<f2").astype(np.float32)
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
    data = np.round(v * 65535.0).astype(np.uint16)
    raw = b"".join(b"\x00" + data[y].tobytes() for y in range(h))

    def chunk(tag, payload):
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def apply_relief(gray, radius=1.5, smooth_map=None, smooth_radius=0.0):
    r = max(0.0, float(radius))
    g = np.asarray(gray.convert("L"), dtype=np.float32)
    if smooth_map is not None and smooth_radius > 0:
        a = np.asarray(gray.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)
        b = np.asarray(gray.filter(ImageFilter.GaussianBlur(r + smooth_radius)), dtype=np.float32)
        g = a * (1 - smooth_map) + b * smooth_map
    else:
        g = np.asarray(gray.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)
    gy, gx = np.gradient(g)
    n = np.sqrt(1 + gx * gx + gy * gy)
    lx, ly, lz = 0.5, 0.5, 1.0
    ln = np.sqrt(lx * lx + ly * ly + lz * lz)
    shade = np.clip((-gx * lx - gy * ly + lz) / (n * ln), 0, 1)
    return Image.fromarray((shade * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r * 0.5))


def _points_field(dfloat, img_rgb, fov_y=90.0, scale=1.0):
    d = np.clip(np.asarray(dfloat, dtype=np.float32), 0.0, 1.0)
    h, w = d.shape
    z = np.clip(d, 0.05, 1.0) * float(scale)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    fy = (h / 2.0) / math.tan(math.radians(max(1.0, float(fov_y)) / 2.0))
    ys, xs = np.mgrid[0:h, 0:w]
    X = ((xs - cx) * z / fy)
    Y = (-(ys - cy) * z / fy)
    rgb = np.asarray(img_rgb.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
    return X.reshape(-1), Y.reshape(-1), z.reshape(-1), rgb


def _grid_faces(w, h):
    faces = []
    for i in range(h - 1):
        base = i * w
        for j in range(w - 1):
            a = base + j
            faces.append((a, a + 1, a + w + 1, a + w))
    return faces


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
            "end_header\n" % (n, len(faces))
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


def _infer_depth(model_path, img, nw, nh, metric=False):
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
        while proc.poll() is None:
            time.sleep(0.2)
            av = avail_ram_mb()
            if av is not None and av < floor:
                proc.kill()
                raise MemoryError("Обработка остановлена: свободная ОЗУ закончилась.\n"
                                  "Уменьшите «Множитель» или включите «Эконом ОЗУ».")
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


def write_ply(path, dfloat, img_rgb, fov_y=90.0, scale=1.0):
    X, Y, z, rgb = _points_field(dfloat, img_rgb, fov_y=fov_y, scale=scale)
    h, w = np.asarray(dfloat).shape[:2]
    n = X.size
    faces = _grid_faces(w, h)
    with open(path, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\nelement vertex %d\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "element face %d\nproperty list uchar int vertex_indices\n"
            "end_header\n" % (n, len(faces))
        )
        np.savetxt(
            f,
            np.column_stack([X, Y, z, rgb[:, 0], rgb[:, 1], rgb[:, 2]]),
            fmt="%.6f %.6f %.6f %d %d %d",
        )
        for fc in faces:
            f.write("4 %d %d %d %d\n" % fc)


def write_glb(path, dfloat, img_rgb, fov_y=90.0, scale=1.0):
    import struct
    X, Y, z, rgb = _points_field(dfloat, img_rgb, fov_y=fov_y, scale=scale)
    h, w = dfloat.shape[0], dfloat.shape[1]
    n = X.size
    pos = np.stack([X, Y, z], axis=-1).astype(np.float32)
    lcol = ((rgb / 255.0) ** 2.2).astype(np.float32)
    col = np.concatenate([lcol, np.ones((n, 1), dtype=np.float32)], axis=-1)
    faces = []
    for i in range(h - 1):
        base = i * w
        for j in range(w - 1):
            a = base + j
            faces.append((a, a + 1, a + w + 1))
            faces.append((a, a + w + 1, a + w))
    idx = np.asarray(faces, dtype=np.uint32).reshape(-1)
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


class DepthUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Карта глубины — настройки")
        self.geometry("1000x800")
        self.configure(bg="#1e1e1e")

        self.vars = {}

        pad = dict(padx=10, pady=4, anchor="w")

        # --- фото ---
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="Фото:").pack(side="left", **pad)
        self.var_src = tk.StringVar(value=SRC)
        ttk.Entry(row, textvariable=self.var_src).pack(side="left", fill="x", expand=True, padx=10, pady=4)
        ttk.Button(row, text="Поиск фото", command=self.pick_src).pack(side="left", padx=10, pady=4)

        # --- модель ---
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="Модель:", width=15).pack(side="left", **pad)
        _m = str(cfg.get("model", "small")).lower()
        if _m not in ("small", "base", "large", "midas", "zoe",
                  "base_in", "base_out", "large_in", "large_out", "large_mix"):
            _m = "small"
        _mdisp = {"small": "Small", "base": "Base", "large": "Large", "midas": "MiDaS",
                  "zoe": "ZoeDepth", "base_in": "Base Indoor", "base_out": "Base Outdoor",
                  "large_in": "Large Indoor", "large_out": "Large Outdoor",
                  "large_mix": "Large Mix (3 модели)"}
        self.vars["model"] = tk.StringVar(value=_mdisp[_m])
        ttk.Combobox(row, textvariable=self.vars["model"], values=["Small", "Base", "Large", "MiDaS", "ZoeDepth",
                            "Base Indoor", "Base Outdoor", "Large Indoor", "Large Outdoor",
                            "Large Mix (3 модели)"],
                     width=12, state="readonly").pack(side="left", padx=10, pady=4)
        ttk.Label(row, text="(Base/Large — детальнее, но медленнее)", foreground="#888").pack(side="left")

        # --- slider helpers: 2 колонки (левая/правая) ---
        slider_grid = ttk.Frame(self)
        slider_grid.pack(fill="x", padx=5)
        slider_grid.columnconfigure(0, weight=1)
        slider_grid.columnconfigure(1, weight=1)

        def slider(label, key, lo, hi, fmt, r, c):
            cell = ttk.Frame(slider_grid)
            cell.grid(row=r, column=c, sticky="ew", padx=4, pady=3)
            ttk.Label(cell, text=label, width=15).pack(side="left")
            var = tk.DoubleVar(value=float(cfg[key]))
            s = ttk.Scale(cell, from_=lo, to=hi, variable=var, orient="horizontal",
                          command=lambda _v: lbl.configure(text=fmt(var.get())))
            s.pack(side="left", fill="x", expand=True, padx=6)
            lbl = ttk.Label(cell, text=fmt(var.get()), width=7)
            lbl.pack(side="left")
            self.vars[key] = var

        slider("Глубина от %", "depth_lo", 0, 90, lambda v: f"{v:.0f}", 0, 0)
        slider("Оверлей", "overlay_alpha", 0, 1, lambda v: f"{v:.2f}", 0, 1)
        slider("Глубина до %", "depth_hi", 10, 100, lambda v: f"{v:.0f}", 1, 0)
        slider("Размытие DoF", "blur_strength", 0, 20, lambda v: f"{v:.1f}", 1, 1)
        slider("Гамма глубины", "depth_gamma", 0.2, 5.0, lambda v: f"{v:.2f}", 2, 0)
        slider("Ширина фокуса", "focus_width", 5, 100, lambda v: f"{v:.0f}", 2, 1)
        slider("Контраст карты", "depth_contrast", 0.0, 4.0, lambda v: f"{v:.2f}", 3, 0)
        slider("Сглаживание", "guided", 0, 100, lambda v: f"{v:.0f}", 4, 0)
        slider("Шумодав", "denoise", 0, 100, lambda v: f"{v:.0f}", 4, 1)

        # --- разрешение входа ---
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="Множитель:", width=15).pack(side="left", **pad)
        self.vars["input_mult"] = tk.DoubleVar(value=float(cfg.get("input_mult", 2)))
        self._input_scale = ttk.Scale(row, from_=1, to=5, variable=self.vars["input_mult"],
                                      orient="horizontal", command=lambda _v: self._update_input_range())
        self._input_scale.pack(side="left", fill="x", expand=True, padx=10, pady=4)
        self._input_lbl = ttk.Label(row, text=f"{self.vars['input_mult'].get():.0f}", width=8)
        self._input_lbl.pack(side="left", padx=10)
        self.vars["compress"] = tk.BooleanVar(value=bool(cfg.get("compress", False)))

        # --- эконом ОЗУ: под множителем ---
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Checkbutton(row, text="Эконом ОЗУ", variable=self.vars["compress"],
                        command=self._update_input_range).pack(side="left", padx=25, pady=2)
        self._ram_hint = ttk.Label(self, text="", foreground="#e06666")
        self._ram_hint.pack(fill="x", padx=25, pady=(0, 2))
        self._update_input_range()

        # --- сжать оригинал: делитель 1x-4x ---
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="Сжать оригинал до:").pack(side="left", **pad)
        _sd = str(cfg.get("src_div", "1x"))
        if _sd not in ("1x", "2x", "3x", "4x"):
            _sd = "1x"
        self.vars["src_max"] = tk.StringVar(value=_sd)
        ttk.Combobox(row, textvariable=self.vars["src_max"],
                     values=["1x", "2x", "3x", "4x"],
                     width=10, state="readonly").pack(side="left", padx=10, pady=4)

        row = ttk.Frame(self)
        row.pack(fill="x")
        self.vars["invert"] = tk.BooleanVar(value=bool(cfg.get("invert", False)))
        ttk.Checkbutton(row, text="Инверсия", variable=self.vars["invert"]).pack(side="left", padx=10, pady=4)
        self.vars["render2"] = tk.BooleanVar(value=bool(cfg.get("render2", False)))
        ttk.Checkbutton(row, text="Гибрид",
                        variable=self.vars["render2"]).pack(side="left", padx=10, pady=4)
        self.vars["dof_near"] = tk.BooleanVar(value=bool(cfg.get("dof_near", False)))
        ttk.Checkbutton(row, text="DoF: размывать близкое", variable=self.vars["dof_near"]).pack(side="left", padx=10, pady=4)

        row = ttk.Frame(self)
        row.pack(fill="x")
        self.vars["depth16"] = tk.BooleanVar(value=bool(cfg.get("depth16", False)))
        ttk.Checkbutton(row, text="16-бит глубина (PNG)", variable=self.vars["depth16"]).pack(side="left", padx=10, pady=4)
        self.vars["depth32"] = tk.BooleanVar(value=bool(cfg.get("depth32", False)))
        ttk.Checkbutton(row, text="32-бит float (TIFF)", variable=self.vars["depth32"]).pack(side="left", padx=10, pady=4)
        self.vars["depth_exr"] = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Бело-чёрная глубина (EXR, всегда)",
                        variable=self.vars["depth_exr"]).pack(side="left", padx=10, pady=4)
        self.vars["cmap_exr"] = tk.BooleanVar(value=bool(cfg.get("cmap_exr", False)))
        ttk.Checkbutton(row, text="Цветная карта глубины (EXR)",
                        variable=self.vars["cmap_exr"]).pack(side="left", padx=10, pady=4)
        self.vars["normal_png"] = tk.BooleanVar(value=bool(cfg.get("normal_png", False)))
        ttk.Checkbutton(row, text="Карта нормалей (EXR)",
                        variable=self.vars["normal_png"]).pack(side="left", padx=10, pady=4)
        self.vars["ply_out"] = tk.BooleanVar(value=bool(cfg.get("ply_out", False)))
        ttk.Checkbutton(row, text="XYZ-облако (PLY)",
                        variable=self.vars["ply_out"]).pack(side="left", padx=10, pady=4)
        self.vars["glb_out"] = tk.BooleanVar(value=bool(cfg.get("glb_out", False)))
        ttk.Checkbutton(row, text="Модель 3D (GLB)",
                        variable=self.vars["glb_out"]).pack(side="left", padx=10, pady=4)
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="FOV PLY:").pack(side="left", padx=5, pady=4)
        self.vars["ply_fov"] = tk.StringVar(value=str(cfg.get("ply_fov", 90)))
        ttk.Spinbox(row, from_=20, to=170, width=6, textvariable=self.vars["ply_fov"]).pack(side="left")
        ttk.Label(row, text="Масштаб PLY:").pack(side="left", padx=5, pady=4)
        self.vars["ply_scale"] = tk.StringVar(value=str(cfg.get("ply_scale", 1.0)))
        ttk.Spinbox(row, from_=0.1, to=20, increment=0.1, width=6,
                    textvariable=self.vars["ply_scale"]).pack(side="left")

        row = ttk.Frame(self)
        row.pack(fill="x")
        self.vars["focus_enable"] = tk.BooleanVar(value=bool(cfg.get("focus_enable", False)))
        ttk.Checkbutton(row, text="Точка фокуса: клик по предпросмотру", variable=self.vars["focus_enable"]).pack(side="left", padx=10, pady=4)

        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="Размер выхода:").pack(side="left", **pad)
        self.vars["out_mult"] = tk.StringVar(value=str(cfg.get("out_mult", "1x")))
        cmb = ttk.Combobox(row, textvariable=self.vars["out_mult"],
                           values=["1x", "2x", "3x", "4x"], width=8, state="readonly")
        cmb.pack(side="left", padx=10, pady=4)

        # --- кнопки ---
        row = ttk.Frame(self)
        row.pack(fill="x", pady=8)
        self.btn_run = ttk.Button(row, text="▶ Применить", command=self.run)
        self.btn_run.pack(side="left", padx=10)
        ttk.Button(row, text="Открыть папку", command=self.open_folder).pack(side="left", padx=10)
        self.var_autobig = None
        self.lbl_status = ttk.Label(row, text="Готово. Нажми Применить")
        self.lbl_status.pack(side="left", padx=10)

        # --- preview (2 панели + переключение типа) ---
        self.TYPE_NAMES = ["Глубина", "Оверлей", "Depth-of-Field", "Цветная карта", "Рельеф 3D", "Цветное фото"]
        self.type_idx = 0

        grid = ttk.Frame(self)
        grid.pack(fill="both", expand=True, padx=10, pady=10)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(1, weight=1)

        row = ttk.Frame(grid)
        row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        self.lbl_type = ttk.Label(row, text="Тип: Глубина", font=("", 11, "bold"))
        self.lbl_type.pack(side="left", padx=5)
        ttk.Button(row, text="⇄ Сменить тип", command=self.cycle_type).pack(side="left", padx=5)

        prev_frame = ttk.LabelFrame(grid, text="Предыдущий")
        prev_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.lbl_size_prev = ttk.Label(prev_frame, text="—")
        self.lbl_size_prev.pack(fill="x", padx=4, pady=(2, 0))
        self.prev_view = tk.Label(prev_frame, bg="#000", text="—", fg="#888")
        self.prev_view.pack(fill="both", expand=True)

        cur_frame = ttk.LabelFrame(grid, text="Текущий")
        cur_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        self.lbl_size_cur = ttk.Label(cur_frame, text="Размер: —")
        self.lbl_size_cur.pack(fill="x", padx=4, pady=(2, 0))
        self.cur_view = tk.Label(cur_frame, bg="#000", text="Обработайте фото", fg="#888")
        self.cur_view.pack(fill="both", expand=True)

        self._prev_img = None
        self._cur_img = None
        self._prev_ovl = None
        self._cur_ovl = None
        self._prev_dof = None
        self._cur_dof = None
        self._prev_col = None
        self._cur_col = None
        self._prev_rel = None
        self._cur_rel = None
        self._src_img = None

        self._focus_x = int(cfg.get("focus_x", -1))
        self._focus_y = int(cfg.get("focus_y", -1))

        self.cur_view.bind("<Button-1>", self._cur_click)

        self.bind("<Configure>", lambda _e: self._schedule_resize())

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
            if hasattr(self, "lbl_status"):
                self.lbl_status.configure(text=f"Фото выбрано: {os.path.basename(f)}")

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
        self.lbl_status.configure(text=f"Фото выбрано: {path.split('/')[-1]}")

    def _update_input_range(self):
        top = 10 if self.vars["compress"].get() else 5
        self._input_scale.configure(to=top)
        if float(self.vars["input_mult"].get()) > top:
            self.vars["input_mult"].set(float(top))
        self._input_lbl.configure(text=f"{self.vars['input_mult'].get():.0f}")
        v = int(round(float(self.vars["input_mult"].get())))
        if self.vars["compress"].get():
            self._ram_hint.configure(text="")
        elif v >= 5:
            self._ram_hint.configure(text="Осторожно: 5x может съесть более 32 ГБ ОЗУ")
        elif v >= 3:
            self._ram_hint.configure(text="Осторожно: большой множитель сильно увеличивает расход ОЗУ")
        else:
            self._ram_hint.configure(text="")

    def _cur_click(self, e):
        if self.vars["focus_enable"].get():
            self._set_focus(e)

    def _set_focus(self, e):
        scale = getattr(self.cur_view, "_disp_scale", None)
        if scale is None:
            return
        img = self._all_images()[1]
        if img is None:
            return
        x = int((e.x - self.cur_view._disp_ox) / scale)
        y = int((e.y - self.cur_view._disp_oy) / scale)
        if 0 <= x < img.width and 0 <= y < img.height:
            self._focus_x = x
            self._focus_y = y
            self.lbl_status.configure(text=f"Фокус: {x}, {y}")
            self.show_pair()

    def _draw_focus_marker(self, img):
        scale = self.cur_view._disp_scale
        x = int(self._focus_x * scale)
        y = int(self._focus_y * scale)
        d = ImageDraw.Draw(img)
        r = 10
        d.line([(x - r, y), (x + r, y)], fill=(255, 70, 70), width=2)
        d.line([(x, y - r), (x, y + r)], fill=(255, 70, 70), width=2)
        d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 70, 70), width=2)
        return img

    def collect(self):
        c = dict(cfg)
        c["model"] = {"small": "small", "base": "base", "large": "large",
                      "midas": "midas", "zoedepth": "zoe", "zoe": "zoe",
                      "base indoor": "base_in", "base outdoor": "base_out",
                      "large indoor": "large_in",
                      "large outdoor": "large_out",
                      "large mix (in/out)": "large_mix"}.get(self.vars["model"].get().lower(), "small")
        c["src"] = self.var_src.get()
        mult = max(1, int(round(float(self.vars["input_mult"].get()))))
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
        if self.vars["compress"].get():
            scale = (200 * mult) / max(src_img.width, src_img.height)
            c["in_w"] = max(14, int(src_img.width * scale))
            c["in_h"] = max(14, int(src_img.height * scale))
        else:
            c["in_w"] = max(256, mult * src_img.width)
            c["in_h"] = max(256, mult * src_img.height)
        out_mult = int(self.vars["out_mult"].get().replace("x", ""))
        c["out_size"] = f"{out_mult * orig_w}x{out_mult * orig_h}"
        c["compress"] = bool(self.vars["compress"].get())
        for k, var in self.vars.items():
            if isinstance(var, tk.DoubleVar):
                c[k] = float(var.get())
        c["invert"] = bool(self.vars["invert"].get())
        c["render2"] = bool(self.vars["render2"].get())
        c["dof_near"] = bool(self.vars["dof_near"].get())
        c["depth16"] = bool(self.vars["depth16"].get())
        c["depth32"] = bool(self.vars["depth32"].get())
        c["depth_exr"] = bool(self.vars["depth_exr"].get())
        c["cmap_exr"] = bool(self.vars["cmap_exr"].get())
        c["normal_png"] = bool(self.vars["normal_png"].get())
        c["ply_out"] = bool(self.vars["ply_out"].get())
        c["glb_out"] = bool(self.vars["glb_out"].get())
        c["gen"] = "none"
        c["gen_amp"] = 1.0
        c["gen_freq"] = 2.0
        try:
            c["ply_fov"] = float(str(self.vars["ply_fov"].get()).replace(",", "."))
        except ValueError:
            c["ply_fov"] = 90.0
        try:
            c["ply_scale"] = float(str(self.vars["ply_scale"].get()).replace(",", "."))
        except ValueError:
            c["ply_scale"] = 1.0
        c["focus_enable"] = bool(self.vars["focus_enable"].get())
        c["focus_width"] = float(self.vars["focus_width"].get())
        c["focus_x"] = self._focus_x
        c["focus_y"] = self._focus_y
        return c

    def run(self):
        if STATE["busy"]:
            return
        c = self.collect()
        json.dump(c, open(CONFIG, "w"), indent=2)
        STATE["busy"] = True
        self.btn_run.configure(state="disabled")
        self.lbl_status.configure(text="Обработка...")
        threading.Thread(target=self.work, args=(c,), daemon=True).start()

    def work(self, c):
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
                self.after(0, lambda: self.lbl_status.configure(
                    text="Модель «%s» готова (PLY/GLB)" % kind))
                return
            if c["model"] == "midas":
                model_paths = [common.find_model("midas_v21_small_256.onnx")]
                if not os.path.exists(model_paths[0]):
                    raise ValueError("MiDaS: нет файла midas_v21_small_256.onnx в папке Z-depth")
            elif c["model"] == "zoe":
                model_paths = [common.find_model("zoedepth_nk_fp16.onnx")]
                if not os.path.exists(model_paths[0]):
                    raise ValueError("ZoeDepth: нет файла zoedepth_nk_fp16.onnx в папке Z-depth")
            elif c["model"] == "large_mix":
                names = ("large", "large_in", "large_out")
                model_paths = [common.find_model(f"depth_anything_v2_{_n}.onnx") for _n in names]
                for _n, _p in zip(names, model_paths):
                    if not os.path.exists(_p):
                        self.after(0, lambda: self.lbl_status.configure(
                            text=f"Скачивание модели {_n.capitalize()}..."))
                        download_model(_n, _p)
                self.after(0, lambda: self.lbl_status.configure(
                    text="Модели Large/IN/OUT готовы"))
            else:
                model_paths = [common.find_model(f"depth_anything_v2_{c['model']}.onnx")]
                if not os.path.exists(model_paths[0]):
                    self.after(0, lambda: self.lbl_status.configure(
                        text=f"Скачивание модели {c['model'].capitalize()}..."))
                    download_model(c["model"], model_paths[0])
                    self.after(0, lambda: self.lbl_status.configure(text="Модель скачана"))
            gen = str(c.get("gen") or "none")
            if gen != "none":
                out_size = str(c["out_size"]).replace("х", "x").replace("Х", "x").replace(" ", "")
                try:
                    w0, h0 = [int(x) for x in out_size.split("x")]
                except ValueError:
                    w0, h0 = 512, 512
                if w0 < 2 or h0 < 2:
                    w0, h0 = 512, 512
                nw, nh = w0, h0
                dfull = gen_depth_map(gen, h0, w0,
                                      float(c.get("gen_amp", 1.0) or 1.0),
                                      float(c.get("gen_freq", 2.0) or 2.0))
                dfull = np.asarray(dfull, dtype=np.float32)
                img = Image.fromarray((colormap_rgb(dfull) * 255).astype(np.uint8)).convert("RGB")
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

                            ds = []
                            if c["model"] == "large_mix":
                                for _n, _mp in zip(("large", "large_in", "large_out"), model_paths):
                                    ds.append(_infer_depth(_mp, img, nw, nh,
                                                           metric=_n in ("large_in", "large_out")))
                            else:
                                for _mp in model_paths:
                                    ds.append(_infer_depth(_mp, img, nw, nh,
                                                           metric=c["model"] in METRIC_MODELS))
                            if c["model"] == "large_mix":
                                self.__dict__["_mix_ds"] = [np.asarray(_d, dtype=np.float32) for _d in ds]
                            d = np.mean(ds, axis=0)
                            dfull = np.asarray(Image.fromarray(d).resize(img.size, Image.BICUBIC), dtype=np.float32)
            g_strength = float(c.get("guided", 0) or 0)
            if g_strength > 0:
                guide = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
                radius = max(1.0, g_strength * 0.2)
                dfull = guided_filter(guide, dfull, radius)
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
            lo = float(c.get("depth_lo", 0.0) or 0.0) / 100.0
            hi = float(c.get("depth_hi", 100.0) or 100.0) / 100.0
            if hi > lo + 1e-6:
                dfloat = np.clip((dfloat - lo) / (hi - lo), 0.0, 1.0)
            g = float(c.get("depth_gamma", 1.0) or 1.0)
            if g != 1.0:
                dfloat = np.clip(dfloat, 0.0, 1.0) ** (1.0 / g)

            out_size = str(c["out_size"]).replace("х", "x").replace("Х", "x").replace(" ", "")
            w, h = [int(x) for x in out_size.split("x")]
            dfloat = np.asarray(Image.fromarray(dfloat).resize((w, h), Image.LANCZOS), dtype=np.float32)

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

            write_exr(f"{OUT}/photo_depth.exr", np.clip(dfloat, 0.0, 1.0).astype(np.float32))
            # исходное фото всегда сохраняем в EXR (float linear RGB) — для цвета меша
            write_exr(f"{OUT}/photo_src.exr",
                      _srgb_to_linear(np.asarray(img_out, dtype=np.float32) / 255.0))
            if c["depth16"]:
                Image.fromarray((np.clip(dfloat, 0.0, 1.0) * 65535).astype(np.uint16)).save(f"{OUT}/photo_depth_16.png")
            if c.get("depth32"):
                Image.fromarray(np.clip(dfloat, 0.0, 1.0)).save(f"{OUT}/photo_depth_32.tif")
            deep = bool(c.get("depth16")) or bool(c.get("depth32"))
            overlay = Image.blend(img_out, depth_out, c["overlay_alpha"])
            rel_map = None
            dfl = None
            dl = np.clip(dfloat, 0.0, 1.0)
            if c["focus_enable"] and c["focus_x"] >= 0 and c["focus_y"] >= 0:
                d0 = float(dl[int(c["focus_y"]) % dl.shape[0], int(c["focus_x"]) % dl.shape[1]])
                fw = max(1.0, float(c["focus_width"])) / 255.0
                sharp = np.clip(1 - np.abs(dl - d0) / fw, 0, 1)
                sharp = sharp * sharp * (3 - 2 * sharp)
                rel_map = 1 - sharp
                far = Image.fromarray((rel_map * 255).astype(np.uint8))
                dof = Image.composite(depth_out.filter(ImageFilter.GaussianBlur(c["blur_strength"] * 8)), depth_out, far)
                if deep:
                    b = _blur_f(dl, max(1.0, float(c["blur_strength"]) * 8))
                    dfl = b * rel_map + dl * (1 - rel_map)
            else:
                blurred = depth_out.filter(ImageFilter.GaussianBlur(c["blur_strength"] * 3))
                far = depth_out.convert("L").point(lambda p: 255 - p)
                if c["dof_near"]:
                    dof = Image.composite(depth_out, blurred, far)
                    rel_map = dl
                    if deep:
                        b = _blur_f(dl, max(1.0, float(c["blur_strength"]) * 3))
                        dfl = dl * rel_map + b * (1 - rel_map)
                else:
                    dof = Image.composite(blurred, depth_out, far)
                    if deep:
                        m = 1.0 - dl
                        b = _blur_f(dl, max(1.0, float(c["blur_strength"]) * 3))
                        dfl = b * m + dl * (1 - m)
            if dfl is not None:
                Image.fromarray((np.clip(dfl, 0.0, 1.0) * 65535).astype(np.uint16)).save(f"{OUT}/photo_dof_16.png")
                if c.get("depth32"):
                    Image.fromarray(np.clip(dfl, 0.0, 1.0)).save(f"{OUT}/photo_dof_32.tif")

            a = img_out
            b = depth_out
            side = Image.new("RGB", (w * 2 + 10, h), (40, 40, 40))
            side.paste(a, (0, 0))
            side.paste(b, (w + 10, 0))

            crgb = colormap_rgb(np.clip(dfloat, 0.0, 1.0))
            _mix_ds = self.__dict__.get("_mix_ds")
            if _mix_ds:
                for _tag, _di in zip(("large", "in", "out"), _mix_ds):
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

            if c.get("normal_png"):
                # карта нормалей из карты глубины: цвет как у matcap
                # check_normal+y (X=R, Y=G, Z=B, плоскость лицом к камере = +Z=синий)
                d = np.clip(dfloat, 0.0, 1.0).astype(np.float32)
                gy, gx = np.gradient(d)
                n = np.stack([-gx, -gy, np.ones_like(d)], axis=-1).astype(np.float32)
                ln = np.linalg.norm(n, axis=-1, keepdims=True)
                n = n / np.maximum(ln, 1e-6)
                write_exr(f"{OUT}/photo_normal.exr", n * 0.5 + 0.5)

            # исходное цветное фото в EXR (float linear) — для качества в меше
            _i = np.asarray(img_out.convert("RGB"), dtype=np.float32) / 255.0
            write_exr(f"{OUT}/photo_src.exr", _srgb_to_linear(np.clip(_i, 0.0, 1.0)).astype(np.float32))

            if c.get("ply_out") or c.get("glb_out"):
                try:
                    if c.get("ply_out"):
                        write_ply(f"{OUT}/photo_points.ply", dfloat, img_out,
                                  fov_y=c.get("ply_fov", 90.0), scale=c.get("ply_scale", 1.0))
                    if c.get("glb_out"):
                        write_glb(f"{OUT}/photo_points.glb", dfloat, img_out,
                                  fov_y=c.get("ply_fov", 90.0), scale=c.get("ply_scale", 1.0))
                except Exception as e:
                    print("ply/glb error:", e)

            if c.get("render2") and os.path.realpath(str(c["src"])) != os.path.realpath(f"{OUT}/photo_colormap.exr"):
                # первый рендер не пропадает: сохраняем его отдельно
                try:
                    write_exr(f"{OUT}/photo_colormap_1.exr", np.concatenate([
                        _srgb_to_linear(crgb).astype(np.float32),
                        np.clip(dfloat, 0.0, 1.0)[..., None].astype(np.float32)], axis=-1))
                    Image.fromarray((crgb * 255.0).round().astype(np.uint8)).save(f"{OUT}/photo_colormap_1.png")
                    Image.fromarray((np.clip(dfloat, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB").save(
                        f"{OUT}/photo_depth_1.png")
                except Exception:
                    pass
                # второй проход: свежая цветная карта сама становится источником;
                # результат гибридизируется с первым (50/50)
                self._hybrid_prev = {
                    "a": np.clip(dfloat, 0.0, 1.0).astype(np.float32),
                    "rgb": np.clip(crgb, 0.0, 1.0).astype(np.float32),
                    "size": (w, h),
                }
                c2 = dict(c)
                c2["src"] = f"{OUT}/photo_colormap.exr"
                c2["render2"] = False
                c2["hybrid"] = True
                try:
                    self.after(0, lambda: self.lbl_status.configure(text="Рендер 1/3 готов, считаю второй…"))
                except Exception:
                    pass
                return self.work(c2)


            depth_p = depth_out.copy()
            overlay_p = overlay.copy()
            dof_p = dof.copy()
            col_p = Image.fromarray((crgb * 255.0).round().astype(np.uint8))
            rel_p = apply_relief(depth_out.convert("L"), c["blur_strength"], rel_map, c["blur_strength"])
            src_p = img_out.copy()

            self.after(0, lambda: self.done(side.copy(), depth_p.copy(), overlay_p.copy(), dof_p.copy(),
                                            col_p.copy(), rel_p.copy(), src_p.copy()))
        except Exception as e:
            msg = str(e)
            import traceback
            with open(os.path.join(common.data_dir(), "ui.log"), "a") as f:
                f.write(traceback.format_exc() + "\n")
            self.after(0, lambda msg=msg: self.fail(msg))

    def cycle_type(self):
        self.type_idx = (self.type_idx + 1) % len(self.TYPE_NAMES)
        self.lbl_type.configure(text=f"Тип: {self.TYPE_NAMES[self.type_idx]}")
        self.show_pair()

    def _schedule_resize(self):
        if hasattr(self, "_resize_after"):
            self.after_cancel(self._resize_after)
        self._resize_after = self.after(80, self._on_resized)

    def _on_resized(self):
        if hasattr(self, "_resize_after"):
            del self._resize_after
        if self._cur_img is not None:
            self.show_pair()

    def _all_images(self):
        if self._cur_img is None:
            return None, None
        if self.type_idx == 0:
            return self._prev_img, self._cur_img
        if self.type_idx == 1:
            return self._prev_ovl, self._cur_ovl
        if self.type_idx == 2:
            return self._prev_dof, self._cur_dof
        if self.type_idx == 3:
            return self._prev_col, self._cur_col
        if self.type_idx == 4:
            return self._prev_rel, self._cur_rel
        return self._src_img, self._src_img

    def done(self, side, depth, overlay, dof, col, rel, src):
        STATE["busy"] = False
        self.btn_run.configure(state="normal")
        self.lbl_status.configure(text="Готово ✔")
        if self._cur_img is not None:
            self._prev_img = self._cur_img
            self._prev_ovl = self._cur_ovl
            self._prev_dof = self._cur_dof
            self._prev_col = self._cur_col
            self._prev_rel = self._cur_rel
        self._cur_img = depth.copy()
        self._cur_ovl = overlay.copy()
        self._cur_dof = dof.copy()
        self._cur_col = col.copy()
        self._cur_rel = rel.copy()
        self._src_img = src.copy()
        self.show_pair()


    def show_pair(self):
        prev, cur = self._all_images()
        self.lbl_size_prev.configure(text=f"{prev.width}×{prev.height}" if prev else "—")
        self.lbl_size_cur.configure(text=f"{cur.width}×{cur.height}" if cur else "—")
        self._set_view(self.prev_view, prev, "—")
        self._set_view(self.cur_view, cur, "Обработайте фото")

    def _set_view(self, widget, img, placeholder):
        if img is not None:
            scale = 1.0
            pw, ph = widget.winfo_width(), widget.winfo_height()
            if pw > 10 and ph > 10:
                scale = min(pw / img.width, ph / img.height)
            w, h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
            disp = img.convert("RGB").resize((w, h), Image.LANCZOS)
            widget._disp_scale = scale
            widget._disp_ox = (pw - w) / 2 if pw > 10 else 0
            widget._disp_oy = (ph - h) / 2 if ph > 10 else 0
            if widget is self.cur_view and self.vars["focus_enable"].get() and self._focus_x >= 0:
                disp = self._draw_focus_marker(disp)
            photo = ImageTk.PhotoImage(disp)
            widget._photo = photo
            widget.configure(image=photo, text="")
        else:
            widget.configure(image="", text=placeholder)

    def fail(self, msg):
        STATE["busy"] = False
        self.btn_run.configure(state="normal")
        self.lbl_status.configure(text="Ошибка")
        messagebox.showerror("Ошибка", msg)

    def open_folder(self):
        open_path(OUT)


if __name__ == "__main__":
    app = DepthUI()
    app.mainloop()
