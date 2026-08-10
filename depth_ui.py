import json
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageTk, ImageDraw
import onnxruntime as ort

CONFIG = "/home/lynx/Z-depth/settings.json"
OUT = "/home/lynx/Z-depth"
SRC = "/home/lynx/Downloads/download (1).png"

DEFAULTS = {
    "model": "large",
    "input_size": 1024,
    "unsharp_radius": 3.0,
    "unsharp_percent": 120.0,
    "unsharp_thresh": 2.0,
    "depth_contrast": 1.0,
    "overlay_alpha": 0.45,
    "blur_strength": 6.0,
    "out_size": "300x300",
    "invert": False,
    "dof_near": False,
    "compress": False,
    "focus_enable": False,
    "focus_x": -1,
    "focus_y": -1,
    "focus_width": 25.0,
    "depth16": False,
}

cfg = DEFAULTS.copy()
if os.path.exists(CONFIG):
    try:
        cfg.update(json.load(open(CONFIG)))
    except Exception:
        pass

STATE = {"busy": False}


def apply_colormap(gray):
    arr = np.asarray(gray.convert("L"), dtype=np.float32)
    t = arr / 255.0
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
    return Image.fromarray((np.stack([r, g, b], axis=-1) * 255).astype(np.uint8), "RGB")


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
    return Image.fromarray((shade * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(r * 0.5))


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
        ttk.Button(row, text="Поиск фото", command=self.browse_photos).pack(side="left", padx=10, pady=4)

        # --- модель ---
        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="Модель:").pack(side="left", **pad)
        self.vars["model"] = tk.StringVar(value=str(cfg["model"]))
        cmb = ttk.Combobox(row, textvariable=self.vars["model"], values=["large", "small"], width=8, state="readonly")
        cmb.pack(side="left", padx=10, pady=4)

        # --- slider helpers ---
        def slider(label, key, lo, hi, step, fmt):
            row = ttk.Frame(self)
            row.pack(fill="x")
            ttk.Label(row, text=label).pack(side="left", **pad)
            var = tk.DoubleVar(value=float(cfg[key]))
            s = ttk.Scale(row, from_=lo, to=hi, variable=var, orient="horizontal", command=lambda _v: lbl.configure(text=fmt(var.get())))
            s.pack(side="left", fill="x", expand=True, padx=10, pady=4)
            lbl = ttk.Label(row, text=fmt(var.get()), width=8)
            lbl.pack(side="left", padx=10)
            self.vars[key] = var

        slider("Unsharp радиус", "unsharp_radius", 0, 10, 0.5, lambda v: f"{v:.1f}")
        slider("Unsharp сила (%)", "unsharp_percent", 0, 400, 5, lambda v: f"{v:.0f}")
        slider("Unsharp порог", "unsharp_thresh", 0, 10, 0.5, lambda v: f"{v:.1f}")
        slider("Контраст карты", "depth_contrast", 0.0, 4.0, 0.05, lambda v: f"{v:.2f}")

        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="Разрешение входа:").pack(side="left", **pad)
        self.vars["input_size"] = tk.StringVar(value=str(cfg.get("input_mult", "2")))
        self._input_spin = ttk.Spinbox(row, from_=1, to=5, textvariable=self.vars["input_size"], width=5)
        self._input_spin.pack(side="left", padx=10, pady=4)
        self.vars["compress"] = tk.BooleanVar(value=bool(cfg.get("compress", False)))
        ttk.Checkbutton(row, text="Сжать до 14px", variable=self.vars["compress"],
                        command=self._update_input_range).pack(side="left", padx=10)
        self._update_input_range()

        slider("Оверлей", "overlay_alpha", 0, 1, 0.01, lambda v: f"{v:.2f}")
        slider("Размытие DoF", "blur_strength", 0, 20, 0.5, lambda v: f"{v:.1f}")

        row = ttk.Frame(self)
        row.pack(fill="x")
        self.vars["invert"] = tk.BooleanVar(value=bool(cfg.get("invert", False)))
        ttk.Checkbutton(row, text="Инверсия", variable=self.vars["invert"]).pack(side="left", padx=10, pady=4)
        self.vars["dof_near"] = tk.BooleanVar(value=bool(cfg.get("dof_near", False)))
        ttk.Checkbutton(row, text="DoF: размывать близкое", variable=self.vars["dof_near"]).pack(side="left", padx=10, pady=4)
        self.vars["depth16"] = tk.BooleanVar(value=bool(cfg.get("depth16", False)))
        ttk.Checkbutton(row, text="16-бит глубина (PNG)", variable=self.vars["depth16"]).pack(side="left", padx=10, pady=4)

        row = ttk.Frame(self)
        row.pack(fill="x")
        self.vars["focus_enable"] = tk.BooleanVar(value=bool(cfg.get("focus_enable", False)))
        ttk.Checkbutton(row, text="Точка фокуса: клик по предпросмотру", variable=self.vars["focus_enable"]).pack(side="left", padx=10, pady=4)
        slider("Ширина фокуса", "focus_width", 5, 100, 1, lambda v: f"{v:.0f}")

        row = ttk.Frame(self)
        row.pack(fill="x")
        ttk.Label(row, text="Размер выхода:").pack(side="left", **pad)
        self.vars["out_mult"] = tk.StringVar(value=str(cfg.get("out_mult", "1")))
        cmb = ttk.Combobox(row, textvariable=self.vars["out_mult"],
                           values=["1x", "2x", "3x", "4x"], width=8, state="readonly")
        cmb.pack(side="left", padx=10, pady=4)

        # --- кнопки ---
        row = ttk.Frame(self)
        row.pack(fill="x", pady=8)
        self.btn_run = ttk.Button(row, text="▶ Применить", command=self.run)
        self.btn_run.pack(side="left", padx=10)
        ttk.Button(row, text="Открыть результат", command=self.open_result).pack(side="left", padx=10)
        ttk.Button(row, text="Открыть папку", command=self.open_folder).pack(side="left", padx=10)
        self.var_autobig = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Крупное окно автоматически", variable=self.var_autobig).pack(side="left", padx=10)
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

        self.prev_view.bind("<Button-1>", lambda _e: self.show_large(0))
        self.cur_view.bind("<Button-1>", self._cur_click)

        self.bigwin = None

        self.bind("<Configure>", lambda _e: self._schedule_resize())

    def pick_src(self):
        f = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All", "*.*")])
        if f:
            self.var_src.set(f)

    def browse_photos(self):
        win = tk.Toplevel(self)
        win.title("Поиск фото")
        win.geometry("900x600")
        win.configure(bg="#1e1e1e")

        folders = {
            "Downloads": "/home/lynx/Downloads",
            "Pictures": "/home/lynx/Pictures",
            "Desktop": "/home/lynx/Desktop",
            "Z-depth": "/home/lynx/Z-depth",
        }
        self._browse_cache = {}

        top = tk.Frame(win, bg="#1e1e1e")
        top.pack(fill="x", padx=8, pady=6)
        var_fold = tk.StringVar(value="Downloads")
        ttk.Combobox(top, textvariable=var_fold, values=list(folders), width=12, state="readonly").pack(side="left")
        var_search = tk.StringVar()
        ttk.Entry(top, textvariable=var_search).pack(side="left", fill="x", expand=True, padx=8)

        def toggle_maximize(_e=None):
            if win.state() == "zoomed":
                win.state("normal")
            else:
                win.state("zoomed")

        top.bind("<Double-Button-1>", toggle_maximize)
        win.bind("<Escape>", lambda _e: win.destroy())

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

        def relayout(*_a):
            cols = max(1, canvas.winfo_width() // 170)
            for i, tile in enumerate(tiles):
                tile.grid(row=i // cols, column=i % cols, padx=4, pady=4)

        canvas.bind("<Configure>", relayout)

        def load_thumbs(paths):
            for path in paths:
                if load_gen[0] == 0 or path not in tile_refs:
                    return
                try:
                    im = Image.open(path)
                    im.thumbnail((150, 150), Image.LANCZOS)
                    im = im.convert("RGB")
                except Exception:
                    continue
                self.after(0, lambda p=path, im=im: self._set_thumb(p, im, tile_refs, load_gen))

        def refresh(*_a):
            load_gen[0] += 1
            for w in inner.winfo_children():
                w.destroy()
            tiles.clear()
            tile_refs.clear()
            folder = folders[var_fold.get()]
            q = var_search.get().strip().lower()
            try:
                files = sorted(os.listdir(folder), reverse=True)
            except Exception:
                files = []
            exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
            paths = []
            for i, name in enumerate(files):
                if not name.lower().endswith(exts):
                    continue
                if q and q not in name.lower():
                    continue
                path = os.path.join(folder, name)
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
                paths.append(path)
            win.after_idle(relayout)
            threading.Thread(target=load_thumbs, args=(paths,), daemon=True).start()

        var_fold.trace_add("write", refresh)
        var_search.trace_add("write", refresh)
        refresh()

    def _set_thumb(self, path, im, tile_refs, load_gen):
        if load_gen[0] == 0 or path not in tile_refs:
            return
        lbl = tile_refs[path]
        try:
            photo = ImageTk.PhotoImage(im)
        except Exception:
            return
        self._browse_cache[path] = photo
        lbl.configure(image=photo, text="", width=0, height=0)
        lbl.image = photo

    def _pick_photo(self, path, win):
        self.var_src.set(path)
        win.destroy()
        self.lbl_status.configure(text=f"Фото выбрано: {path.split('/')[-1]}")

    def _update_input_range(self):
        top = 100 if self.vars["compress"].get() else 5
        self._input_spin.configure(to=top)
        try:
            v = int(str(self.vars["input_size"].get()).replace("x", "").strip())
        except ValueError:
            v = 1
        if v > top:
            self.vars["input_size"].set(str(top))

    def _cur_click(self, e):
        if self.vars["focus_enable"].get():
            self._set_focus(e)
            return
        self.show_large(1)

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
        c["model"] = self.vars["model"].get()
        c["src"] = self.var_src.get()
        mult = int(str(self.vars["input_size"].get()).replace("x", "").strip())
        src_img = Image.open(c["src"]).convert("RGB")
        if self.vars["compress"].get():
            scale = (14 * mult) / max(src_img.width, src_img.height)
            c["in_w"] = max(14, int(src_img.width * scale))
            c["in_h"] = max(14, int(src_img.height * scale))
            c["out_size"] = f"{src_img.width}x{src_img.height}"
        else:
            c["in_w"] = max(256, mult * src_img.width)
            c["in_h"] = max(256, mult * src_img.height)
            out_mult = int(self.vars["out_mult"].get().replace("x", ""))
            c["out_size"] = f"{out_mult * src_img.width}x{out_mult * src_img.height}"
        for k, var in self.vars.items():
            if isinstance(var, tk.DoubleVar):
                c[k] = float(var.get())
        c["invert"] = bool(self.vars["invert"].get())
        c["dof_near"] = bool(self.vars["dof_near"].get())
        c["depth16"] = bool(self.vars["depth16"].get())
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
            model = f"/home/lynx/Z-depth/depth_anything_v2_{c['model']}.onnx"
            img = Image.open(c["src"]).convert("RGB")
            sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
            nw = int(c["in_w"]) - int(c["in_w"]) % 14
            nh = int(c["in_h"]) - int(c["in_h"]) % 14
            inp = img.resize((nw, nh), Image.LANCZOS)
            arr = np.asarray(inp, dtype=np.float32) / 255.0
            arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
            arr = arr.transpose(2, 0, 1)[None]
            name = sess.get_inputs()[0].name
            pred = sess.run(None, {name: arr})[0]
            d = pred[0]
            d = d - d.min()
            d = d / (d.max() + 1e-8)
            depth = Image.fromarray((d * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
            depth = ImageOps.autocontrast(depth)
            depth = depth.filter(ImageFilter.UnsharpMask(
                radius=max(1, int(c["unsharp_radius"])), percent=int(c["unsharp_percent"]), threshold=int(c["unsharp_thresh"])))
            depth = ImageOps.autocontrast(depth)
            if c["depth_contrast"] != 1.0:
                depth = ImageEnhance.Contrast(depth).enhance(c["depth_contrast"])

            if c["invert"]:
                depth = depth.point(lambda p: 255 - p)

            out_size = str(c["out_size"]).replace("х", "x").replace("Х", "x").replace(" ", "")
            w, h = [int(x) for x in out_size.split("x")]
            depth_out = depth.convert("RGB").resize((w, h), Image.LANCZOS)
            img_out = img.convert("RGB").resize((w, h), Image.LANCZOS)

            scale_ratio = max(w, h) / max(1, max(int(c["in_w"]) - int(c["in_w"]) % 14, int(c["in_h"]) - int(c["in_h"]) % 14))
            smooth = min(3.0, max(0.0, (scale_ratio - 1) * 0.35))
            if smooth > 0:
                depth_out = depth_out.filter(ImageFilter.GaussianBlur(smooth))

            depth_out.save(f"{OUT}/photo_depth.png")
            if c["depth16"]:
                df = d.copy()
                if c["depth_contrast"] != 1.0:
                    df = (df - 0.5) * c["depth_contrast"] + 0.5
                if c["invert"]:
                    df = 1 - df
                df = np.clip(df, 0, 1)
                Image.fromarray((df * 65535).astype(np.uint16)).resize((w, h), Image.LANCZOS).save(f"{OUT}/photo_depth_16.png")
            overlay = Image.blend(img_out, depth_out, c["overlay_alpha"])
            overlay.save(f"{OUT}/photo_depth_overlay.png")
            rel_map = None
            if c["focus_enable"] and c["focus_x"] >= 0 and c["focus_y"] >= 0:
                L = np.asarray(depth_out.convert("L"), dtype=np.float32)
                d0 = float(L[int(c["focus_y"]) % L.shape[0], int(c["focus_x"]) % L.shape[1]])
                fw = max(1.0, float(c["focus_width"]))
                sharp = np.clip(1 - np.abs(L - d0) / fw, 0, 1)
                sharp = sharp * sharp * (3 - 2 * sharp)
                rel_map = 1 - sharp
                far = Image.fromarray((rel_map * 255).astype(np.uint8), "L")
                dof = Image.composite(depth_out.filter(ImageFilter.GaussianBlur(c["blur_strength"] * 8)), depth_out, far)
            else:
                blurred = depth_out.filter(ImageFilter.GaussianBlur(c["blur_strength"] * 3))
                far = depth_out.convert("L").point(lambda p: 255 - p)
                if c["dof_near"]:
                    dof = Image.composite(depth_out, blurred, far)
                    rel_map = np.asarray(depth_out.convert("L"), dtype=np.float32) / 255.0
                else:
                    dof = Image.composite(blurred, depth_out, far)
            dof.save(f"{OUT}/photo_dof.png")

            a = img_out
            b = depth_out
            side = Image.new("RGB", (w * 2 + 10, h), (40, 40, 40))
            side.paste(a, (0, 0))
            side.paste(b, (w + 10, 0))
            side.save(f"{OUT}/photo_color_plus_depth.png")

            depth_p = depth_out.copy()
            overlay_p = overlay.copy()
            dof_p = dof.copy()
            col_p = apply_colormap(depth_out)
            rel_p = apply_relief(depth_out.convert("L"), c["blur_strength"], rel_map, c["blur_strength"])
            src_p = img_out.copy()

            self.after(0, lambda: self.done(side.copy(), depth_p.copy(), overlay_p.copy(), dof_p.copy(),
                                            col_p.copy(), rel_p.copy(), src_p.copy()))
        except Exception as e:
            msg = str(e)
            import traceback
            with open("/home/lynx/Z-depth/ui.log", "a") as f:
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
        if self.var_autobig.get():
            self.show_large(1, auto=True)

    def show_large(self, side_idx, auto=False):
        img = self._all_images()[side_idx]
        if img is None:
            return
        if self.bigwin is None or not self.bigwin.winfo_exists():
            self.bigwin = tk.Toplevel(self)
            self.bigwin.title("Просмотр")
            self.bigwin.configure(bg="#1e1e1e")
            self.big_canvas = tk.Canvas(self.bigwin, bg="#1e1e1e", highlightthickness=0)
            self.big_canvas.pack(fill="both", expand=True)
            self._big_scale = 1.0
            self._big_item = self.big_canvas.create_image(0, 0, anchor="nw", image=self._big_photo if hasattr(self, "_big_photo") else None)
            self.bigwin.bind("<Configure>", lambda _e: self._update_big())
            self.bigwin.bind("<MouseWheel>", self._big_wheel)
            self.bigwin.bind("<Button-4>", self._big_wheel)
            self.bigwin.bind("<Button-5>", self._big_wheel)
            self.big_canvas.bind("<MouseWheel>", self._big_wheel)
            self.big_canvas.bind("<Button-4>", self._big_wheel)
            self.big_canvas.bind("<Button-5>", self._big_wheel)
            self.big_canvas.bind("<ButtonPress-2>", self._big_pan_start)
            self.big_canvas.bind("<B2-Motion>", self._big_pan_move)
            self.bigwin.protocol("WM_DELETE_WINDOW", lambda: self.bigwin.withdraw())
        if auto:
            self.bigwin.deiconify()
        self._big_img = img.copy()
        self._big_scale = 1.0
        self._update_big()

    def _big_wheel(self, e):
        delta = e.delta if hasattr(e, "delta") else (0 if e.num == 0 else (-1 if e.num == 4 else 1))
        step = delta / 120 if e.delta else delta
        self._big_scale = max(0.05, min(40.0, self._big_scale * (1.1 ** step)))
        self._update_big()

    def _big_pan_start(self, e):
        self._pan_x = e.x
        self._pan_y = e.y

    def _big_pan_move(self, e):
        dx = e.x - self._pan_x
        dy = e.y - self._pan_y
        self._pan_x = e.x
        self._pan_y = e.y
        self.big_canvas.move(self._big_item, dx, dy)

    def _update_big(self):
        img = getattr(self, "_big_img", None)
        if img is None or not self.bigwin.winfo_exists():
            return
        pw, ph = self.bigwin.winfo_width(), self.bigwin.winfo_height()
        if pw < 10 or ph < 10:
            return
        fit = min(pw / img.width, ph / img.height)
        scale = fit * self._big_scale
        w, h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
        self._big_photo = ImageTk.PhotoImage(img.convert("RGB").resize((w, h), Image.LANCZOS))
        self.big_canvas.itemconfigure(self._big_item, image=self._big_photo)
        cx = pw / 2
        cy = ph / 2
        self.big_canvas.coords(self._big_item, cx - w / 2, cy - h / 2)

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

    def open_result(self):
        os.system(f'xdg-open "{OUT}/photo_color_plus_depth.png" &')

    def open_folder(self):
        os.system(f'xdg-open "{OUT}" &')


if __name__ == "__main__":
    app = DepthUI()
    app.mainloop()
