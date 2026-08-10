import os
import numpy as np
from PIL import Image, ImageFilter

import common

W, H = 1024, 768
rng = np.random.default_rng(7)

ys, xs = np.mgrid[0:H, 0:W]
u = (xs + 0.5) / W * 2.0 - 1.0
v = (ys + 0.5) / H * 2.0 - 1.0
u *= W / H

orig = np.zeros((H, W, 3), dtype=np.float64)
dirv = np.stack([u, v, np.full_like(u, -1.0)], axis=-1)
dirv /= np.linalg.norm(dirv, axis=-1, keepdims=True)
cam = np.array([0.0, 0.0, 4.0])

spheres = []
colors = []
for i in range(7):
    a = rng.uniform(0, 2 * np.pi)
    r = rng.uniform(0.3, 0.7)
    cx = 1.5 * np.cos(a)
    cy = 1.5 * np.sin(a) * 0.7
    cz = rng.uniform(-2.0, 0.5)
    spheres.append((cx, cy, cz, r))
    colors.append(rng.uniform(0.2, 1.0, 3))

light_dir = np.array([0.6, 0.9, -0.4])
light_dir /= np.linalg.norm(light_dir)

depth = np.full((H, W), np.inf)
hit = np.zeros((H, W), dtype=bool)

def intersect_sphere(o, d, c, r):
    oc = o - c
    b = 2.0 * np.sum(oc * d, axis=-1)
    cc = np.sum(oc * oc, axis=-1) - r * r
    disc = b * b - 4.0 * cc
    ok = disc > 0
    sq = np.sqrt(np.maximum(disc, 0))
    t1 = (-b - sq) / 2.0
    t2 = (-b + sq) / 2.0
    t = np.where(ok, np.maximum(t1, 0.0), np.inf)
    return t, ok

for (cx, cy, cz, r), col in zip(spheres, colors):
    d2 = dirv - cam
    t, ok = intersect_sphere(cam, dirv, np.array([cx, cy, cz]), r)
    closer = t < depth
    m = ok & closer
    if not m.any():
        continue
    p = cam + dirv * t[..., None]
    n = (p - np.array([cx, cy, cz])) / r
    diff = np.maximum(np.sum(n * light_dir, axis=-1), 0.0)
    base = col * (0.25 + 0.75 * diff[..., None])
    base += np.array([0.08, 0.05, 0.1]) * np.power(np.maximum(-n[..., 2], 0.0), 4)[..., None]
    base = np.clip(base, 0, 1)
    orig[m] = base[m]
    depth[m] = t[m]
    hit[m] = True

gx, gy = np.mgrid[-3:3:100j, -3:3:100j]
plane_t = (0.0 - cam[2]) / dirv[..., 2]
pp = cam + dirv * plane_t[..., None]
pm = (plane_t > 0) & (pp[..., 0] > -2.4) & (pp[..., 0] < 2.4) & (pp[..., 1] > -1.8) & (pp[..., 1] < 1.8)
pt = plane_t[..., None]
m2 = pm & (~hit) & (plane_t < depth)
if m2.any():
    checker = np.where((np.floor(pp[..., 0] * 2) + np.floor(pp[..., 1] * 2)) % 2 == 0, 0.9, 0.35)
    diffuse = np.maximum(np.sum(np.array([0.0, 0.0, 1.0]) * light_dir), 0.0)
    base = (checker * (0.25 + 0.75 * diffuse))[..., None]
    orig[m2] = base[m2]
    depth[m2] = plane_t[m2]

horizon = np.clip(1.0 - np.abs(v) * 0.8, 0, 1)
sky = np.stack([0.05 + 0.55 * horizon, 0.12 + 0.4 * horizon, 0.45 + 0.35 * horizon], axis=-1)
orig[~hit] = sky[~hit]

near = float(np.nanmin(depth[np.isfinite(depth)]))
far = float(np.nanmax(depth[np.isfinite(depth)]))
depth_n = np.clip(1.0 - (depth - near) / (far - near), 0, 1)
depth_n[~np.isfinite(depth)] = 0.0

img = Image.fromarray((orig * 255).astype(np.uint8))
depth_img = Image.fromarray((depth_n * 255).astype(np.uint8)).convert("L")

img.save(os.path.join(common.data_dir(), "color.png"))
depth_img.save(os.path.join(common.data_dir(), "depth.png"))

# combine 1: depth-of-field (far = blurry)
blurred = img.filter(ImageFilter.GaussianBlur(6))
dof = Image.composite(blurred, img, depth_img.point(lambda p: 255 - p))
dof.save(os.path.join(common.data_dir(), "color_dof.png"))

# combine 2: depth gradient overlay on color photo
depth_rgb = depth_img.convert("RGB")
combined = Image.blend(img, depth_rgb, 0.45)
combined.save(os.path.join(common.data_dir(), "color_depth_overlay.png"))

print("done")
