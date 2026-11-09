import warnings
import numpy as np
import pydicom
import napari
from pathlib import Path
from qtpy.QtWidgets import QApplication, QFileDialog

warnings.filterwarnings("ignore", module="pydicom")

# ======================= Settings ======================================
_app = QApplication.instance() or QApplication([])
_folder = QFileDialog.getExistingDirectory(None, "Select the Zeiss export folder")
if not _folder:
    raise SystemExit("No folder selected")
FOLDER = Path(_folder)

SCALE_UM = (10, 10, 1.95)        # (B-scan, A-scan, depth) µm per pixel, PLEX Elite 3x3 mm
TRANSPOSE_SURFACES = False       # set True if the segmentation lines don't follow the layers
PAR_STRENGTH = 0.8               # projection removal: 0 = off, 1 = full subtraction
OCTA_PROJECTION = "mean"   # "mean" (cleaner) or "max" (as before)
NOISE_SIGMA = 2.0          # structural noise threshold: vitreous mean + N x std
ENHANCE = True             # denoise + local contrast for display
UPSAMPLE = 3               # display upsampling factor for en-face images (1 = off)
DEPTH_EXAGGERATION = 1   # 1 = true proportions; 2-3 makes the layers easier to see
SURFACE_STEP = 1           # mesh resolution for the ILM/RPE surfaces (1 = full)
FLIP_UD = True   # flip the B-scan order (up-down in the en-face view)
FLIP_LR = True    # flip the A-scan direction (left-right in en-face and B-scans)

# Zeiss surfaces in the "Background Processing" file (provisional names)
LAYERS = {
    "ILM":      0x1510,
    "RNFL/GCL": 0x1515,
    "GCL/IPL":  0x1520,
    "IPL/INL":  0x1525,
    "INL/OPL":  0x1530,
    "OPL/ONL":  0x1535,
    "RPE":      0x1540,
    "BM":       0x1550,
    "Choroid":  0x1555,
}

# En-face slabs: name -> (top surface, bottom surface, top offset px, bottom offset px)
SLABS = {
    "Superficial (ILM-IPL)":  ("ILM", "GCL/IPL", 0, 0),
    "Deep (IPL-OPL)":         ("GCL/IPL", "INL/OPL", 0, 0),
    "Outer retina (OPL-RPE)": ("INL/OPL", "RPE", 0, 0),
    "Choriocapillaris":       ("BM", "BM", 0, 10),
    "Whole retina (ILM-RPE)": ("ILM", "RPE", 0, 0),
}
# =======================================================================

ITEM_TAG = b"\xfe\xff\x00\xe0"      # DICOM item tag (FFFE,E000)


def clean_int(value, default=1):
    if value is None or str(value).strip() == "":
        return default
    return int(str(value).split("\x00")[0].strip())


def clean(v):
    return str(v).split("\x00")[0].strip() if v is not None else ""


def load_zeiss(path):
    """Read Zeiss pixel data, either wrapped in DICOM items or plain."""
    ds = pydicom.dcmread(path, force=True)
    rows, cols = clean_int(ds.get("Rows")), clean_int(ds.get("Columns"))
    dtype = np.uint8 if clean_int(ds.get("BitsAllocated"), 8) == 8 else np.uint16
    frame_bytes = rows * cols * np.dtype(dtype).itemsize
    pix = ds.PixelData

    if pix[:4] == ITEM_TAG:                       # encapsulated: walk through the items
        frames, pos = [], 0
        while pos + 8 <= len(pix) and pix[pos:pos + 4] == ITEM_TAG:
            n = int.from_bytes(pix[pos + 4:pos + 8], "little")
            item = pix[pos + 8:pos + 8 + n]
            if len(item) == frame_bytes:          # skips the empty offset table
                frames.append(np.frombuffer(item, dtype=dtype))
            pos += 8 + n
        return np.stack(frames).reshape(-1, rows, cols)

    raw = np.frombuffer(pix, dtype=dtype)         # plain pixel data
    n_frames = raw.size // (rows * cols)
    return raw[:n_frames * rows * cols].reshape(n_frames, rows, cols)


def load_2d(path):
    """Small images: standard decoder first, Zeiss method as fallback."""
    try:
        img = pydicom.dcmread(path, force=True).pixel_array
        if img.ndim == 3:
            img = img[..., 0] if img.shape[-1] in (3, 4) else img[0]
        return img
    except Exception:
        return load_zeiss(path)[0]


def load_surfaces_raw(path, shape_2d):
    ds = pydicom.dcmread(path, force=True)
    surfaces = {}
    for name, el in LAYERS.items():
        if (0x0073, el) in ds:
            a = np.frombuffer(ds[0x0073, el].value, dtype="<u2").reshape(shape_2d).astype(np.int32)
            if a.max() > 0:
                surfaces[name] = a
    return surfaces


def sample_at(vol, surf, half=2):
    """Mean intensity of vol within +-half pixels of a depth surface."""
    idx = np.clip(surf[..., None] + np.arange(-half, half + 1), 0, vol.shape[2] - 1)
    return float(np.take_along_axis(vol, idx, axis=2).mean())


ORIENTATIONS = {
    "as stored":       lambda a: a,
    "transpose":       lambda a: a.T,
    "flip B-scans":    lambda a: a[::-1, :],
    "flip A-scans":    lambda a: a[:, ::-1],
    "rotate 180":      lambda a: a[::-1, ::-1],
    "transpose+flipB": lambda a: a.T[::-1, :],
    "transpose+flipA": lambda a: a.T[:, ::-1],
    "transpose+rot180": lambda a: a.T[::-1, ::-1],
}


def align_surfaces(surfaces, struct, max_shift=60):
    """Find the orientation and depth shift that put the RPE on the brightest layer."""
    rpe = surfaces["RPE"]
    scores = {k: sample_at(struct, f(rpe)) for k, f in ORIENTATIONS.items()}
    best = max(scores, key=scores.get)
    print("Orientation scores (RPE brightness):",
          {k: round(v, 1) for k, v in sorted(scores.items(), key=lambda x: -x[1])})

    f = ORIENTATIONS[best]
    shifts = range(-max_shift, max_shift + 1)
    s_scores = [sample_at(struct, f(rpe) + s) for s in shifts]
    shift = list(shifts)[int(np.argmax(s_scores))]
    print(f"-> using '{best}', depth shift {shift} px")
    return {k: np.clip(f(v) + shift, 0, struct.shape[2] - 1) for k, v in surfaces.items()}, best


def slab_enface(vol, top, bottom, mode="max"):
    d = np.arange(vol.shape[2], dtype=np.int32)
    mask = (d >= top[..., None]) & (d < bottom[..., None])
    data = np.where(mask, vol, 0)
    if mode == "max":
        return data.max(axis=2).astype(np.float32)
    return (data.sum(axis=2) / np.maximum(mask.sum(axis=2), 1)).astype(np.float32)


def normalise(img):
    return img / max(np.percentile(img, 99.5), 1e-6)


def contrast(img):
    lo, hi = np.percentile(img, (1, 99.7))
    return float(lo), float(hi)


def best_position(frame, cubes, step=4):
    """Find the best-matching slice: (r, cube name, axis, index, flipped)."""
    best = (-1.0, None, 0, 0, False)
    for flipped in (False, True):
        fr = frame[::-1] if flipped else frame
        f = fr[:, ::step].astype(np.float32)
        f = (f - f.mean()) / (f.std() + 1e-6)
        for cname, cube in cubes.items():
            sub = cube[:, :, ::step].astype(np.float32)
            for axis in (0, 1):
                for k in range(sub.shape[axis]):
                    c = sub[k] if axis == 0 else sub[:, k]
                    r = float(np.mean(f * (c - c.mean()) / (c.std() + 1e-6)))
                    if r > best[0]:
                        best = (r, cname, axis, k, flipped)
    return best


# ======================= 1. Identify the files ==========================
files = {"cubes": [], "seg": None, "enface": [], "fundus": [], "bscans": []}

for f in sorted(p for p in FOLDER.rglob("*") if p.is_file() and p.suffix.upper() == ".DCM"):
    ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
    sop = clean(ds.get("SOPClassUID"))
    rows, cols = clean_int(ds.get("Rows"), 0), clean_int(ds.get("Columns"), 0)
    frames = clean_int(ds.get("NumberOfFrames"), 1)

    if sop == "1.2.840.10008.5.1.4.1.1.66" and (0x0073, 0x1510) in ds:
        files["seg"] = f
    elif rows and cols and frames >= 100:
        files["cubes"].append(f)
    elif rows == 300 and cols == 300:
        files["enface"].append(f)
    elif sop.startswith("1.2.840.10008.5.1.4.1.1.77.1.5.1"):
        files["fundus"].append(f)
    elif rows and cols:
        files["bscans"].append(f)

for k, v in files.items():
    print(f"{k:7s}:", [p.name[:12] for p in v] if isinstance(v, list) else (v.name[:12] if v else None))

assert len(files["cubes"]) == 2, "Expected two cubes (structure + flow)"
assert files["seg"] is not None, "Segmentation file not found"

# ======================= 2. Load and sort the cubes =====================
cube_a, cube_b = (load_zeiss(p) for p in files["cubes"])

# Structure is brighter overall than flow (independent of the segmentation)
mean_a, mean_b = float(cube_a[::4, ::4, ::4].mean()), float(cube_b[::4, ::4, ::4].mean())
struct, flow = (cube_a, cube_b) if mean_a > mean_b else (cube_b, cube_a)
print(f"Mean {mean_a:.1f} vs {mean_b:.1f} -> structure = "
      f"{files['cubes'][0 if mean_a > mean_b else 1].name[:12]}")

surfaces, orientation = align_surfaces(load_surfaces_raw(files["seg"], struct.shape[:2]), struct)
# Put everything into Zeiss's B-scan order (so it matches the Zeiss display)
# Flip the cubes into Zeiss's B-scan order, and keep the surfaces as Zeiss stored them
# Display orientation: flip cubes and surfaces together, so they stay aligned
if FLIP_UD:
    struct, flow = struct[::-1], flow[::-1]
    surfaces = {k: v[::-1] for k, v in surfaces.items()}
if FLIP_LR:
    struct, flow = struct[:, ::-1], flow[:, ::-1]
    surfaces = {k: v[:, ::-1] for k, v in surfaces.items()}
struct, flow = np.ascontiguousarray(struct), np.ascontiguousarray(flow)
surfaces = {k: np.ascontiguousarray(v) for k, v in surfaces.items()}
print("Surfaces:", list(surfaces), "| orientation:", orientation)

# ======================= 3. Segmentation as a labels volume ==============
seg = np.zeros(struct.shape, dtype=np.uint8)
bb, aa = np.indices(struct.shape[:2])
seg_names = list(surfaces)
for k, name in enumerate(seg_names, start=1):
    seg[bb, aa, np.clip(surfaces[name], 0, struct.shape[2] - 1)] = k

ds_seg = pydicom.dcmread(files["seg"], force=True)
for el in [0x1370, 0x1375, 0x1380, 0x1385, 0x1390, 0x1395]:
    if (0x0073, el) in ds_seg:
        print(f"(0073,{el:04X}):", ds_seg[0x0073, el].value)

# Standard spacing fields and all numeric private fields in the cube header
ds_cube = pydicom.dcmread(files["cubes"][0], force=True, stop_before_pixels=True)
for kw in ["PixelSpacing", "SliceThickness", "SpacingBetweenSlices"]:
    print(kw, ":", ds_cube.get(kw))

# ======================= 4. En-face slabs ===============================
octa, octa_par, struct_slabs = {}, {}, {}
for name, (top, bot, off_t, off_b) in SLABS.items():
    if top in surfaces and bot in surfaces:
        t, b = surfaces[top] + off_t, surfaces[bot] + off_b
        octa[name] = slab_enface(flow, t, b, "max")
        struct_slabs[name] = slab_enface(struct, t, b, "mean")
        if top != "ILM":                                   # slabs below the superficial layer
            source = slab_enface(flow, surfaces["ILM"], t, "max")
            octa_par[name] = np.clip(normalise(octa[name])
                                     - PAR_STRENGTH * normalise(source), 0, None)

# ======================= 5. 3D visualisation ==============================
# Crop depth to retina + choroid
top = max(int(surfaces["ILM"].min()) - 30, 0)
bottom = min(int(surfaces["Choroid"].max()) + 30, struct.shape[2])
s_crop = struct[:, :, top:bottom]
f_crop = flow[:, :, top:bottom]
surf = {k: v - top for k, v in surfaces.items()}
scale3d = (SCALE_UM[0], SCALE_UM[1], SCALE_UM[2] * DEPTH_EXAGGERATION)

# Noise mask from the vitreous (same idea as for the en-face slabs)
d = np.arange(s_crop.shape[2], dtype=np.int32)
vitreous = d < (surf["ILM"] - 10)[..., None]
noise = s_crop[vitreous].astype(np.float32)
thr3d = noise.mean() + NOISE_SIGMA * noise.std()
reliable = s_crop > thr3d


def slab_volume(top_name, bot_name, off_b=0):
    """Flow inside one slab only (zero elsewhere)."""
    mask = (d >= surf[top_name][..., None]) & (d < (surf[bot_name] + off_b)[..., None])
    return np.where(mask & reliable, f_crop, 0).astype(np.uint8)


vessel_layers = {
    "Superficial plexus": (slab_volume("ILM", "GCL/IPL"), "red"),
    "Deep plexus":        (slab_volume("GCL/IPL", "INL/OPL"), "green"),
    "Choroid":            (slab_volume("BM", "Choroid"), "blue"),
}


def surface_mesh(depth_map, step=SURFACE_STEP):
    """Turn a 300x300 depth map into a triangle mesh (vertices in pixel units)."""
    b, a = np.mgrid[0:depth_map.shape[0]:step, 0:depth_map.shape[1]:step]
    z = depth_map[::step, ::step].astype(np.float32)
    verts = np.stack([b.ravel(), a.ravel(), z.ravel()], axis=1).astype(np.float32)
    nr, nc = b.shape
    idx = np.arange(nr * nc).reshape(nr, nc)
    tl, tr = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    bl, br = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.stack([tl, bl, tr], 1), np.stack([tr, bl, br], 1)])
    return verts, faces, z.ravel()


viewer = napari.Viewer(ndisplay=3, title="3D OCT / OCTA")

viewer.add_image(s_crop, name="Structure", colormap="gray", rendering="attenuated_mip",
                 attenuation=0.3, opacity=0.35, scale=scale3d,
                 contrast_limits=contrast(s_crop[::4, ::4, ::4]), visible=False)

for name, (vol, cmap) in vessel_layers.items():
    nz = vol[vol > 0]
    hi = float(np.percentile(nz, 99.5)) if nz.size else 1.0
    viewer.add_image(vol, name=name, colormap=cmap, rendering="mip",
                     blending="additive", scale=scale3d,
                     contrast_limits=(float(np.percentile(nz, 20)) if nz.size else 0, hi))

for name, cmap in [("ILM", "gray"), ("RPE", "yellow")]:
    viewer.add_surface(surface_mesh(surf[name]), name=f"{name} surface",
                       colormap=cmap, opacity=0.25, scale=scale3d,
                       blending="translucent")

viewer.add_labels(seg[:, :, top:bottom], name="Segmentation (2D check)",
                  scale=scale3d, opacity=0.8, visible=False)

viewer.dims.axis_labels = ("B-scan", "A-scan", "Depth")
viewer.camera.angles = (0, -60, 90)       # a tilted view; rotate freely with the mouse
viewer.reset_view()

# ======================= 6. En-face images with matplotlib ===============
import matplotlib.pyplot as plt


def show_grid(images, title, ncols=4, extent=(0, 3, 3, 0)):
    """Plot a dict of 2D images in a grid; extent in mm (None = pixels)."""
    if not images:
        return
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for ax, (name, img) in zip(axes.ravel(), images.items()):
        lo, hi = contrast(img)
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, extent=extent,
                  interpolation="bicubic")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()


# Reference images: Zeiss en-face and fundus
zeiss = {f"Zeiss en-face ({f.name[:6]})": load_2d(f).astype(np.float32) for f in files["enface"]}
fundus = {f"Fundus {img.shape[0]}x{img.shape[1]}": img
          for img in (load_2d(f) for f in files["fundus"])}

show_grid({**zeiss, **{f"OCTA: {k}": v for k, v in octa.items()}},
          "OCTA slabs (with Zeiss en-face for comparison)")
show_grid({f"OCTA (PAR): {k}": v for k, v in octa_par.items()},
          f"OCTA slabs after projection removal (strength {PAR_STRENGTH})")
show_grid({f"Structure: {k}": v for k, v in struct_slabs.items()},
          "Structural en-face slabs")
show_grid(fundus, "Fundus images", ncols=2, extent=None)

# ---- Single B-scans next to the matching cube slices ----
bscan_rows = []
for f in files["bscans"]:
    for j, fr in enumerate(load_zeiss(f)):
        r, cname, axis, k, flipped = best_position(fr, {"structure": struct, "flow": flow})
        fr = fr[::-1] if flipped else fr
        s_slice = struct[k] if axis == 0 else struct[:, k]
        f_slice = flow[k] if axis == 0 else flow[:, k]
        kind = "fast" if axis == 0 else "slow"
        info = f"best match: {cname}, {kind}-axis slice {k}, r={r:.2f}"
        print(f"B-scan {f.name[:5]}#{j}: {info}")
        bscan_rows.append((f"{f.name[:5]}#{j}", fr, s_slice, f_slice, info))

if bscan_rows:
    # Crop depth to the retina and choroid using the segmentation
    top = max(int(surfaces["ILM"].min()) - 60, 0)
    bottom = min(int(surfaces["Choroid"].max()) + 60, struct.shape[2])
    depth_mm = (bottom - top) * SCALE_UM[2] / 1000

    fig, axes = plt.subplots(len(bscan_rows), 3, figsize=(15, 3.5 * len(bscan_rows)),
                             squeeze=False)
    for row, (tag, fr, s, fl, info) in zip(axes, bscan_rows):
        panels = [(fr, f"Single B-scan {tag}"),
                  (s, f"Cube structure\n({info})"),
                  (fl, "Cube flow (same slice)")]
        for ax, (img, title) in zip(row, panels):
            img = img[:, top:bottom].T.astype(np.float32)     # depth vertical
            lo, hi = contrast(img)
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, aspect="auto",
                      extent=[0, 3, depth_mm, 0], interpolation="bicubic")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("mm")
        row[0].set_ylabel("depth (mm)")
    fig.suptitle("Single B-scans vs matching cube slices")
    fig.tight_layout()

plt.show(block=False)
plt.pause(0.1)

napari.run()