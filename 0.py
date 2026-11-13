import warnings
import numpy as np
import pydicom
import napari
from pathlib import Path
from qtpy.QtWidgets import QApplication, QFileDialog
import re
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", module="pydicom")

# ======================= Settings ======================================
_app = QApplication.instance() or QApplication([])
_folder = QFileDialog.getExistingDirectory(None, "Select the Zeiss export folder")
if not _folder:
    raise SystemExit("No folder selected")
FOLDER = Path(_folder)

SCAN_MM = (6.0, 6.0)        # scan size in mm (slow axis, fast axis): set to match the scan, e.g. (6.0, 6.0)
DEPTH_UM_PER_PX = 1.95      # PLEX Elite axial pixel size in tissue
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
    """En-face image between two depth surfaces; reads only the voxels inside the slab."""
    D = vol.shape[2]
    top = np.clip(top, 0, D - 1)
    thick = np.clip(bottom - top, 0, None)
    n = max(int(thick.max()), 1)
    idx = np.clip(top[..., None] + np.arange(n), 0, D - 1)
    data = np.take_along_axis(vol, idx, axis=2)
    valid = np.arange(n) < thick[..., None]
    if mode == "max":
        return np.where(valid, data, 0).max(axis=2).astype(np.float32)
    if mode == "mean_reliable":
        valid = valid & (data > 0)
    return (np.where(valid, data, 0).sum(axis=2) / np.maximum(valid.sum(axis=2), 1)).astype(np.float32)


def normalise(img):
    return img / max(np.percentile(img, 99.5), 1e-6)


def contrast(img):
    lo, hi = np.percentile(img, (1, 99.7))
    return float(lo), float(hi)


_MATCH_CACHE = {}


def _slice_table(cube, axis, step_l, step_d):
    """All slices of one direction, subsampled and normalised; computed once per cube."""
    key = (id(cube), axis, step_l, step_d)
    if key not in _MATCH_CACHE:
        if axis == 0:                                        # fast-axis slices: cube[k]
            t = cube[:, ::step_l, ::step_d]
        else:                                                # slow-axis slices: cube[:, k]
            t = cube[::step_l, :, ::step_d].transpose(1, 0, 2)
        t = t.reshape(t.shape[0], -1).astype(np.float32)
        t -= t.mean(axis=1, keepdims=True)
        t /= t.std(axis=1, keepdims=True) + 1e-6
        _MATCH_CACHE[key] = t
    return _MATCH_CACHE[key]


def best_position(frame, cubes, step_l=2, step_d=8):
    """Best-matching slice: (r, cube name, axis, index, flipped)."""
    best = (-1.0, None, 0, 0, False)
    for flipped in (False, True):
        fr = frame[::-1] if flipped else frame
        f = fr[::step_l, ::step_d].astype(np.float32).ravel()
        f = (f - f.mean()) / (f.std() + 1e-6)
        for cname, cube in cubes.items():
            for axis in (0, 1):
                t = _slice_table(cube, axis, step_l, step_d)
                if t.shape[1] != f.size:
                    continue
                r = t @ f / f.size                           # correlation with every slice at once
                k = int(np.argmax(r))
                if r[k] > best[0]:
                    best = (float(r[k]), cname, axis, k, flipped)
    return best

def neighbour_score(img):
    """Average correlation between neighbouring pixels (high = real image, ~0 = noise)."""
    a = img.astype(np.float32)
    c0 = np.corrcoef(a[:-1].ravel(), a[1:].ravel())[0, 1]
    c1 = np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1]
    return float(np.nanmean([c0, c1]))


def fix_layout(fr):
    """Try reading the same bytes depth-first; keep whichever layout looks like a real image."""
    alt = fr.reshape(-1).reshape(fr.shape[1], fr.shape[0]).T     # depth-first storage
    s_cur, s_alt = neighbour_score(fr), neighbour_score(alt)
    if s_alt > s_cur + 0.1:
        return alt, "depth-first", s_cur, s_alt
    return fr, "as stored", s_cur, s_alt

# ======================= 1. Identify the files ==========================
headers = []
for f in sorted(p for p in FOLDER.rglob("*") if p.is_file() and p.suffix.upper() == ".DCM"):
    ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
    headers.append((f, clean(ds.get("SOPClassUID")),
                    clean_int(ds.get("Rows"), 0), clean_int(ds.get("Columns"), 0),
                    clean_int(ds.get("NumberOfFrames"), 1), ds))

files = {"cubes": [], "seg": None, "enface": [], "fundus": [], "bscans": []}
for f, sop, rows, cols, frames, ds in headers:
    if sop == "1.2.840.10008.5.1.4.1.1.66" and (0x0073, 0x1510) in ds:
        files["seg"] = f
    elif rows and cols and frames >= 100:
        files["cubes"].append(f)

assert len(files["cubes"]) == 2, "Expected two cubes (structure + flow)"
assert files["seg"] is not None, "Segmentation file not found"

# Cube dimensions decide what the other images are
_, _, n_a, n_d, n_b, _ = next(h for h in headers if h[0] == files["cubes"][0])
for f, sop, rows, cols, frames, ds in headers:
    if f in files["cubes"] or f == files["seg"] or not (rows and cols):
        continue
    if {rows, cols} == {n_b, n_a} and cols != n_d:
        files["enface"].append(f)              # same size as the scan area
    elif cols == n_d and rows in (n_a, n_b):
        files["bscans"].append(f)              # same depth as the cube
    else:
        files["fundus"].append(f)              # any other 2D image (camera images)

print(f"Cube: {n_b} B-scans x {n_a} A-scans x {n_d} depth")
for k, v in files.items():
    print(f"{k:7s}:", [p.name[:12] for p in v] if isinstance(v, list) else v.name[:12])

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
print("Applied flips -> FLIP_UD:", FLIP_UD, "| FLIP_LR:", FLIP_LR)
struct, flow = np.ascontiguousarray(struct), np.ascontiguousarray(flow)
surfaces = {k: np.ascontiguousarray(v) for k, v in surfaces.items()}
print("Surfaces:", list(surfaces), "| orientation:", orientation)
def header_floats(value):
    """Numbers from a Zeiss field, ignoring trailing junk."""
    return [float(x) for x in re.findall(r"\d+\.\d+(?:[eE]-?\d+)?", str(value))]

hdr = pydicom.dcmread(files["cubes"][0], force=True, stop_before_pixels=True)
px = header_floats(hdr.get("PixelSpacing"))
sl = header_floats(hdr.get("SpacingBetweenSlices"))
if len(px) >= 2 and sl:
    SCALE_UM = (sl[0] * 1000, px[0] * 1000, px[1] * 1000)
    SCAN_MM = (SCALE_UM[0] * (struct.shape[0] - 1) / 1000, SCALE_UM[1] * (struct.shape[1] - 1) / 1000)
    print("Voxel size from header (µm):", tuple(round(s, 3) for s in SCALE_UM))
else:
    SCALE_UM = (SCAN_MM[0] * 1000 / struct.shape[0], SCAN_MM[1] * 1000 / struct.shape[1], DEPTH_UM_PER_PX)
    print("Header spacing not found; using SCAN_MM:", SCALE_UM)

# ======================= 3. Segmentation as a labels volume ==============
seg = np.zeros(struct.shape, dtype=np.uint8)
bb, aa = np.indices(struct.shape[:2])
seg_names = list(surfaces)
for k, name in enumerate(seg_names, start=1):
    seg[bb, aa, np.clip(surfaces[name], 0, struct.shape[2] - 1)] = k

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

# Depth-dependent noise floor from the noise-reference frames (if present)
noise_frames = [fr for f in files["bscans"] for fr in load_zeiss(f)
                if fr.std() > 1 and best_position(fr, {"structure": struct})[0] < 0.5]
if noise_frames:
    nz = np.concatenate(noise_frames, axis=0).astype(np.float32)   # (A-scans, depth)
    thr_depth = nz.mean(axis=0) + NOISE_SIGMA * nz.std(axis=0)      # one threshold per depth
    print(f"Noise floor from {len(noise_frames)} reference frame(s): "
          f"{thr_depth[:50].mean():.1f} (top) to {thr_depth[-50:].mean():.1f} (bottom)")
else:                                                               # fall back to the vitreous
    d = np.arange(struct.shape[2], dtype=np.int32)
    vitreous = (d < (surfaces["ILM"] - 30)[..., None]) & (d > 50)
    noise = struct[vitreous].astype(np.float32)
    thr_depth = np.full(struct.shape[2], noise.mean() + NOISE_SIGMA * noise.std())

flow_masked = np.where(struct > thr_depth[None, None, :], flow, 0)
reliable = s_crop > thr_depth[top:bottom][None, None, :]


def slab_volume(top_name, bot_name, off_b=0, use_noise_mask=True):
    """Flow inside one slab only (zero elsewhere)."""
    d = np.arange(f_crop.shape[2], dtype=np.int32)              # depth indices of the cropped volume
    mask = (d >= surf[top_name][..., None]) & (d < (surf[bot_name] + off_b)[..., None])
    if use_noise_mask:
        mask &= reliable
    return np.where(mask, f_crop, 0).astype(np.uint8)


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
viewer.dims.order = (0, 2, 1)                      # slider = B-scan; depth vertical, A-scans horizontal
viewer.reset_view()

# # ---- Orthogonal views with crosshair (like Fiji Ctrl+Shift+H) ----------------
# ref_layer = viewer.layers["Structure"]
# raw = np.asarray(ref_layer.data)
# sc = tuple(ref_layer.scale)
#
# # Work out where depth is in the layer: last axis (B, A, D) or middle axis (B, D, A)
# depth_last = (viewer.dims.axis_labels[2] == "Depth")
# if depth_last:
#     ortho_vol = raw.transpose(0, 2, 1)             # -> (B-scan, depth, A-scan)
#     sB, sD, sA = sc[0], sc[2], sc[1]
#     to_layer = lambda b, d, a: [b, a, d]           # (b, d, a) -> layer coordinates
#     from_layer = lambda p: (p[0], p[2], p[1])      # layer coordinates -> (b, d, a)
# else:
#     ortho_vol = raw
#     sB, sD, sA = sc
#     to_layer = lambda b, d, a: [b, d, a]
#     from_layer = lambda p: (p[0], p[1], p[2])
# print("Layer layout:", "(B, A, D)" if depth_last else "(B, D, A)", "| shape:", raw.shape)
#
# nB, nD, nA = ortho_vol.shape
# o_lo, o_hi = contrast(ortho_vol[::4, ::4, ::4])
# state = {"b": nB // 2, "d": nD // 2, "a": nA // 2}
#
# fig_xz = Figure(figsize=(6, 3), tight_layout=True)    # below the main view
# fig_yz = Figure(figsize=(3, 6), tight_layout=True)    # right of the main view
# canvas_xz = FigureCanvasQTAgg(fig_xz)
# canvas_yz = FigureCanvasQTAgg(fig_yz)
# ax_en = fig_xz.add_subplot(1, 1, 1)                   # XZ
# ax_sl = fig_yz.add_subplot(1, 1, 1)                   # YZ
# cross = viewer.add_shapes(ndim=3, name="Crosshair", scale=sc)
#
#
# def clip_state():
#     state["b"] = int(np.clip(state["b"], 0, nB - 1))
#     state["d"] = int(np.clip(state["d"], 0, nD - 1))
#     state["a"] = int(np.clip(state["a"], 0, nA - 1))
#
#
# def redraw():
#     b, d, a = state["b"], state["d"], state["a"]
#
#     # Crosshair on the current B-scan in napari (in the layer's own axis order)
#     cross.data = []
#     cross.add_lines([np.array([to_layer(b, d, 0), to_layer(b, d, nA - 1)], dtype=float),
#                      np.array([to_layer(b, 0, a), to_layer(b, nD - 1, a)], dtype=float)],
#                     edge_color="yellow", edge_width=2)
#
#     # En-face slice at depth d
#     ax_en.clear()
#     ax_en.imshow(ortho_vol[:, d, :], cmap="gray", vmin=o_lo, vmax=o_hi,
#                  extent=[0, nA * sA / 1000, nB * sB / 1000, 0])
#     ax_en.axhline((b + 0.5) * sB / 1000, color="yellow", lw=0.8)
#     ax_en.axvline((a + 0.5) * sA / 1000, color="yellow", lw=0.8)
#     ax_en.set_title(f"En-face at depth {d}", fontsize=9)
#     ax_en.set_title(f"XZ (en-face) at Y = {d}", fontsize=9)
#
#     # Slow-axis cross-section at A-scan a
#     ax_sl.clear()
#     ax_sl.imshow(ortho_vol[:, :, a].T, cmap="gray", vmin=o_lo, vmax=o_hi,
#                  extent=[0, nB * sB / 1000, nD * sD / 1000, 0])
#     ax_sl.axhline((d + 0.5) * sD / 1000, color="yellow", lw=0.8)
#     ax_sl.axvline((b + 0.5) * sB / 1000, color="yellow", lw=0.8)
#     ax_sl.set_title(f"Slow-axis section at A-scan {a}", fontsize=9)
#     ax_sl.set_title(f"YZ (slow-axis section) at X = {a}", fontsize=9)
#
#     canvas_xz.draw_idle()
#     canvas_yz.draw_idle()
#
# def set_from_world(position):
#     p = np.round(ref_layer.world_to_data(position)).astype(int)[:3]
#     b, d, a = from_layer(p)
#     state.update(b=b, d=d, a=a)
#     clip_state()
#     redraw()
#
#
# @viewer.mouse_drag_callbacks.append
# def _on_click(v, event):
#     if v.dims.ndisplay == 2:
#         set_from_world(event.position)
#
#
# @viewer.mouse_move_callbacks.append
# def _on_move(v, event):
#     if v.dims.ndisplay == 2 and "Shift" in event.modifiers:
#         set_from_world(event.position)
#
#
# def _on_slider(event=None):
#     b = int(np.clip(round(ref_layer.world_to_data(viewer.dims.point)[0]), 0, nB - 1))
#     if b != state["b"]:
#         state["b"] = b
#         redraw()
#
#
# viewer.dims.events.current_step.connect(_on_slider)
#
#
# def _on_side_click(event):
#     if event.inaxes is ax_en and event.xdata is not None:
#         state["a"] = int(event.xdata * 1000 / sA)
#         state["b"] = int(event.ydata * 1000 / sB)
#     elif event.inaxes is ax_sl and event.xdata is not None:
#         state["b"] = int(event.xdata * 1000 / sB)
#         state["d"] = int(event.ydata * 1000 / sD)
#     else:
#         return
#     clip_state()
#     viewer.dims.set_current_step(0, state["b"])
#     redraw()
#
#
# canvas_xz.mpl_connect("button_press_event", _on_side_click)
# canvas_yz.mpl_connect("button_press_event", _on_side_click)
# viewer.window.add_dock_widget(canvas_yz, area="right", name="YZ")
# viewer.window.add_dock_widget(canvas_xz, area="bottom", name="XZ")
# redraw()

ref = viewer.layers["Structure"]
nB = ref.data.shape[0]
centre = tuple(s / 2 for s in ref.data.shape)

plane_layer = viewer.add_image(
    ref.data, name="B-scan plane (3D)", colormap="gray", scale=ref.scale,
    contrast_limits=ref.contrast_limits, blending="translucent",
    depiction="plane",
    plane={"position": (nB // 2, centre[1], centre[2]),
           "normal": (1, 0, 0),           # plane perpendicular to the B-scan axis
           "thickness": 1},
)


def _dims_to_plane(event=None):
    """Move the 3D plane to the B-scan selected with napari's own slider."""
    b = int(np.clip(round(ref.world_to_data(viewer.dims.point)[0]), 0, nB - 1))
    pos = list(plane_layer.plane.position)
    if int(pos[0]) != b:
        pos[0] = b
        plane_layer.plane.position = tuple(pos)


viewer.dims.events.current_step.connect(_dims_to_plane)

# ======================= 6. En-face images with matplotlib ===============
def show_grid(images, title, ncols=4, extent=None):
    """Plot a dict of 2D images in a grid.
    extent=None -> axes in mm from SCAN_MM; extent=False -> axes in pixels."""
    if not images:
        return
    if extent is None:
        extent = (0, SCAN_MM[1], SCAN_MM[0], 0)
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for ax, (name, img) in zip(axes.ravel(), images.items()):
        lo, hi = contrast(img)
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi,
                  extent=None if extent is False else extent,
                  interpolation="bicubic")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()


# Reference images: Zeiss en-face and fundus
zeiss = {}
for f in files["enface"]:
    img = load_2d(f).astype(np.float32)
    img = ORIENTATIONS[orientation](img)   # same Zeiss-to-cube correction as the surfaces
    if FLIP_UD:
        img = np.flipud(img)
    if FLIP_LR:
        img = np.fliplr(img)
    zeiss[f"Zeiss en-face ({f.name[:6]})"] = img

fundus = {f"Fundus {img.shape[0]}x{img.shape[1]}": img
          for img in (load_2d(f) for f in files["fundus"])}

show_grid({**zeiss, **{f"OCTA: {k}": v for k, v in octa.items()}},
          "OCTA slabs (with Zeiss en-face for comparison)")
show_grid({f"OCTA (PAR): {k}": v for k, v in octa_par.items()},
          f"OCTA slabs after projection removal (strength {PAR_STRENGTH})")
show_grid({f"Structure: {k}": v for k, v in struct_slabs.items()},
          "Structural en-face slabs")
show_grid(fundus, "Fundus images", ncols=2, extent=False)

# ---- Single B-scans next to the matching cube slices ----
bscan_rows = []
for f in files["bscans"]:
    for j, fr in enumerate(load_zeiss(f)):
        if fr.shape[1] != struct.shape[2]:
            print(f"Skipping {f.name[:12]}: shape {fr.shape} is not a B-scan of this cube")
            continue
        tag = f"{f.name[:5]}#{j}"
        if fr.std() < 1:
            print(f"Skipping {tag}: blank frame")
            continue
        fr, layout, s_cur, s_alt = fix_layout(fr)
        print(f"{tag}: layout {layout} (smoothness {s_cur:.2f} as stored, {s_alt:.2f} depth-first)")
        if max(s_cur, s_alt) < 0.3:
            print(f"Skipping {tag}: still looks like noise (possibly encoded by Zeiss)")
            continue
        r, cname, axis, k, flipped = best_position(fr, {"structure": struct, "flow": flow})
        if r < 0.5:
            print(f"Skipping {tag}: no match in the cube (r={r:.2f}), probably a noise reference")
            continue
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
            img = img[:, top:bottom].T.astype(np.float32)        # depth vertical
            width_mm = img.shape[1] * SCALE_UM[1] / 1000         # A-scans x spacing
            lo, hi = contrast(img)
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, aspect="auto",
                      extent=[0, width_mm, depth_mm, 0], interpolation="bicubic")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("mm")
        row[0].set_ylabel("depth (mm)")
    fig.suptitle("Single B-scans vs matching cube slices")
    fig.tight_layout()

plt.show(block=False)
plt.pause(0.1)

napari.run()
