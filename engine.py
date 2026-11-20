"""Silhouette engine: local cut-out, cleanup, smooth outline, cut files.

Everything here runs on this computer. The only network use is rembg's
one-time download of the model file itself; photos are never sent anywhere.
"""
from __future__ import annotations

import gc
import logging
import os
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage import measure

try:  # iPhone photos (HEIC); optional
    from pillow_heif import register_heif_opener

    register_heif_opener()
except Exception:  # pragma: no cover - the app works without it
    pass

REV = 2  # raise when outline() changes its result, so saved outlines are redone
WORK_MAX = 2000  # longest side of the working photo, px
# A loaded model is let go after this long unused. The best one holds about
# 10 GB of graphics memory while loaded; loading it again costs about 10 s.
IDLE_SECONDS = 90
UNITS_PER_INCH = 72.0  # Cricut Design Space reads SVG numbers as 72 per inch

MODELS = {
    "best": "birefnet-portrait",  # sharpest hair and face edges, slower, ~1 GB
    "fast": "u2net_human_seg",  # quick, ~170 MB
}

DEFAULT_SETTINGS = {
    "height_in": 2.25,  # finished height of the cut piece
    "smooth": 3.0,  # 0..10
    "detail_mm": 0.5,  # thinnest feature kept, at finished size
    "flip": False,
    "tilt_deg": 0.0,  # straighten: turns the picture clockwise by this much
    "cut_on": True,
    "cut": None,  # [[x,y],[x,y],[x,y]] as fractions of the photo: left, middle pull, right
}

_sessions: dict[str, object] = {}
_lock = threading.Lock()
_idle: threading.Timer | None = None


def release_models() -> None:
    """Forget the loaded models. The next cut-out loads what it needs again."""
    with _lock:
        _sessions.clear()
    gc.collect()


def _arm_idle() -> None:
    global _idle
    if _idle:
        _idle.cancel()
    _idle = threading.Timer(IDLE_SECONDS, release_models)
    _idle.daemon = True
    _idle.start()


def load_photo(fp) -> Image.Image:
    """Open a photo upright, in RGB, shrunk to the working size."""
    img = Image.open(fp)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((WORK_MAX, WORK_MAX), Image.LANCZOS)
    return img


def segment(img: Image.Image, quality: str = "best") -> tuple[np.ndarray, str]:
    """Return (0/255 mask of the person, name of the model that made it).
    Runs the local model; "best" falls back to the fast one if it cannot run."""
    from rembg import new_session, remove

    # The big model needs the graphics card (DirectML); on the CPU it runs out
    # of memory. If it cannot run at all, the small model takes over.
    tries = [("birefnet-portrait", ["DmlExecutionProvider", "CPUExecutionProvider"])] if quality != "fast" else []
    tries.append((MODELS["fast"], ["CPUExecutionProvider"]))
    with _lock:
        for i, (name, providers) in enumerate(tries):
            try:
                if name not in _sessions:
                    _sessions[name] = new_session(name, providers=providers)
                mask = remove(img, session=_sessions[name], only_mask=True)
                break
            except Exception:
                _sessions.pop(name, None)
                if i == len(tries) - 1:
                    raise
    _arm_idle()
    arr = np.asarray(mask.convert("L"))
    if arr.shape[:2] != (img.height, img.width):
        arr = cv2.resize(arr, (img.width, img.height), interpolation=cv2.INTER_LINEAR)
    return clean_cutout(np.where(arr > 127, 255, 0).astype(np.uint8)), name


def clean_cutout(mask: np.ndarray) -> np.ndarray:
    """Tidy a model's mask once, before anyone edits it: fill the specks it
    leaves inside the head. Done here and not in outline(), so a small gap the
    user erases on purpose is never filled back in."""
    return _fill_small_holes(mask, 0.003)


def turn(img: Image.Image, mask: np.ndarray, k: int):
    """Turn photo and mask together by k quarter turns counter-clockwise."""
    k %= 4
    if not k:
        return img, mask
    op = {1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}[k]
    return img.transpose(op), np.ascontiguousarray(np.rot90(mask, k))


def upright_turns(mask: np.ndarray) -> int:
    """Quarter turns (counter-clockwise) that put the body at the bottom.

    A head-and-shoulders photo has the neck and shoulders running off one edge;
    that edge should be the bottom. Returns 0 when the photo already looks
    upright or the cut-out does not clearly run off one edge.
    """
    m = mask > 0
    b = max(3, min(mask.shape) // 200)
    touch = {0: m[-b:].any(axis=0).mean(), 2: m[:b].any(axis=0).mean(),
             1: m[:, :b].any(axis=1).mean(), 3: m[:, -b:].any(axis=1).mean()}
    k = max(touch, key=touch.get)
    if k == 0 or touch[k] < 0.10 or touch[k] < 1.5 * touch[0]:
        return 0
    return k


def _largest(binary: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:
        return binary
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == keep, 255, 0).astype(np.uint8)


def _fill_small_holes(binary: np.ndarray, frac: float) -> np.ndarray:
    """Fill enclosed holes smaller than frac of the shape's area.

    Cut-out models leave specks inside a head; each would be cut as a tiny hole.
    Bigger gaps (between a ponytail and the neck) are real and stay."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(binary), connectivity=4)
    if n <= 2:
        return binary
    h, w = binary.shape
    limit = frac * int(np.count_nonzero(binary))
    small = [i for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] < limit
             and stats[i, cv2.CC_STAT_LEFT] > 0 and stats[i, cv2.CC_STAT_TOP] > 0
             and stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] < w
             and stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] < h]
    if not small:
        return binary
    out = binary.copy()
    out[np.isin(labels, small)] = 255
    return out


def _bbox(binary: np.ndarray):
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def default_cut(mask: np.ndarray) -> list[list[float]]:
    """Guess a neckline: find the neck (narrowest row in the lower half) and
    put a gentle curve a little below it."""
    h, w = mask.shape
    box = _bbox(_largest(mask))
    if box is None:
        return [[0.1, 0.85], [0.5, 0.92], [0.9, 0.85]]
    x0, y0, x1, y1 = box
    bh = y1 - y0
    widths = (mask[y0:y1] > 0).sum(axis=1).astype(np.float32)
    k = max(3, bh // 40) | 1
    widths = cv2.blur(widths.reshape(-1, 1), (1, k)).ravel()
    a, b = int(bh * 0.45), int(bh * 0.92)
    neck = a + int(np.argmin(widths[a:b])) if b > a else int(bh * 0.7)
    y = min(y0 + neck + 0.10 * bh, y1 - 0.02 * bh)
    pad = 0.04 * w
    cx = (x0 + x1) / 2
    return [
        [max(0.0, (x0 - pad) / w), (y - 0.03 * bh) / h],
        [cx / w, min(1.0, (y + 0.07 * bh) / h)],
        [min(1.0, (x1 + pad) / w), (y - 0.03 * bh) / h],
    ]


def _rot(pts, deg: float, w: int, h: int) -> np.ndarray:
    """Turn points clockwise (as seen on screen) about the photo's centre."""
    pts = np.asarray(pts, dtype=np.float64)
    if not deg:
        return pts
    t = np.deg2rad(deg)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    c = np.array([w / 2.0, h / 2.0])
    return (pts - c) @ r.T + c


def _apply_cut(binary: np.ndarray, cut, tilt: float = 0.0) -> np.ndarray:
    h, w = binary.shape
    p0, c, p2 = (np.array([p[0] * w, p[1] * h], dtype=np.float64) for p in cut)
    t = np.linspace(0, 1, 60).reshape(-1, 1)
    curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t**2 * p2
    big = 10 * max(w, h)
    left, right = (p0, p2) if p0[0] <= p2[0] else (p2, p0)
    if p0[0] > p2[0]:
        curve = curve[::-1]
    poly = np.vstack([[-big, left[1]], curve, [big, right[1]], [big, big], [-big, big]])
    poly = _rot(poly, -tilt, w, h)  # the neckline is drawn on the straightened picture
    out = binary.copy()
    cv2.fillPoly(out, [np.round(poly).astype(np.int32)], 0)
    return out


def _bezier_path(pts: np.ndarray) -> list[tuple]:
    """Closed smooth curve through pts (Catmull-Rom as cubic Beziers).
    Returns [(start), (c1, c2, end), ...] as arrays."""
    n = len(pts)
    prev, nxt, nxt2 = np.roll(pts, 1, 0), np.roll(pts, -1, 0), np.roll(pts, -2, 0)
    c1 = pts + (nxt - prev) / 6.0
    c2 = nxt - (nxt2 - pts) / 6.0
    return [pts[0]] + [(c1[i], c2[i], nxt[i]) for i in range(n)]


def outline(mask: np.ndarray, settings: dict) -> dict:
    """Clean the mask and trace it.

    Returns {"loops": [bezier loops in photo px], "bbox": [x0,y0,x1,y1],
    "final": cleaned 0/255 mask} or {"loops": []} when nothing is left.
    """
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    tilt = float(s.get("tilt_deg") or 0.0)
    if s["cut_on"] and s.get("cut"):
        binary = _apply_cut(binary, s["cut"], tilt)
    binary = _largest(binary)
    box = _bbox(binary)
    if box is None:
        return {"loops": [], "bbox": None, "final": binary}
    bh = box[3] - box[1]

    # Drop anything too thin to cut at the finished size, close hairline gaps.
    px_per_mm = bh / (float(s["height_in"]) * 25.4)
    r = int(round(float(s["detail_mm"]) / 2 * px_per_mm))
    if r >= 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        binary = _largest(binary)

    # Smooth by blurring the mask and tracing its half level (sub-pixel).
    pad = 8
    sigma = max(0.8, float(s["smooth"]) * 0.0012 * bh)
    soft = cv2.copyMakeBorder(binary, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    soft = cv2.GaussianBlur(soft.astype(np.float32) / 255.0, (0, 0), sigma)
    final = np.where(soft[pad:-pad, pad:-pad] > 0.5, 255, 0).astype(np.uint8)

    eps = max(0.5, bh / 2500.0)
    min_area = max(30.0, (bh * 0.004) ** 2)
    loops, allpts = [], []
    for c in measure.find_contours(soft, 0.5):
        pts = c[:, ::-1] - pad  # (row, col) -> (x, y)
        if len(pts) < 8 or not np.allclose(pts[0], pts[-1]):
            continue
        pts = pts[:-1].astype(np.float32)
        if abs(cv2.contourArea(pts.reshape(-1, 1, 2))) < min_area:
            continue
        simp = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
        if len(simp) < 4:
            continue
        simp = _rot(simp, tilt, mask.shape[1], mask.shape[0])  # loops come out straightened
        loop = _bezier_path(simp)
        loops.append(loop)
        # measure the curve itself (it bows a little past its anchor points)
        p0 = np.vstack([loop[0]] + [seg[2] for seg in loop[1:-1]])
        c1, c2, p3 = (np.vstack([seg[i] for seg in loop[1:]]) for i in range(3))
        for t in (0.25, 0.5, 0.75):
            allpts.append((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * p3)
        allpts.append(simp)
    if not loops:
        return {"loops": [], "bbox": None, "final": final}
    allpts = np.vstack(allpts)
    bbox = [float(allpts[:, 0].min()), float(allpts[:, 1].min()),
            float(allpts[:, 0].max()), float(allpts[:, 1].max())]
    return {"loops": loops, "bbox": bbox, "final": final}


def path_d(loops, scale=1.0, dx=0.0, dy=0.0, flip_about=None, nd=2) -> str:
    """SVG path data. Point -> ((x or mirrored x) * scale + dx, y * scale + dy)."""

    def f(p):
        x = (flip_about - p[0]) if flip_about is not None else p[0]
        return f"{x * scale + dx:.{nd}f} {p[1] * scale + dy:.{nd}f}"

    parts = []
    for loop in loops:
        parts.append("M" + f(loop[0]))
        parts.extend("C" + f(c1) + " " + f(c2) + " " + f(e) for c1, c2, e in loop[1:])
        parts.append("Z")
    return "".join(parts)


def size_in(result: dict, settings: dict) -> tuple[float, float]:
    x0, y0, x1, y1 = result["bbox"]
    h = float({**DEFAULT_SETTINGS, **settings}["height_in"])
    return (x1 - x0) / (y1 - y0) * h, h


def placed_d(result: dict, settings: dict, x_in=0.0, y_in=0.0) -> str:
    """Path data at finished size, top-left at (x_in, y_in), in 72-per-inch units."""
    s = {**DEFAULT_SETTINGS, **settings}
    x0, y0, x1, y1 = result["bbox"]
    scale = float(s["height_in"]) * UNITS_PER_INCH / (y1 - y0)
    flip = (x0 + x1) if s["flip"] else None
    return path_d(result["loops"], scale, x_in * UNITS_PER_INCH - x0 * scale,
                  y_in * UNITS_PER_INCH - y0 * scale, flip)


def _svg(w_in: float, h_in: float, body: str) -> str:
    # Numbers are 72 per inch (what Design Space assumes) and width/height say
    # the same thing in inches, so every program agrees on the size.
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_in:.4f}in" height="{h_in:.4f}in" '
        f'viewBox="0 0 {w_in * UNITS_PER_INCH:.2f} {h_in * UNITS_PER_INCH:.2f}">\n{body}</svg>\n'
    )


def _xml(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


BORDER_FILL = "#d9b27c"
TEXT_FILL = "#3a3a3a"
TEXT_GAP = 0.12  # inches between the shape and the first line of text

# ---- lettering -------------------------------------------------------------
# Design Space ignores <text> in an uploaded SVG, so names are written as
# outlines, traced from a font that is already on this computer.
# Any font installed on the computer can be used. A font's key is its file
# name in lower case ("georgia.ttf"; "cambria.ttc#1" inside a collection).
_OLD_KEYS = {  # choices from earlier versions -> the files they meant, first one found wins
    "serif": ["georgia.ttf", "times.ttf", "dejavuserif.ttf", "liberationserif-regular.ttf"],
    "sans": ["arial.ttf", "segoeui.ttf", "dejavusans.ttf", "liberationsans-regular.ttf"],
    "bold": ["arialbd.ttf", "segoeuib.ttf", "dejavusans-bold.ttf", "liberationsans-bold.ttf"],
    "script": ["segoesc.ttf", "segoepr.ttf", "comic.ttf"],
}
_SUGGESTED = ["georgia.ttf", "arial.ttf", "arialbd.ttf", "segoesc.ttf", "segoepr.ttf", "gabriola.ttf", "inkfree.ttf",
              "lhandw.ttf", "brushsci.ttf", "comic.ttf", "times.ttf", "verdana.ttf", "cooperblack.ttf", "coopbl.ttf",
              "dejavuserif.ttf", "dejavusans.ttf", "dejavusans-bold.ttf"]


def _font_dirs() -> list[Path]:
    dirs = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
            Path(os.environ.get("LOCALAPPDATA", "~")).expanduser() / "Microsoft" / "Windows" / "Fonts",  # fonts you installed
            Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts",
            Path.home() / ".local" / "share" / "fonts",
            Path("/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path.home() / "Library" / "Fonts"]
    return [d for d in dirs if d.is_dir()]


logging.getLogger("fontTools").setLevel(logging.ERROR)  # it grumbles about odd dates in old fonts
_scan_lock = threading.Lock()


def _registry() -> dict:
    """{key: (path, number in its collection or -1, name shown)} for every
    font here that has ordinary letters and digits. Looking through a few
    hundred fonts takes a few seconds, so it is done once."""
    with _scan_lock:
        return _scan()


@lru_cache(maxsize=1)
def _scan() -> dict:
    try:
        from fontTools.ttLib import TTCollection, TTFont
    except ImportError:
        return {}
    out, seen = {}, set()
    for d in _font_dirs():
        for f in sorted(d.rglob("*")):
            ext = f.suffix.lower()
            if ext not in (".ttf", ".otf", ".ttc"):
                continue
            try:
                count = len(TTCollection(str(f), lazy=True).fonts) if ext == ".ttc" else 1
                for i in range(count):
                    number = i if ext == ".ttc" else -1
                    font = TTFont(str(f), fontNumber=number, lazy=True)
                    cmap = font.getBestCmap() or {}
                    if not all(ord(c) in cmap for c in "AaZz09"):
                        continue  # a symbol font
                    names = font["name"]
                    family, style = names.getBestFamilyName(), names.getBestSubFamilyName()
                    label = family if style in (None, "", "Regular", "Normal", "Roman", "Book") else f"{family} {style}"
                    key = f.name.lower() + (f"#{i}" if ext == ".ttc" else "")
                    if label and label not in seen and key not in out:
                        seen.add(label)
                        out[key] = (f, number, label)
            except Exception:
                continue  # a damaged or unusual font file: leave it out
    return out


def resolve_font(key) -> str | None:
    """A usable font key for this choice: itself, what an older choice meant,
    or the default. None when the computer has no usable font at all."""
    reg = _registry()
    if key in reg:
        return key
    for old in (key, "serif"):
        for name in _OLD_KEYS.get(old, []):
            if name in reg:
                return name
    return next(iter(sorted(reg, key=lambda k: reg[k][2].lower())), None)


def fonts() -> list[dict]:
    """Every lettering choice here, by name: [{"key", "label", "suggested"}]."""
    reg = _registry()
    return [{"key": k, "label": v[2], "suggested": k in _SUGGESTED}
            for k, v in sorted(reg.items(), key=lambda kv: kv[1][2].lower())]


@lru_cache(maxsize=64)
def _font(key: str):
    from fontTools.ttLib import TTFont

    path, number, _ = _registry()[resolve_font(key)]
    return TTFont(str(path), fontNumber=number, lazy=True)


def _run(text: str, key: str):
    """([(glyph name, x in font units)], total width in font units, units per em)."""
    font = _font(key)
    cmap, hmtx = font.getBestCmap(), font["hmtx"]
    out, x = [], 0
    for ch in text:
        g = cmap.get(ord(ch)) or cmap.get(ord("?")) or ".notdef"
        out.append((g, x))
        x += hmtx[g][0]
    return out, x, font["head"].unitsPerEm


def text_width_em(text: str, key: str) -> float:
    _, w, upm = _run(text, key)
    return w / upm


def text_d(text: str, key: str, size: float, cx: float, baseline: float) -> str:
    """Path data for text as outlines: letters `size` tall (the em), centred on
    cx, sitting on `baseline`, all in the output's units."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen

    glyphs = _font(key).getGlyphSet()
    run, w, upm = _run(text, key)
    k = size / upm
    x0 = cx - w * k / 2
    pen = SVGPathPen(glyphs, ntos=lambda v: f"{v:.2f}")
    for g, x in run:
        glyphs[g].draw(TransformPen(pen, (k, 0, 0, -k, x0 + x * k, baseline)))
    return pen.getCommands()


def arc_d(segments: list[tuple[str, str, float]], cx: float, cy: float, radius: float) -> str:
    """Text as outlines along the bottom of a circle: centred at the bottom,
    reading left to right, tops of the letters towards the centre. segments is
    [(text, font key, letter size)], one after another, so the name and the
    second line can differ in font and size."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen

    runs = []
    for text, key, size in segments:
        run, w, upm = _run(text, key)
        runs.append((run, _font(key), size / upm, w * size / upm))
    total = sum(r[3] for r in runs)
    out, start = [], 0.0
    for run, font, k, width in runs:
        glyphs, hmtx = font.getGlyphSet(), font["hmtx"]
        pen = SVGPathPen(glyphs, ntos=lambda v: f"{v:.2f}")
        for g, x in run:
            adv = hmtx[g][0] * k
            th = (start + x * k + adv / 2 - total / 2) / radius  # angle of this letter's middle, 0 = bottom
            c, sn = np.cos(th), np.sin(th)
            px, py = cx + radius * sn, cy + radius * c
            glyphs[g].draw(TransformPen(pen, (k * c, -k * sn, -k * sn, -k * c, px - adv / 2 * c, py + adv / 2 * sn)))
        out.append(pen.getCommands())
        start += width
    return "".join(out)


def text_arc_d(text: str, key: str, size: float, cx: float, cy: float, radius: float) -> str:
    return arc_d([(text, key, size)], cx, cy, radius)


# ---- ornaments ---------------------------------------------------------------
ORNAMENTS = {"circle": "Circle", "oval": "Oval", "square": "Square", "rect": "Rectangle",
             "hexagon": "Hexagon", "heart": "Heart", "star": "Star"}

def ornament_points(shape: str, w: float, h: float, n: int = 200) -> np.ndarray:
    """Outline of an ornament w wide and h tall, centred on (0, 0)."""
    if shape in ("square", "rect"):
        pts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
    elif shape == "hexagon":
        a = np.deg2rad(np.arange(6) * 60 + 30)
        pts = np.column_stack([np.cos(a), np.sin(a)])
    elif shape == "star":
        a = np.deg2rad(np.arange(10) * 36 - 90)
        r = np.where(np.arange(10) % 2 == 0, 1.0, 0.5)
        pts = np.column_stack([r * np.cos(a), r * np.sin(a)])
    elif shape == "heart":
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([16 * np.sin(t) ** 3,
                               -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))])
    else:  # circle, oval
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([np.cos(t), np.sin(t)])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return (pts - (lo + hi) / 2) / (hi - lo) * np.array([w, h])  # fill the w x h box exactly


def ornament_svg(shape: str, w: float, h: float, cx: float, cy: float, u: float, attrs: str) -> str:
    """The ornament as one SVG element, centre (cx, cy), all in inches times u."""
    if shape in ("circle", "oval"):
        return f'<ellipse cx="{cx * u:.2f}" cy="{cy * u:.2f}" rx="{w / 2 * u:.2f}" ry="{h / 2 * u:.2f}" {attrs}/>'
    if shape in ("square", "rect"):
        return (f'<rect x="{(cx - w / 2) * u:.2f}" y="{(cy - h / 2) * u:.2f}" width="{w * u:.2f}" '
                f'height="{h * u:.2f}" {attrs}/>')
    pts = " ".join(f"{(cx + x) * u:.2f},{(cy + y) * u:.2f}" for x, y in ornament_points(shape, w, h))
    return f'<polygon points="{pts}" {attrs}/>'


@lru_cache(maxsize=64)
def _poly(shape: str, w: float, h: float) -> np.ndarray:
    return ornament_points(shape, w, h, 240).astype(np.float32)


def _room(shape: str, w: float, h: float, y: float) -> float:
    """Usable width inside the ornament at height y (0 = its middle), for
    something centred left to right. 0 where the middle is outside the shape
    (above the dip of a heart)."""
    if shape in ("circle", "oval"):
        t = 1 - (y / (h / 2)) ** 2
        return w * (t**0.5 if t > 0 else 0.0)
    pts = _poly(shape, w, h)
    x0, y0 = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    hit = (y0 - y) * (y1 - y) < 0  # edges this height passes through
    xs = x0[hit] + (y - y0[hit]) * (x1[hit] - x0[hit]) / (y1[hit] - y0[hit])
    right, left = xs[xs > 0], xs[xs <= 0]
    if len(right) % 2 == 0 or not len(left):  # an even count to the right: the middle is outside
        return 0.0
    return 2 * float(min(right.min(), -left.max()))


def _inside(shape: str, w: float, h: float, x: float, y: float) -> bool:
    if shape in ("circle", "oval"):
        return (x / (w / 2)) ** 2 + (y / (h / 2)) ** 2 <= 1.03
    return cv2.pointPolygonTest(_poly(shape, w, h).reshape(-1, 1, 2), (float(x), float(y)), True) >= -0.015 * w


# ---- pieces ------------------------------------------------------------------
def _opts(opts) -> dict:
    """What goes around and under the shape in a file.

    border: none | rect (a box) | ornament (the ornament's own shape)
    space: blank inches around, when there is no ornament border
    shape, orn, orn_h: the ornament (a key of ORNAMENTS, width, height in inches;
        the height only counts for ovals and rectangles)
    name_on, line2: letter the child's name and/or a second line
    place: below (straight, under the shape) | arc (curved along the bottom of a circle)
    font, tsize: the name's font (see fonts()) and letter size in inches
    font2, tsize2: the same for the second line (by default the name's font, a little smaller)
        Sizes are what you ask for; lettering shrinks when it would not fit.
    """
    o = {"border": "none", "space": 0.0, "shape": "circle", "orn": 3.0, "orn_h": 0.0,
         "name_on": False, "line2": "", "place": "below", "font": "serif", "tsize": 0.3,
         "font2": None, "tsize2": None, **(opts or {})}
    o["font"] = resolve_font(o["font"])
    o["font2"] = resolve_font(o["font2"]) if o["font2"] else o["font"]
    o["tsize2"] = float(o["tsize2"]) if o["tsize2"] else float(o["tsize"]) * 0.8
    if o["border"] == "circle":  # the name older links use
        o["border"] = "ornament"
    if o["shape"] not in ORNAMENTS:
        o["shape"] = "circle"
    return o


def orn_size(o: dict) -> tuple[float, float]:
    w = float(o["orn"])
    h = float(o["orn_h"]) if o["shape"] in ("oval", "rect") and float(o["orn_h"]) > 0 else w
    return w, h


def _shape_points(result: dict, settings: dict) -> np.ndarray:
    """Points along the cut line, in inches from the shape's top-left corner."""
    s = {**DEFAULT_SETTINGS, **settings}
    x0, y0, x1, y1 = result["bbox"]
    k = float(s["height_in"]) / (y1 - y0)
    pts = np.vstack([np.vstack([loop[0]] + [seg[2] for seg in loop[1:]]) for loop in result["loops"]])
    xs = (x1 - pts[:, 0]) if s["flip"] else (pts[:, 0] - x0)
    return np.column_stack([xs * k, (pts[:, 1] - y0) * k])


def _wanted_lines(name: str, o: dict) -> list[list]:
    """[[text, letter size, font]] the user asked for, before fitting."""
    if not o["font"]:
        return []
    out = []
    if o["name_on"] and name.strip():
        out.append([" ".join(name.split()), float(o["tsize"]), o["font"]])
    if str(o["line2"]).strip():
        out.append([" ".join(str(o["line2"]).split()), float(o["tsize2"]), o["font2"]])
    return out


def _wrap(text: str, key: str) -> list[str]:
    """Split at the space that makes the two halves most alike in width."""
    words = text.split(" ")
    if len(words) < 2:
        return [text]
    best = min(range(1, len(words)), key=lambda i: abs(text_width_em(" ".join(words[:i]), key)
                                                        - text_width_em(" ".join(words[i:]), key)))
    return [" ".join(words[:best]), " ".join(words[best:])]


def _layout(result: dict, settings: dict, opts, name: str, framed: bool = False) -> dict:
    """Where everything in one piece goes, in inches from the piece's top-left.

    {"w", "h", "ox", "oy" (the shape's top-left), "texts": [(text, size, centre_x, baseline_y, font)],
     "arc": {"segments": [(text, font, size)], "cx", "cy", "radius"} or None,
     "fits": everything is inside the ornament}

    Lettering is always fitted to the ornament: a long name wraps onto two
    lines, then shrinks, so it never runs off the edge."""
    o = _opts(opts)
    sw, sh = size_in(result, settings)
    ow, oh = orn_size(o)
    lines = _wanted_lines(name, o)
    use_arc = bool(lines) and o["place"] == "arc" and o["shape"] == "circle"
    framed = framed or o["border"] == "ornament" or use_arc
    texts, arc, lift = [], None, 0.0  # everything below is relative to the ornament's centre first

    if use_arc:
        first_font = lines[0][2]
        dot = " \u00b7 " if _font(first_font).getBestCmap().get(0xB7) else " - "
        segs = [[t + (dot if i < len(lines) - 1 else ""), f, size] for i, (t, size, f) in enumerate(lines)]
        for _ in range(4):  # the radius depends on the sizes, so settle it
            biggest = max(seg[2] for seg in segs)
            radius = ow / 2 - 0.12 - 0.26 * biggest
            longest = radius * np.deg2rad(200)
            width = sum(text_width_em(t, f) * size for t, f, size in segs)
            if width <= longest:
                break
            for seg in segs:  # everything shrinks together, keeping the two sizes in proportion
                seg[2] = max(0.06, seg[2] * longest / width)
        arc = {"segments": [tuple(seg) for seg in segs], "radius": radius}
        lift = min(0.45 * biggest + 0.04, max(0.0, (oh - sh) / 2))  # the shape moves up to clear the letters
        top = -sh / 2 - lift
    else:
        wrapped = False
        for _ in range(6):  # sizes move the lines, which changes the room each one has
            block = TEXT_GAP + sum(1.2 * size for _, size, _ in lines) if lines else 0.0
            top = -(sh + block) / 2
            y, placed, again = top + sh + TEXT_GAP, [], False
            for i, (t, size, f) in enumerate(lines):
                # the narrower of the room at the top and at the bottom of the letters
                room = max(0.2 * ow, 0.92 * min(_room(o["shape"], ow, oh, y + 0.1 * size),
                                                _room(o["shape"], ow, oh, y + 0.95 * size)))
                width = text_width_em(t, f) * size
                if width > room and not wrapped and i == 0 and o["name_on"] and name.strip() and " " in t:
                    lines[0:1] = [[part, size, f] for part in _wrap(t, f)]  # a long name: two lines
                    wrapped = again = True
                    break
                if width > room:
                    lines[i][1] = size = max(0.06, size * room / width)
                    if wrapped and i < 2:  # the two halves of a name stay the same size
                        lines[0][1] = lines[1][1] = size = min(lines[0][1], lines[1][1])
                    again = True
                placed.append((t, size, 0.0, y + 0.78 * size, f))
                y += 1.2 * size
            if not again:
                break
        texts = placed

    corners = [(-sw / 2, top), (sw / 2, top), (-sw / 2, top + sh), (sw / 2, top + sh)]
    inked = [(x - sw / 2, y + top) for x, y in _shape_points(result, settings)]  # the outline itself, not its box
    for t, size, _, base, f in texts:
        half = text_width_em(t, f) * size / 2
        box = [(-half, base - 0.7 * size), (half, base - 0.7 * size), (-half, base + 0.2 * size), (half, base + 0.2 * size)]
        corners += box
        inked += box
    fits = all(_inside(o["shape"], ow, oh, x, y) for x, y in inked)
    left = min(x for x, _ in corners)
    right = max(x for x, _ in corners)
    bottom = max(y for _, y in corners)

    if framed:  # the piece is the ornament (or bigger, if things stick out)
        w, h = max(ow, right - left), max(oh, 2 * max(-top, bottom))
        cx, cy = w / 2, h / 2
    else:  # the piece hugs what is in it
        sp = max(0.0, float(o["space"]))
        w, h = (right - left) + 2 * sp, (bottom - top) + 2 * sp
        cx, cy = sp - left, sp - top
    return {"w": w, "h": h, "ox": cx - sw / 2, "oy": cy + top, "cx": cx, "cy": cy, "fits": fits,
            "texts": [(t, size, cx + x, cy + base, f) for t, size, x, base, f in texts],
            "arc": dict(arc, cx=cx, cy=cy) if arc else None, "o": o}


def layout(result: dict, settings: dict, opts=None, name: str = ""):
    """(width, height, shape_x, shape_y) of one exported piece, in inches."""
    lay = _layout(result, settings, opts, name)
    return lay["w"], lay["h"], lay["ox"], lay["oy"]


def _piece(name: str, result: dict, settings: dict, lay: dict, x: float, y: float,
           border: bool = True, fill: str = "#000000", text_fill: str = TEXT_FILL) -> str:
    """One laid-out child at (x, y) inches: the border (its own layer, behind),
    the shape, then any lettering (its own layer too)."""
    o, u = lay["o"], UNITS_PER_INCH
    out = ""
    if border and o["border"] == "ornament":
        out = ornament_svg(o["shape"], *orn_size(o), x + lay["cx"], y + lay["cy"], u, f'fill="{BORDER_FILL}"')
        out = out[:-2] + f"><title>{_xml(name)} ornament</title></{out[1:out.index(' ')]}>\n"
    elif border and o["border"] == "rect":
        out = (f'<rect x="{x * u:.2f}" y="{y * u:.2f}" width="{lay["w"] * u:.2f}" height="{lay["h"] * u:.2f}" '
               f'fill="{BORDER_FILL}"><title>{_xml(name)} border</title></rect>\n')
    d = placed_d(result, settings, x + lay["ox"], y + lay["oy"])
    out += f'<path fill="{fill}" fill-rule="evenodd" d="{d}"><title>{_xml(name)}</title></path>\n'
    td = "".join(text_d(t, f, size * u, (x + cx) * u, (y + base) * u) for t, size, cx, base, f in lay["texts"])
    if lay["arc"]:
        a = lay["arc"]
        td += arc_d([(t, f, size * u) for t, f, size in a["segments"]], (x + a["cx"]) * u, (y + a["cy"]) * u, a["radius"] * u)
    if td:
        out += f'<path fill="{text_fill}" d="{td}"><title>{_xml(name)} lettering</title></path>\n'
    return out


def piece_svg(name: str, result: dict, settings: dict, opts, x: float, y: float) -> str:
    return _piece(name, result, settings, _layout(result, settings, opts, name), x, y)


def on_ornament(name: str, result, settings: dict, opts, fill: str = "#000000",
                orn_attrs: str = f'fill="{BORDER_FILL}"') -> dict:
    """The piece as it will sit on the ornament, for the preview and the proof
    sheet: {"w", "h" (inches), "body" (SVG, 72 per inch), "fits"}. result may
    be None (nothing to cut yet): then just the ornament."""
    o, u = _opts(opts), UNITS_PER_INCH
    ow, oh = orn_size(o)
    if not result or not result.get("loops"):
        return {"w": ow, "h": oh, "fits": True,
                "body": ornament_svg(o["shape"], ow, oh, ow / 2, oh / 2, u, orn_attrs) + "\n"}
    lay = _layout(result, settings, o, name, framed=True)
    body = ornament_svg(o["shape"], ow, oh, lay["cx"], lay["cy"], u, orn_attrs) + "\n"
    return {"w": lay["w"], "h": lay["h"], "fits": lay["fits"],
            "body": body + _piece(name, result, settings, lay, 0.0, 0.0, border=False, fill=fill, text_fill=fill)}


def svg_single(result: dict, settings: dict, name: str = "", opts=None) -> str:
    lay = _layout(result, settings, opts, name)
    return _svg(lay["w"], lay["h"], _piece(name, result, settings, lay, 0.0, 0.0))


def svg_sheets(items: list[tuple[str, dict, dict]], sheet_w=11.5, sheet_h=11.5,
               gap=0.2, opts=None) -> list[str]:
    """Pack (name, result, settings) onto as many sheets as needed, in rows."""
    laid = [(name, result, settings, _layout(result, settings, opts, name)) for name, result, settings in items]
    sheets, body, x, y, row_h = [], [], 0.0, 0.0, 0.0
    for name, result, settings, lay in sorted(laid, key=lambda it: -it[3]["h"]):
        w, h = lay["w"], lay["h"]
        if x > 0 and x + w > sheet_w:
            x, y, row_h = 0.0, y + row_h + gap, 0.0
        if y > 0 and y + h > sheet_h:
            sheets.append(_svg(sheet_w, sheet_h, "".join(body)))
            body, x, y, row_h = [], 0.0, 0.0, 0.0
        body.append(_piece(name, result, settings, lay, x, y))
        x, row_h = x + w + gap, max(row_h, h)
    if body:
        sheets.append(_svg(sheet_w, sheet_h, "".join(body)))
    return sheets


def png_cutout(result: dict, settings: dict, opts=None) -> Image.Image:
    """Black shape on transparent (plus the border, if any), dpi set to the
    finished size. Lettering is left out: it exists only as outlines in the SVG."""
    from PIL import ImageDraw

    s = {**DEFAULT_SETTINGS, **settings}
    o = dict(_opts(opts), name_on=False, line2="")
    final = result["final"]
    tilt = float(s.get("tilt_deg") or 0.0)
    if tilt:
        h, w = final.shape
        pad = int(np.hypot(w, h) - min(w, h)) // 2 + 2
        final = cv2.copyMakeBorder(final, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        m = cv2.getRotationMatrix2D((final.shape[1] / 2.0, final.shape[0] / 2.0), -tilt, 1.0)
        final = cv2.warpAffine(final, m, (final.shape[1], final.shape[0]), flags=cv2.INTER_LINEAR)
        final = np.where(final > 127, 255, 0).astype(np.uint8)
    x0, y0, x1, y1 = _bbox(final)
    crop = final[y0:y1, x0:x1]
    if s["flip"]:
        crop = crop[:, ::-1]
    dpi = (y1 - y0) / float(s["height_in"])
    lay = _layout(result, settings, o, "")
    size = (max(1, round(lay["w"] * dpi)), max(1, round(lay["h"] * dpi)))
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    if o["border"] == "rect":
        ImageDraw.Draw(img).rectangle((0, 0, size[0] - 1, size[1] - 1), fill=BORDER_FILL)
    elif o["border"] == "ornament":
        pts = ornament_points(o["shape"], *orn_size(o)) + np.array([lay["cx"], lay["cy"]])
        ImageDraw.Draw(img).polygon([(float(x) * dpi, float(y) * dpi) for x, y in pts], fill=BORDER_FILL)
    alpha = Image.fromarray(np.ascontiguousarray(crop))
    img.paste(Image.new("RGBA", alpha.size, (0, 0, 0, 255)), (round(lay["ox"] * dpi), round(lay["oy"] * dpi)), alpha)
    img.info["dpi"] = (dpi, dpi)
    return img


# ---- what this computer can do ---------------------------------------------
BEST_NEEDS_GB = 10.0  # graphics memory the best model was measured to use while cutting out


def _windows_gpus() -> list[tuple[str, float]]:
    """[(name, dedicated memory in GB)] of the graphics cards in this PC."""
    import winreg

    try:  # the cards that are present now (the registry also remembers removed ones)
        present = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True, text=True, timeout=10, creationflags=0x08000000).stdout.split("\n")
        present = {p.strip() for p in present if p.strip()}
    except Exception:
        present = set()
    found = {}
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as cls:
            for i in range(64):
                try:
                    sub = winreg.EnumKey(cls, i)
                except OSError:
                    break
                try:
                    with winreg.OpenKey(cls, sub) as card:
                        name = winreg.QueryValueEx(card, "DriverDesc")[0]
                        try:
                            mem = winreg.QueryValueEx(card, "HardwareInformation.qwMemorySize")[0]
                        except OSError:
                            mem = winreg.QueryValueEx(card, "HardwareInformation.MemorySize")[0]
                        if isinstance(mem, bytes):
                            mem = int.from_bytes(mem, "little")
                        if not present or name in present:
                            found[name] = max(found.get(name, 0.0), mem / 1024**3)
                except OSError:
                    continue
    except OSError:
        pass
    return list(found.items()) or [(n, 0.0) for n in present]


def _ram_gb() -> float:
    try:
        if sys.platform == "win32":
            import ctypes

            class Mem(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("total", "avail", "tp", "ap", "tv", "av", "aev")]

            m = Mem()
            m.length = ctypes.sizeof(m)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.total / 1024**3
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except Exception:
        return 0.0


@lru_cache(maxsize=1)
def hardware() -> dict:
    """{"gpu": name or None, "vram_gb", "ram_gb", "gpu_runtime": the cut-out
    models can use the graphics card here}. Looked up once."""
    gpus = []
    try:
        if sys.platform == "win32":
            gpus = _windows_gpus()
        else:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            gpus = [(n.strip(), float(m) / 1024) for n, m in (line.split(",") for line in out.splitlines() if "," in line)]
    except Exception:
        pass
    name, vram = max(gpus, key=lambda g: g[1], default=(None, 0.0))
    try:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
    except Exception:
        providers = []
    return {"gpu": name, "vram_gb": round(vram, 1), "ram_gb": round(_ram_gb(), 1),
            "gpu_runtime": any(p in providers for p in ("DmlExecutionProvider", "CUDAExecutionProvider"))}


def advice(hw: dict, best_ran: bool = False, best_failed: bool = False) -> dict:
    """Which method suits this computer: {"recommend": best | fast,
    "best": yes | maybe | no (can the best method run here), "why": one sentence}."""
    gpu, vram = hw.get("gpu"), hw.get("vram_gb") or 0.0
    card = f"the {gpu} ({vram:g} GB)" if gpu and vram else (f"the {gpu}" if gpu else "this computer")
    if best_ran and not best_failed:
        return {"recommend": "best", "best": "yes", "why": f"Best is recommended: it has run on {card}."}
    if best_failed:
        return {"recommend": "fast", "best": "no",
                "why": "Fast is recommended: Best could not run on this computer the last time it was tried."}
    if hw.get("gpu_runtime") and vram >= BEST_NEEDS_GB:
        return {"recommend": "best", "best": "yes",
                "why": f"Best is recommended: {card} has enough graphics memory for it (it uses about {BEST_NEEDS_GB:g} GB)."}
    if hw.get("gpu_runtime") and vram >= 6:
        return {"recommend": "best", "best": "maybe",
                "why": f"Best is worth trying: it likes about {BEST_NEEDS_GB:g} GB of graphics memory and {card} has less, "
                       "so it may be slow. If it cannot run, the app switches to Fast and tells you."}
    if hw.get("gpu_runtime"):
        why = (f"{card} has too little graphics memory for Best (it uses about {BEST_NEEDS_GB:g} GB)" if gpu
               else "no graphics card was found")
        return {"recommend": "fast", "best": "no", "why": f"Fast is recommended: {why}."}
    if (hw.get("ram_gb") or 0) >= 32:
        return {"recommend": "fast", "best": "maybe",
                "why": "Fast is recommended: here Best would run on the processor, which takes minutes a photo "
                       "and about 16 GB of free memory."}
    return {"recommend": "fast", "best": "no",
            "why": "Fast is recommended: Best needs a graphics card or a lot of memory, and this computer has neither."}
