"""Silhouette engine: local cut-out, cleanup, smooth outline, cut files.

Everything here runs on this computer. The only network use is rembg's
one-time download of the model file itself; photos are never sent anywhere.
"""
from __future__ import annotations

import gc
import os
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
FONTS = {  # key: (label, CSS for the on-screen preview, font files tried in order)
    "serif": ("Serif", "font-family:Georgia,'Times New Roman',serif",
              ["georgia.ttf", "times.ttf", "DejaVuSerif.ttf", "LiberationSerif-Regular.ttf", "Georgia.ttf"]),
    "sans": ("Plain", "font-family:Arial,Helvetica,sans-serif",
             ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"]),
    "bold": ("Bold", "font-family:Arial,Helvetica,sans-serif;font-weight:bold",
             ["arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"]),
    "script": ("Handwriting", "font-family:'Segoe Script','Comic Sans MS',cursive",
               ["segoesc.ttf", "segoepr.ttf", "comic.ttf", "Comic Sans MS.ttf"]),
}


def _font_dirs() -> list[Path]:
    dirs = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts", Path("/usr/share/fonts"),
            Path("/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path.home() / ".fonts"]
    return [d for d in dirs if d.is_dir()]


@lru_cache(maxsize=None)
def font_file(key: str):
    """Path of the font for this choice, or None when it is not on this computer."""
    if key not in FONTS:
        return None
    try:
        import fontTools  # noqa: F401
    except ImportError:
        return None
    names = FONTS[key][2]
    for d in _font_dirs():
        for n in names:
            if (d / n).is_file():
                return d / n
    for d in _font_dirs():  # Linux keeps fonts in sub-folders
        for n in names:
            hit = next(d.rglob(n), None)
            if hit:
                return hit
    return None


def fonts() -> list[dict]:
    """The lettering choices that work here."""
    return [{"key": k, "label": v[0], "css": v[1]} for k, v in FONTS.items() if font_file(k)]


@lru_cache(maxsize=None)
def _font(key: str):
    from fontTools.ttLib import TTFont

    return TTFont(str(font_file(key)), lazy=True)


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


# ---- pieces ----------------------------------------------------------------
def _opts(opts) -> dict:
    """border: none | rect | circle; space: blank inches around; orn: ornament
    diameter (for the circle); text: none | name | both (name + line2);
    font: a key of FONTS; tsize: letter size in inches."""
    return {"border": "none", "space": 0.0, "orn": 3.0,
            "text": "none", "line2": "", "font": "serif", "tsize": 0.3, **(opts or {})}


def _lines(name: str, o: dict) -> list[tuple[str, float]]:
    """[(text, letter size in inches)] to put under the shape."""
    if o["text"] not in ("name", "both") or not font_file(o["font"]):
        return []
    size, out = float(o["tsize"]), []
    if name.strip():
        out.append((name.strip(), size))
    if o["text"] == "both" and str(o["line2"]).strip():
        out.append((str(o["line2"]).strip(), size * 0.8))
    return out


def _layout(result: dict, settings: dict, opts, name: str):
    """(width, height, shape_x, shape_y, [(text, size, centre_x, baseline_y)]) in inches."""
    o = _opts(opts)
    sw, sh = size_in(result, settings)
    lines = _lines(name, o)
    block = TEXT_GAP + sum(1.2 * size for _, size in lines) if lines else 0.0
    cw = max([sw] + [text_width_em(t, o["font"]) * size for t, size in lines])
    ch = sh + block
    if o["border"] == "circle":
        w = h = max(float(o["orn"]), cw, ch)
        left, top = (w - cw) / 2, (h - ch) / 2
    else:
        sp = max(0.0, float(o["space"]))
        w, h, left, top = cw + 2 * sp, ch + 2 * sp, sp, sp
    ox, oy = left + (cw - sw) / 2, top
    texts, y = [], oy + sh + TEXT_GAP
    for t, size in lines:
        texts.append((t, size, left + cw / 2, y + 0.78 * size))
        y += 1.2 * size
    return w, h, ox, oy, texts


def layout(result: dict, settings: dict, opts=None, name: str = ""):
    """(width, height, shape_x, shape_y) of one exported piece, in inches."""
    return _layout(result, settings, opts, name)[:4]


def piece_svg(name: str, result: dict, settings: dict, opts, x: float, y: float, border: bool = True) -> str:
    """One child at (x, y) inches: the border (its own layer, behind), the
    shape, then any lettering (its own layer too)."""
    o, u = _opts(opts), UNITS_PER_INCH
    w, h, ox, oy, texts = _layout(result, settings, o, name)
    out = ""
    if border and o["border"] == "circle":
        out = (f'<circle cx="{(x + w / 2) * u:.2f}" cy="{(y + h / 2) * u:.2f}" r="{w / 2 * u:.2f}" '
               f'fill="{BORDER_FILL}"><title>{_xml(name)} border</title></circle>\n')
    elif border and o["border"] == "rect":
        out = (f'<rect x="{x * u:.2f}" y="{y * u:.2f}" width="{w * u:.2f}" height="{h * u:.2f}" '
               f'fill="{BORDER_FILL}"><title>{_xml(name)} border</title></rect>\n')
    d = placed_d(result, settings, x + ox, y + oy)
    out += f'<path fill="#000000" fill-rule="evenodd" d="{d}"><title>{_xml(name)}</title></path>\n'
    if texts:
        td = "".join(text_d(t, o["font"], size * u, (x + cx) * u, (y + base) * u) for t, size, cx, base in texts)
        out += f'<path fill="{TEXT_FILL}" d="{td}"><title>{_xml(name)} lettering</title></path>\n'
    return out


def svg_single(result: dict, settings: dict, name: str = "", opts=None) -> str:
    w, h, _, _ = layout(result, settings, opts, name)
    return _svg(w, h, piece_svg(name, result, settings, opts, 0.0, 0.0))


def svg_sheets(items: list[tuple[str, dict, dict]], sheet_w=11.5, sheet_h=11.5,
               gap=0.2, opts=None) -> list[str]:
    """Pack (name, result, settings) onto as many sheets as needed, in rows."""
    sheets, body, x, y, row_h = [], [], 0.0, 0.0, 0.0
    for name, result, settings in sorted(items, key=lambda it: -layout(it[1], it[2], opts, it[0])[1]):
        w, h, _, _ = layout(result, settings, opts, name)
        if x > 0 and x + w > sheet_w:
            x, y, row_h = 0.0, y + row_h + gap, 0.0
        if y > 0 and y + h > sheet_h:
            sheets.append(_svg(sheet_w, sheet_h, "".join(body)))
            body, x, y, row_h = [], 0.0, 0.0, 0.0
        body.append(piece_svg(name, result, settings, opts, x, y))
        x, row_h = x + w + gap, max(row_h, h)
    if body:
        sheets.append(_svg(sheet_w, sheet_h, "".join(body)))
    return sheets


def png_cutout(result: dict, settings: dict, opts=None) -> Image.Image:
    """Black shape on transparent (plus the border, if any), dpi set to the
    finished size. Lettering is left out: it exists only as outlines in the SVG."""
    from PIL import ImageDraw

    s = {**DEFAULT_SETTINGS, **settings}
    o = dict(_opts(opts), text="none")
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
    w_in, h_in, ox, oy = layout(result, settings, o)
    size = (max(1, round(w_in * dpi)), max(1, round(h_in * dpi)))
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    if o["border"] in ("rect", "circle"):
        draw = ImageDraw.Draw(img)
        box = (0, 0, size[0] - 1, size[1] - 1)
        (draw.ellipse if o["border"] == "circle" else draw.rectangle)(box, fill=BORDER_FILL)
    alpha = Image.fromarray(np.ascontiguousarray(crop))
    img.paste(Image.new("RGBA", alpha.size, (0, 0, 0, 255)), (round(ox * dpi), round(oy * dpi)), alpha)
    img.info["dpi"] = (dpi, dpi)
    return img
