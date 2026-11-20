"""Silhouette Maker: local server. Listens on this computer only (127.0.0.1)."""
from __future__ import annotations

import html
import io
import json
import math
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image

import engine

__version__ = "0.4.0"
ROOT = Path(__file__).resolve().parent


def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


DATA = Path(_arg("--data", ROOT / "data"))  # --data and --port are for testing
PORT = int(_arg("--port", 8777))
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
RESULTS_KEPT = 6  # full outlines (with their bitmap) kept in memory, for exports
_cache: OrderedDict[str, tuple] = OrderedDict()  # id -> (key, outline result)
_outlines: dict[str, tuple] = {}  # id -> (key, what the page needs of the outline)
_cache_lock = threading.Lock()
CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
       "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")


@app.before_request
def _local_only():
    # Only this computer's own browser page may talk to the app.
    if request.host not in ALLOWED_HOSTS:
        abort(403)
    if request.path == "/api/restore":  # a backup of a whole class is bigger than one photo
        request.max_content_length = 4 * 1024**3
    origin = request.headers.get("Origin")
    if origin and origin.split("://", 1)[-1] not in ALLOWED_HOSTS:
        abort(403)
    # a page on another site (or another local port) may not call the API
    if request.path.startswith("/api/") and request.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site"):
        abort(403)


@app.after_request
def _headers(resp):
    # a photo never changes once added, so the gallery may keep its pictures
    keep = request.path.endswith(("/thumb", "/photo")) and resp.status_code == 200
    resp.headers["Cache-Control"] = "private, max-age=86400" if keep else "no-store"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = CSP
    return resp


@app.errorhandler(413)
def _too_big(_e):
    return jsonify({"error": "That file is too big (the limit is 80 MB)."}), 413


def _num(value, lo: float, hi: float, default: float) -> float:
    """A finite number within [lo, hi]; the default for anything else."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return min(hi, max(lo, v)) if math.isfinite(v) else default


def _dir(item_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", item_id):
        abort(404)
    d = DATA / item_id
    if not d.is_dir():
        abort(404)
    return d


def _meta(d: Path) -> dict:
    return json.loads((d / "meta.json").read_text(encoding="utf-8"))


def _save_meta(d: Path, meta: dict) -> None:
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    tmp.replace(d / "meta.json")


def _mask(d: Path) -> np.ndarray:
    return np.asarray(Image.open(d / "mask.png").convert("L"))


def _key(d: Path, meta: dict) -> list:
    """Changes whenever the shape, its settings or the outline code changes."""
    return [(d / "mask.png").stat().st_mtime_ns, json.dumps(meta["settings"], sort_keys=True), engine.REV]


def _result(d: Path, meta: dict) -> dict:
    """Full outline for the item's current mask + settings (a few are kept in memory)."""
    key = _key(d, meta)
    with _cache_lock:
        hit = _cache.get(d.name)
        if hit and hit[0] == key:
            _cache.move_to_end(d.name)
            return hit[1]
    res = engine.outline(_mask(d), meta["settings"])
    with _cache_lock:
        _cache[d.name] = (key, res)
        _cache.move_to_end(d.name)
        while len(_cache) > RESULTS_KEPT:
            _cache.popitem(last=False)
    return res


def _outline_json(d: Path, meta: dict) -> dict:
    """What the page draws. Saved next to the photo, so opening the app with a
    full class does not trace every child again."""
    key = _key(d, meta)
    hit = _outlines.get(d.name)
    if hit and hit[0] == key:
        return hit[1]
    saved = d / "outline.json"
    out = None
    try:
        got = json.loads(saved.read_text(encoding="utf-8"))
        if got["key"] == key:
            out = got["outline"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if out is None:
        res = _result(d, meta)
        if not res["loops"]:
            out = {"d": "", "bbox": None, "width_in": 0, "height_in": 0}
        else:
            w, h = engine.size_in(res, meta["settings"])
            out = {"d": engine.path_d(res["loops"], nd=1), "bbox": res["bbox"],
                   "width_in": round(w, 3), "height_in": round(h, 3)}
        tmp = d / "outline.json.tmp"
        tmp.write_text(json.dumps({"key": key, "outline": out}), encoding="utf-8")
        tmp.replace(saved)
    _outlines[d.name] = (key, out)
    return out


def _added(d: Path, meta: dict) -> float:
    return meta.get("added") or d.stat().st_ctime  # older photos have no saved time


def _items() -> list[tuple[Path, dict]]:
    """Every photo, oldest first."""
    out = []
    if DATA.is_dir():
        for d in DATA.iterdir():
            if (d / "meta.json").exists() and (d / "mask.png").exists():
                out.append((d, _meta(d)))
    return sorted(out, key=lambda dm: _added(*dm))


def _public(d: Path, meta: dict) -> dict:
    return {"id": d.name, "name": meta["name"], "w": meta["w"], "h": meta["h"],
            "done": bool(meta.get("done")), "added": _added(d, meta), "rev": meta.get("rev", 0),
            "photo": (d / "photo.jpg").exists(),
            "settings": meta["settings"], "outline": _outline_json(d, meta)}


def _chosen():
    """(name, outline, settings) for the ids in ?ids=a,b (or every photo), skipping empty shapes."""
    ids = [i for i in request.args.get("ids", "").split(",") if i]
    found = [(d, _meta(d)) for d in map(_dir, ids)] if ids else _items()
    out = []
    for d, meta in found:
        if _outline_json(d, meta)["bbox"]:  # cheap check first: most are saved
            out.append((meta["name"], _result(d, meta), meta["settings"]))
    return out


def _cut_out(img, quality: str):
    """(mask, note). The note says so when the best model could not run here."""
    mask, used = engine.segment(img, quality)
    fell_back = quality != "fast" and used == engine.MODELS["fast"]
    return mask, ("The best cut-out model could not run on this computer, so the fast one was used."
                  if fell_back else "")


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\- ]+", "", name).strip()[:60] or "silhouette"


def _export_opts() -> dict:
    """What goes around and under the shape, from the page's query string."""
    a = request.args
    border = {"circle": "ornament"}.get(a.get("border"), a.get("border"))  # "circle": older links
    return {"border": border if border in ("none", "rect", "ornament") else "none",
            "space": _num(a.get("space"), 0.0, 6.0, 0.0),
            "shape": a.get("shape") if a.get("shape") in engine.ORNAMENTS else "circle",
            "orn": _num(a.get("orn"), 0.5, 24.0, 3.0),
            "orn_h": _num(a.get("orn_h"), 0.0, 24.0, 0.0),
            "name_on": a.get("name_on") == "1" or a.get("text") in ("name", "both"),
            "line2": (a.get("line2") or "")[:40],
            "place": "arc" if a.get("place") == "arc" else "below",
            "font": (a.get("font") or "serif")[:120],  # the engine falls back when it is not a font here
            "tsize": _num(a.get("tsize"), 0.08, 2.0, 0.3),
            "font2": (a.get("font2") or "")[:120] or None,  # none: the same as the name
            "tsize2": _num(a.get("tsize2"), 0.08, 2.0, 0.0) or None}


def _colour(value, default: str) -> str:
    return value if isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) else default


def _name_from_file(filename: str) -> str:
    """A photo called Ava.jpg is Ava. A camera's own file name (IMG_2041,
    PXL_20261119_190204) is not a name: leave it blank for the user to fill in."""
    stem = Path(filename or "").stem.strip()[:60]
    letters = re.sub(r"(?i)^(img|pxl|dsc|dscn|dscf|mvimg|photo|image|picture|screenshot|scan|p|wp)(?=[\W_\d]|$)", "", stem)
    return "" if not re.search(r"[^\W\d_]", letters) else stem


NO_PHOTO = "The photo was deleted, so it cannot be cut out again."


def _merge_settings(cur: dict, s) -> dict:
    """Copy the valid parts of s (from the page, or a backup) into cur."""
    s = s if isinstance(s, dict) else {}
    for k, lo, hi in (("height_in", 0.25, 24.0), ("smooth", 0.0, 10.0),
                      ("tilt_deg", -45.0, 45.0), ("detail_mm", 0.0, 3.0)):
        if k in s:
            cur[k] = _num(s[k], lo, hi, cur.get(k, engine.DEFAULT_SETTINGS[k]))
    for k in ("flip", "cut_on"):
        if k in s:
            cur[k] = bool(s[k])
    cut = s.get("cut")
    if isinstance(cut, list) and len(cut) == 3 and all(isinstance(p, list) and len(p) == 2 for p in cut):
        # the middle point is a pull point and may sit a little outside the photo
        cur["cut"] = [[_num(p[0], -2.0, 3.0, 0.5), _num(p[1], -2.0, 3.0, 0.9)] for p in cut]
    return cur


@app.get("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.get("/api/items")
def list_items():
    return jsonify([_public(d, meta) for d, meta in _items()])


@app.post("/api/items")
def add_item():
    f = request.files.get("photo")
    if not f:
        abort(400)
    try:
        img = engine.load_photo(f.stream)
    except Exception:
        return jsonify({"error": "That file could not be opened as a photo."}), 400
    quality = request.form.get("quality", "best")
    try:
        mask, note = _cut_out(img, quality)
    except Exception as e:  # model missing and no internet, out of memory, ...
        return jsonify({"error": f"The cut-out model could not run: {str(e)[:200]}"}), 500
    img, mask = engine.turn(img, mask, engine.upright_turns(mask))  # upside-down or sideways photos
    item_id = uuid.uuid4().hex[:12]
    d = DATA / item_id
    d.mkdir(parents=True)
    img.save(d / "photo.jpg", quality=92)
    Image.fromarray(mask).save(d / "mask.png")
    Image.fromarray(mask).save(d / "start.png")  # what "reset" goes back to
    settings = dict(engine.DEFAULT_SETTINGS, cut=engine.default_cut(mask))
    settings["height_in"] = _num(request.form.get("height_in"), 0.25, 24.0, settings["height_in"])
    name = _name_from_file(f.filename)
    meta = {"name": name, "w": img.width, "h": img.height, "added": time.time(), "settings": settings}
    _save_meta(d, meta)
    return jsonify(dict(_public(d, meta), note=note))


@app.get("/api/items/<item_id>/photo")
def get_photo(item_id):
    f = _dir(item_id) / "photo.jpg"
    if not f.exists():
        abort(404)
    return send_file(f, mimetype="image/jpeg")


@app.get("/api/items/<item_id>/thumb")
def get_thumb(item_id):
    d = _dir(item_id)
    t = d / "thumb.jpg"
    if not (d / "photo.jpg").exists():
        abort(404)
    if not t.exists():
        img = Image.open(d / "photo.jpg")
        img.thumbnail((420, 420), Image.LANCZOS)
        img.save(t, quality=85)
    return send_file(t, mimetype="image/jpeg")


@app.get("/api/items/<item_id>/mask")
def get_mask(item_id):
    return send_file(_dir(item_id) / "mask.png", mimetype="image/png")


@app.put("/api/items/<item_id>/mask")
def put_mask(item_id):
    d = _dir(item_id)
    meta = _meta(d)
    try:
        img = Image.open(io.BytesIO(request.get_data())).convert("L")
    except Exception:
        abort(400)
    if img.size != (meta["w"], meta["h"]):
        abort(400)
    arr = np.where(np.asarray(img) > 127, 255, 0).astype(np.uint8)
    Image.fromarray(arr).save(d / "mask.png")
    return jsonify(_outline_json(d, meta))


@app.post("/api/items/<item_id>/redo")
def redo_cutout(item_id):
    """Run the model again (throws away brush edits)."""
    d = _dir(item_id)
    meta = _meta(d)
    if not (d / "photo.jpg").exists():
        return jsonify({"error": NO_PHOTO}), 409
    quality = (request.get_json(silent=True) or {}).get("quality", "best")
    try:
        mask, note = _cut_out(Image.open(d / "photo.jpg").convert("RGB"), quality)
    except Exception as e:
        return jsonify({"error": f"The cut-out model could not run: {str(e)[:200]}"}), 500
    Image.fromarray(mask).save(d / "mask.png")
    Image.fromarray(mask).save(d / "start.png")  # a new cut-out is the new starting point
    return jsonify(dict(_outline_json(d, meta), note=note))


@app.post("/api/items/<item_id>/reset")
def reset_item(item_id):
    """Back to the automatic cut-out and the starting settings (the size is kept)."""
    d = _dir(item_id)
    meta = _meta(d)
    note = ""
    if (d / "start.png").exists():
        mask = np.asarray(Image.open(d / "start.png").convert("L"))
    elif not (d / "photo.jpg").exists():
        return jsonify({"error": NO_PHOTO}), 409
    else:  # added before the app kept a starting copy: make the cut-out again
        quality = (request.get_json(silent=True) or {}).get("quality", "best")
        try:
            mask, note = _cut_out(Image.open(d / "photo.jpg").convert("RGB"), quality)
        except Exception as e:
            return jsonify({"error": f"The cut-out model could not run: {str(e)[:200]}"}), 500
        Image.fromarray(mask).save(d / "start.png")
    Image.fromarray(mask).save(d / "mask.png")
    meta["settings"] = dict(engine.DEFAULT_SETTINGS, cut=engine.default_cut(mask),
                            height_in=meta["settings"].get("height_in", engine.DEFAULT_SETTINGS["height_in"]))
    _save_meta(d, meta)
    return jsonify(dict(_public(d, meta), note=note))


@app.post("/api/items/<item_id>/rotate")
def rotate_item(item_id):
    """Turn the photo and its shape a quarter turn (turns: 1 = right, -1 = left, 2 = upside down)."""
    d = _dir(item_id)
    meta = _meta(d)
    turns = int(_num((request.get_json(silent=True) or {}).get("turns"), -3, 3, 1))
    if (d / "photo.jpg").exists():
        img, mask = engine.turn(Image.open(d / "photo.jpg").convert("RGB"), _mask(d), -turns)
        img.save(d / "photo.jpg", quality=95)
    else:  # the photo was deleted: the shape still turns
        mask = np.ascontiguousarray(np.rot90(_mask(d), -turns % 4))
    Image.fromarray(mask).save(d / "mask.png")
    if (d / "start.png").exists():  # the starting copy turns with the photo
        start = np.rot90(np.asarray(Image.open(d / "start.png").convert("L")), -turns % 4)
        Image.fromarray(np.ascontiguousarray(start)).save(d / "start.png")
    (d / "thumb.jpg").unlink(missing_ok=True)
    meta.update(w=mask.shape[1], h=mask.shape[0], rev=meta.get("rev", 0) + 1)
    meta["settings"]["cut"] = engine.default_cut(mask)
    _save_meta(d, meta)
    return jsonify(_public(d, meta))


@app.patch("/api/items/<item_id>")
def patch_item(item_id):
    d = _dir(item_id)
    meta = _meta(d)
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        abort(400)
    if "name" in body:
        meta["name"] = str(body["name"])[:60]
    if "done" in body:
        meta["done"] = bool(body["done"])
    _merge_settings(meta["settings"], body.get("settings"))
    _save_meta(d, meta)
    return jsonify(_public(d, meta))


@app.post("/api/height")
def height_for_all():
    """Give every photo the same shape height, so the ornaments match."""
    h = _num((request.get_json(silent=True) or {}).get("height_in"), 0.25, 24.0, -1.0)
    if h < 0:
        abort(400)
    out = []
    for d, meta in _items():
        if meta["settings"].get("height_in") != h:
            meta["settings"]["height_in"] = h
            _save_meta(d, meta)
        out.append(_public(d, meta))
    return jsonify(out)


@app.delete("/api/items/<item_id>")
def delete_item(item_id):
    d = _dir(item_id)
    shutil.rmtree(d)
    _cache.pop(item_id, None)
    _outlines.pop(item_id, None)
    return jsonify({"ok": True})


@app.get("/api/items/<item_id>/export.<ext>")
def export_item(item_id, ext):
    d = _dir(item_id)
    meta = _meta(d)
    res = _result(d, meta)
    if not res["loops"]:
        abort(409)
    name = _safe_name(meta["name"])
    if ext == "svg":
        data = engine.svg_single(res, meta["settings"], meta["name"], _export_opts()).encode("utf-8")
        return send_file(io.BytesIO(data), mimetype="image/svg+xml", as_attachment=True,
                         download_name=f"{name}.svg")
    if ext == "png":
        img = engine.png_cutout(res, meta["settings"], _export_opts())
        buf = io.BytesIO()
        img.save(buf, "PNG", dpi=img.info["dpi"])
        buf.seek(0)
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name=f"{name}.png")
    abort(404)


@app.get("/api/sheet")
def export_sheet():
    w = _num(request.args.get("w"), 2.0, 24.0, 11.5)
    h = _num(request.args.get("h"), 2.0, 24.0, 11.5)
    items = _chosen()
    if not items:
        abort(409)
    sheets = engine.svg_sheets(items, w, h, opts=_export_opts())
    if len(sheets) == 1:
        return send_file(io.BytesIO(sheets[0].encode("utf-8")), mimetype="image/svg+xml",
                         as_attachment=True, download_name="silhouettes-sheet.svg")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, s in enumerate(sheets, 1):
            z.writestr(f"silhouettes-sheet-{i}.svg", s)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="silhouettes-sheets.zip")


@app.get("/api/svgs.zip")
def export_zip():
    items = _chosen()
    if not items:
        abort(409)
    buf, used, opts = io.BytesIO(), {}, _export_opts()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, res, settings in items:
            base = _safe_name(name)
            used[base.lower()] = n = used.get(base.lower(), 0) + 1
            z.writestr(f"{base}{'' if n == 1 else f' ({n})'}.svg", engine.svg_single(res, settings, name, opts))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="silhouettes.zip")


@app.post("/api/items/<item_id>/forget")
def forget_photo(item_id):
    """Delete the original photo and keep the silhouette."""
    d = _dir(item_id)
    meta = _meta(d)
    for f in ("photo.jpg", "thumb.jpg"):
        (d / f).unlink(missing_ok=True)
    return jsonify(_public(d, meta))


@app.get("/api/fonts")
def list_fonts():
    """Every font on this computer that can letter a name, and what the
    choices of earlier versions map to."""
    return jsonify({"fonts": engine.fonts(), "default": engine.resolve_font("serif"),
                    "old": {k: engine.resolve_font(k) for k in ("serif", "sans", "bold", "script")}})


PROOF_CSS = """
body{font:14px/1.4 "Segoe UI",system-ui,sans-serif;color:#111;margin:0.4in}
header{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin-bottom:0.2in}
h1{font-size:18px;margin:0}
.rule{width:2in;height:0.12in;background:#111}
.note{color:#444;font-size:12px}
main{display:flex;flex-wrap:wrap;gap:0.15in}
figure{margin:0;text-align:center;break-inside:avoid}
figure svg{display:block}
figcaption{font-size:12px}
@media print{button{display:none}body{margin:0}}
@page{margin:0.5in}
"""


@app.get("/api/items/<item_id>/preview")
def preview(item_id):
    """The child on the ornament, exactly as the files will have it."""
    d = _dir(item_id)
    meta = _meta(d)
    a = request.args
    name = a.get("name", meta["name"])[:60]
    fg, bg = _colour(a.get("fg"), "#111111"), _colour(a.get("bg"), engine.BORDER_FILL)
    res = _result(d, meta) if _outline_json(d, meta)["bbox"] else None
    o = _export_opts()
    on = engine.on_ornament(name, res, meta["settings"], o, fill=fg,
                            orn_attrs=f'fill="{bg}" stroke="#00000033" stroke-width="1"')
    u, pad = engine.UNITS_PER_INCH, 0.12
    ow, oh = engine.orn_size(engine._opts(o))
    hole = ""
    if o["shape"] not in ("heart", "star"):  # where the ribbon goes
        hole = (f'<circle cx="{on["w"] / 2 * u:.2f}" cy="{(on["h"] / 2 - oh / 2 + 0.06 * oh) * u:.2f}" '
                f'r="{0.02 * ow * u:.2f}" fill="#00000066"/>')
    return jsonify({"viewBox": f"{-pad * u:.2f} {-pad * u:.2f} {(on['w'] + 2 * pad) * u:.2f} {(on['h'] + 2 * pad) * u:.2f}",
                    "html": on["body"] + hole, "fits": on["fits"]})


@app.get("/proof")
def proof_sheet():
    """A page to print on plain paper: every shape at its real size, on its ornament."""
    items = _chosen()
    if not items:
        abort(409)
    o = _export_opts()
    u, figs = engine.UNITS_PER_INCH, []
    dashed = 'fill="none" stroke="#888" stroke-width="0.75" stroke-dasharray="4 3"'
    for name, res, settings in items:
        on = engine.on_ornament(name, res, settings, o, orn_attrs=dashed)
        tall = engine.size_in(res, settings)[1]
        figs.append(f'<figure><svg xmlns="http://www.w3.org/2000/svg" width="{on["w"]:.4f}in" height="{on["h"]:.4f}in" '
                    f'viewBox="0 0 {on["w"] * u:.2f} {on["h"] * u:.2f}">{on["body"]}</svg>'
                    f'<figcaption>{html.escape(name) or "Unnamed"} &middot; {tall:.2f} in tall'
                    f'{"" if on["fits"] else " &middot; <b>does not fit</b>"}</figcaption></figure>')
    ow, oh = engine.orn_size(engine._opts(o))
    kind = engine.ORNAMENTS[o["shape"]].lower()
    size = f"{ow:g} in" if ow == oh else f"{ow:g} × {oh:g} in"
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Silhouette proof sheet</title>'
            f'<style>{PROOF_CSS}</style></head><body><header><h1>Proof sheet</h1><button id="print">Print</button>'
            f'<div class="rule"></div><div class="note">Print at 100% (&ldquo;Actual size&rdquo;, not &ldquo;Fit&rdquo;). '
            f'The black bar should measure exactly 2 inches. The dashed outline is the {size} {kind} ornament.</div></header>'
            f'<main>{"".join(figs)}</main><script src="/static/proof.js"></script></body></html>')
    return page


BACKUP_FILES = ("meta.json", "mask.png", "start.png", "photo.jpg")


@app.get("/api/backup")
def backup():
    """Everything, as one zip: photos, shapes and settings."""
    buf = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # the pictures are already compressed
        for d, _meta_ in _items():
            for f in BACKUP_FILES:
                if (d / f).exists():
                    z.write(d / f, f"{d.name}/{f}")
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"silhouette-maker-backup-{time.strftime('%Y-%m-%d')}.zip")


@app.post("/api/restore")
def restore():
    """Add the photos from a backup zip. Ones already here are left alone."""
    f = request.files.get("backup")
    if not f:
        abort(400)
    try:
        z = zipfile.ZipFile(f.stream)
    except zipfile.BadZipFile:
        return jsonify({"error": "That file is not a Silhouette Maker backup."}), 400
    groups: dict[str, dict] = {}
    for info in z.infolist():  # only our own file names, so nothing can land outside the data folder
        m = re.fullmatch(r"([0-9a-f]{12})/(meta\.json|mask\.png|start\.png|photo\.jpg)", info.filename)
        if m and info.file_size <= 60 * 1024 * 1024:
            groups.setdefault(m.group(1), {})[m.group(2)] = info
    added = skipped = 0
    DATA.mkdir(exist_ok=True)
    for item_id, files in groups.items():
        if "meta.json" not in files or "mask.png" not in files:
            continue
        if (DATA / item_id).exists():
            skipped += 1
            continue
        try:
            old = json.loads(z.read(files["meta.json"]))
            mask = Image.open(io.BytesIO(z.read(files["mask.png"]))).convert("L")
            arr = np.where(np.asarray(mask) > 127, 255, 0).astype(np.uint8)
            meta = {"name": str(old["name"])[:60], "w": mask.width, "h": mask.height,
                    "done": bool(old.get("done")), "rev": int(_num(old.get("rev"), 0, 1e9, 0)),
                    "added": _num(old.get("added"), 0, 1e11, time.time()),
                    "settings": _merge_settings(dict(engine.DEFAULT_SETTINGS, cut=engine.default_cut(arr)),
                                                old.get("settings"))}
            extra = {}
            for name in ("start.png", "photo.jpg"):
                if name in files:
                    img = Image.open(io.BytesIO(z.read(files[name])))
                    if img.size == mask.size:  # a picture of the wrong size is not this child's
                        extra[name] = img.convert("L" if name == "start.png" else "RGB")
        except Exception:
            continue  # not one of ours, or damaged: skip it
        tmp = DATA / f".restore-{item_id}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        Image.fromarray(arr).save(tmp / "mask.png")
        for name, img in extra.items():
            img.save(tmp / name, **({"quality": 95} if name.endswith(".jpg") else {}))
        _save_meta(tmp, meta)
        tmp.rename(DATA / item_id)
        added += 1
    return jsonify({"added": added, "skipped": skipped})


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
    threading.Thread(target=engine.fonts, daemon=True).start()  # look through the fonts while the page opens
    print(f"Silhouette Maker is running at http://127.0.0.1:{PORT}/  (close this window to stop)")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
