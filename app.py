"""Silhouette Maker: local server. Listens on this computer only (127.0.0.1)."""
from __future__ import annotations

import io
import json
import math
import re
import shutil
import sys
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

__version__ = "0.2.0"
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
    a = request.args
    border = a.get("border", "none")
    return {"border": border if border in ("none", "rect", "circle") else "none",
            "space": _num(a.get("space"), 0.0, 6.0, 0.0),
            "orn": _num(a.get("orn"), 0.5, 24.0, 3.0)}


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
    name = Path(f.filename or "photo").stem[:60]
    meta = {"name": name, "w": img.width, "h": img.height, "added": time.time(), "settings": settings}
    _save_meta(d, meta)
    return jsonify(dict(_public(d, meta), note=note))


@app.get("/api/items/<item_id>/photo")
def get_photo(item_id):
    return send_file(_dir(item_id) / "photo.jpg", mimetype="image/jpeg")


@app.get("/api/items/<item_id>/thumb")
def get_thumb(item_id):
    d = _dir(item_id)
    t = d / "thumb.jpg"
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
    img, mask = engine.turn(Image.open(d / "photo.jpg").convert("RGB"), _mask(d), -turns)
    img.save(d / "photo.jpg", quality=95)
    Image.fromarray(mask).save(d / "mask.png")
    if (d / "start.png").exists():  # the starting copy turns with the photo
        start = np.rot90(np.asarray(Image.open(d / "start.png").convert("L")), -turns % 4)
        Image.fromarray(np.ascontiguousarray(start)).save(d / "start.png")
    (d / "thumb.jpg").unlink(missing_ok=True)
    meta.update(w=img.width, h=img.height, rev=meta.get("rev", 0) + 1)
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
    s = body.get("settings")
    s = s if isinstance(s, dict) else {}
    cur = meta["settings"]
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
    _save_meta(d, meta)
    return jsonify(_public(d, meta))


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


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
    print(f"Silhouette Maker is running at http://127.0.0.1:{PORT}/  (close this window to stop)")
    app.run(host="127.0.0.1", port=PORT, threaded=True)
