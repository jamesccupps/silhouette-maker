"""Silhouette Maker: local server. Listens on this computer only (127.0.0.1)."""
from __future__ import annotations

import io
import json
import re
import shutil
import sys
import threading
import uuid
import webbrowser
import zipfile
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image

import engine

__version__ = "0.1.0"
ROOT = Path(__file__).resolve().parent
def _arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


DATA = Path(_arg("--data", ROOT / "data"))  # --data and --port are for testing
PORT = int(_arg("--port", 8777))
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 80 * 1024 * 1024
_cache: dict[str, tuple] = {}  # id -> (key, outline result)
_cache_lock = threading.Lock()


@app.before_request
def _local_only():
    # Only this computer's own browser page may talk to the app.
    if request.host not in ALLOWED_HOSTS:
        abort(403)
    origin = request.headers.get("Origin")
    if origin and origin.split("://", 1)[-1] not in ALLOWED_HOSTS:
        abort(403)


@app.after_request
def _no_store(resp):
    # a photo never changes once added, so the gallery may keep its pictures
    keep = request.path.endswith(("/thumb", "/photo")) and resp.status_code == 200
    resp.headers["Cache-Control"] = "private, max-age=86400" if keep else "no-store"
    return resp


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


def _result(d: Path, meta: dict) -> dict:
    """Outline for the item's current mask + settings (cached)."""
    key = ((d / "mask.png").stat().st_mtime_ns, json.dumps(meta["settings"], sort_keys=True))
    with _cache_lock:
        hit = _cache.get(d.name)
        if hit and hit[0] == key:
            return hit[1]
    res = engine.outline(_mask(d), meta["settings"])
    with _cache_lock:
        _cache[d.name] = (key, res)
    return res


def _outline_json(d: Path, meta: dict) -> dict:
    res = _result(d, meta)
    if not res["loops"]:
        return {"d": "", "bbox": None, "width_in": 0, "height_in": 0}
    w, h = engine.size_in(res, meta["settings"])
    return {"d": engine.path_d(res["loops"], nd=1), "bbox": res["bbox"],
            "width_in": round(w, 3), "height_in": round(h, 3)}


def _public(d: Path, meta: dict) -> dict:
    return {"id": d.name, "name": meta["name"], "w": meta["w"], "h": meta["h"],
            "done": bool(meta.get("done")), "added": d.stat().st_ctime, "rev": meta.get("rev", 0),
            "settings": meta["settings"], "outline": _outline_json(d, meta)}


def _chosen():
    """(name, outline, settings) for the ids in ?ids=a,b (or every photo), skipping empty shapes."""
    ids = [i for i in request.args.get("ids", "").split(",") if i]
    dirs = [_dir(i) for i in ids] if ids else (
        sorted(DATA.iterdir(), key=lambda p: p.stat().st_ctime) if DATA.is_dir() else [])
    out = []
    for d in dirs:
        if (d / "meta.json").exists() and (d / "mask.png").exists():
            meta = _meta(d)
            res = _result(d, meta)
            if res["loops"]:
                out.append((meta["name"], res, meta["settings"]))
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
            "space": min(6.0, max(0.0, float(a.get("space") or 0))),
            "orn": min(24.0, max(0.5, float(a.get("orn") or 3)))}


@app.get("/")
def index():
    return send_from_directory(ROOT / "static", "index.html")


@app.get("/api/items")
def list_items():
    out = []
    if DATA.is_dir():
        for d in sorted(DATA.iterdir(), key=lambda p: p.stat().st_ctime):
            if (d / "meta.json").exists() and (d / "mask.png").exists():
                out.append(_public(d, _meta(d)))
    return jsonify(out)


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
    settings = dict(engine.DEFAULT_SETTINGS, cut=engine.default_cut(mask))
    for k in ("height_in",):
        if request.form.get(k):
            settings[k] = float(request.form[k])
    name = Path(f.filename or "photo").stem[:60]
    meta = {"name": name, "w": img.width, "h": img.height, "settings": settings}
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
    img = Image.open(io.BytesIO(request.get_data())).convert("L")
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
    return jsonify(dict(_outline_json(d, meta), note=note))


@app.post("/api/items/<item_id>/rotate")
def rotate_item(item_id):
    """Turn the photo and its shape a quarter turn (turns: 1 = right, -1 = left, 2 = upside down)."""
    d = _dir(item_id)
    meta = _meta(d)
    turns = int((request.get_json(silent=True) or {}).get("turns", 1))
    img, mask = engine.turn(Image.open(d / "photo.jpg").convert("RGB"), _mask(d), -turns)
    img.save(d / "photo.jpg", quality=95)
    Image.fromarray(mask).save(d / "mask.png")
    (d / "thumb.jpg").unlink(missing_ok=True)
    meta.update(w=img.width, h=img.height, rev=meta.get("rev", 0) + 1)
    meta["settings"]["cut"] = engine.default_cut(mask)
    _save_meta(d, meta)
    return jsonify(_public(d, meta))


@app.patch("/api/items/<item_id>")
def patch_item(item_id):
    d = _dir(item_id)
    meta = _meta(d)
    body = request.get_json(force=True)
    if "name" in body:
        meta["name"] = str(body["name"])[:60]
    if "done" in body:
        meta["done"] = bool(body["done"])
    s = body.get("settings") or {}
    cur = meta["settings"]
    if "height_in" in s:
        cur["height_in"] = min(24.0, max(0.25, float(s["height_in"])))
    if "smooth" in s:
        cur["smooth"] = min(10.0, max(0.0, float(s["smooth"])))
    if "tilt_deg" in s:
        cur["tilt_deg"] = min(45.0, max(-45.0, float(s["tilt_deg"])))
    if "detail_mm" in s:
        cur["detail_mm"] = min(3.0, max(0.0, float(s["detail_mm"])))
    for k in ("flip", "cut_on"):
        if k in s:
            cur[k] = bool(s[k])
    if s.get("cut"):
        cur["cut"] = [[float(p[0]), float(p[1])] for p in s["cut"]][:3]
    _save_meta(d, meta)
    return jsonify(_public(d, meta))


@app.delete("/api/items/<item_id>")
def delete_item(item_id):
    d = _dir(item_id)
    shutil.rmtree(d)
    _cache.pop(item_id, None)
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
    w = min(24.0, max(2.0, float(request.args.get("w", 11.5))))
    h = min(24.0, max(2.0, float(request.args.get("h", 11.5))))
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
