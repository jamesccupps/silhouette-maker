"""3D-print files: the silhouette alone, or the whole ornament, as STL.

Everything is in millimetres, lying flat on the print bed (z up), looking the
same from above as the cut file does on screen. The flat shapes come from the
same outline, lettering and ornament code as the cut files, so sizes match.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

import engine

MM = 25.4  # per inch
DEFAULTS = {"base_mm": 3.0,  # thickness of the ornament
            "raise_mm": 1.0,  # how far the silhouette and lettering stand up from it
            "thick_mm": 2.0,  # thickness of the silhouette printed alone
            "hole_mm": 4.0}  # hanging hole across; 0 for none
HOLE_WALL = 2.0  # material kept between the hole and the edge, mm
SINK = 0.2  # in the one-piece file the raised parts start this far inside the ornament, so they fuse


def _solid(shape):
    """A clean polygon or multipolygon (or an empty one) from any geometry."""
    shape = shape.buffer(0)
    if shape.geom_type == "GeometryCollection":
        shape = unary_union([g for g in shape.geoms if g.geom_type in ("Polygon", "MultiPolygon")])
    return shape


def _polys(shape) -> list[Polygon]:
    if shape is None or shape.is_empty:
        return []
    parts = shape.geoms if isinstance(shape, MultiPolygon) else [shape]
    return [p for p in parts if p.geom_type == "Polygon" and p.area > 0.02]  # specks cannot be printed


def _rings_to_shape(rings: list, by_direction: bool):
    """Fill a set of rings the way the cut file is filled. Biggest first: each
    smaller ring either cuts a hole or adds an island.

    by_direction False: rings alternate (the silhouette: outline, gaps, islands).
    by_direction True: a ring drawn the same way round as the biggest adds, the
    other way round cuts (how fonts mark the holes in letters)."""
    polys = []
    for r in rings:
        if len(r) >= 3:
            poly = Polygon(r)
            if poly.area > 1e-9:
                polys.append((poly.exterior.is_ccw, _solid(poly)))
    polys.sort(key=lambda t: -t[1].area)
    shape = Polygon()
    for ccw, poly in polys:
        if by_direction:
            shape = shape.union(poly) if ccw == polys[0][0] else shape.difference(poly)
        else:
            shape = shape.symmetric_difference(poly)
    return _solid(_solid(shape).simplify(0.015))  # points closer than a printer can place add nothing


def silhouette_shape(result: dict, settings: dict, x_in: float = 0.0, y_in: float = 0.0):
    """The cut line as a flat shape in mm, its top-left corner at (x_in, y_in)
    inches, with y turned to point up."""
    s = {**engine.DEFAULT_SETTINGS, **settings}
    x0, y0, x1, y1 = result["bbox"]
    k = float(s["height_in"]) / (y1 - y0)
    t = np.linspace(0, 1, 7)[1:].reshape(-1, 1)
    rings = []
    for loop in result["loops"]:
        pts, p0 = [np.asarray(loop[0], dtype=np.float64).reshape(1, 2)], np.asarray(loop[0], dtype=np.float64)
        for c1, c2, e in loop[1:]:
            c1, c2, e = (np.asarray(v, dtype=np.float64) for v in (c1, c2, e))
            pts.append((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * e)
            p0 = e
        pts = np.vstack(pts)[:-1]  # the last point is the first again
        xs = (x1 - pts[:, 0]) if s["flip"] else (pts[:, 0] - x0)
        rings.append(np.column_stack([(xs * k + x_in) * MM, -((pts[:, 1] - y0) * k + y_in) * MM]))
    return _rings_to_shape(rings, by_direction=False)


def lettering_shape(lay: dict):
    """The lettering of a laid-out piece as a flat shape in mm (y up), or None."""
    pen = engine.ring_pen()
    for t, size, cx, base, f in lay["texts"]:
        engine.text_d(t, f, size * MM, cx * MM, base * MM, pen=pen)
    if lay["arc"]:
        a = lay["arc"]
        engine.arc_d([(t, f, size * MM) for t, f, size in a["segments"]], a["cx"] * MM, a["cy"] * MM, a["radius"] * MM, pen=pen)
    letters = [_rings_to_shape([[(x, -y) for x, y in ring] for ring in glyph], by_direction=True)
               for glyph in pen.glyphs if glyph]
    letters = [g for g in letters if not g.is_empty]
    return _solid(unary_union(letters)) if letters else None


def _hang(base, cx: float, across: float, how: str, keep_clear):
    """Where the ribbon goes: (hole, loop). how is hole (through the ornament,
    near the top), loop (a ring standing on the top edge), none, or auto: a
    hole when it clears the silhouette and lettering, otherwise a loop.
    Either of the two may be None."""
    if across <= 0 or how == "none":
        return None, None
    r = across / 2
    minx, miny, maxx, maxy = base.bounds
    hit = base.intersection(LineString([(cx, miny - 1), (cx, maxy + 1)]))
    if hit.is_empty:
        return None, None
    top = hit.bounds[3]  # the highest point of the ornament on its centre line (the dip of a heart)
    if how in ("hole", "auto"):
        y = top - r - HOLE_WALL
        while y > (miny + maxy) / 2:
            if base.contains(Point(cx, y).buffer(r + HOLE_WALL, 16)):
                hole = Point(cx, y).buffer(r, 24)
                if how == "hole" or keep_clear is None or hole.distance(keep_clear) >= 1.0:
                    return hole, None
                break
            y -= 0.25
    centre = Point(cx, top + r)  # the hole just touches the top edge; the ring around it bites into the ornament
    return centre.buffer(r, 24), centre.buffer(r + HOLE_WALL, 32)


def ornament_shapes(name: str, result: dict, settings: dict, opts, hole_mm: float = DEFAULTS["hole_mm"],
                    hang: str = "auto") -> dict:
    """The flat shapes of the whole ornament, in mm: {"base", "silhouette",
    "lettering" (or None), "hole" (or None), "loop": the hole is in a ring on
    top, "fits"}. Laid out exactly like the preview and the cut files."""
    lay = engine._layout(result, settings, opts, name, framed=True)
    o = lay["o"]
    ow, oh = engine.orn_size(o)
    pts = engine.ornament_points(o["shape"], ow, oh, 240) + np.array([lay["cx"], lay["cy"]])
    base = _solid(Polygon(np.column_stack([pts[:, 0] * MM, -pts[:, 1] * MM])))
    sil = silhouette_shape(result, settings, lay["ox"], lay["oy"])
    letters = lettering_shape(lay)
    hole, loop = _hang(base, lay["cx"] * MM, float(hole_mm), hang, sil if letters is None else sil.union(letters))
    if loop is not None:
        base = _solid(base.union(loop))
    if hole is not None:
        base = _solid(base.difference(hole))
        sil = _solid(sil.difference(hole))  # the hole stays open even under the hair
        if letters is not None:
            letters = _solid(letters.difference(hole))
    if letters is not None:
        letters = _solid(letters.difference(sil))  # no part of the model belongs to two pieces
    return {"base": base, "silhouette": sil, "lettering": letters if letters is not None and not letters.is_empty else None,
            "hole": hole, "loop": loop is not None, "fits": lay["fits"]}


def prism(shape, z0: float, z1: float) -> np.ndarray:
    """Triangles (n, 3, 3) of the flat shape pulled up from z0 to z1: closed,
    facing outwards."""
    tris = []
    for poly in _polys(shape):
        poly = orient(poly, 1.0)  # outline counter-clockwise, holes clockwise
        for t in shapely.constrained_delaunay_triangles(poly).geoms:
            a, b, c = np.asarray(t.exterior.coords)[:3]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
                b, c = c, b
            tris.append([(a[0], a[1], z1), (b[0], b[1], z1), (c[0], c[1], z1)])  # top, facing up
            tris.append([(a[0], a[1], z0), (c[0], c[1], z0), (b[0], b[1], z0)])  # bottom, facing down
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords)
            for (ax, ay), (bx, by) in zip(xy[:-1], xy[1:]):
                tris.append([(ax, ay, z0), (bx, by, z0), (bx, by, z1)])
                tris.append([(ax, ay, z0), (bx, by, z1), (ax, ay, z1)])
    return np.asarray(tris, dtype=np.float32).reshape(-1, 3, 3)


def stl(tris: np.ndarray, title: str = "Silhouette Maker") -> bytes:
    """A binary STL file."""
    v = tris.astype(np.float32)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, length, out=np.zeros_like(n), where=length > 0)
    rec = np.zeros(len(v), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = n, v
    head = title.encode("ascii", "replace")[:80].ljust(80, b" ")
    return head + np.uint32(len(v)).tobytes() + rec.tobytes()


HANG = ("auto", "hole", "loop", "none")


def _sizes(o: dict | None) -> dict:
    """The thicknesses, and "hang" (see _hang), with defaults for what is missing."""
    o = o or {}
    z = {**DEFAULTS, **{k: float(v) for k, v in o.items() if k in DEFAULTS and v is not None}}
    z["hang"] = o.get("hang") if o.get("hang") in HANG else "auto"
    return z


def silhouette_stl(result: dict, settings: dict, sizes=None) -> bytes:
    """The silhouette by itself, flat, thick_mm thick."""
    z = _sizes(sizes)
    return stl(prism(silhouette_shape(result, settings), 0.0, z["thick_mm"]), "Silhouette")


def ornament_stl(name: str, result: dict, settings: dict, opts, sizes=None) -> bytes:
    """The whole ornament as one piece: the ornament's shape with a hanging
    hole, the silhouette and any lettering standing up from it."""
    z = _sizes(sizes)
    sh = ornament_shapes(name, result, settings, opts, z["hole_mm"], z["hang"])
    raised = sh["silhouette"] if sh["lettering"] is None else _solid(sh["silhouette"].union(sh["lettering"]))
    return stl(np.vstack([prism(sh["base"], 0.0, z["base_mm"]),
                          prism(raised, z["base_mm"] - SINK, z["base_mm"] + z["raise_mm"])]), "Ornament")


PARTS_NOTE = """These files are one ornament in separate pieces, for printing in more than one colour.
They share the same position, so together they make the whole ornament.

Bambu Studio / OrcaSlicer / PrusaSlicer: drag ALL the .stl files in at the same
time and answer Yes to "load as a single object with multiple parts". Then give
each part its own filament.

Sizes are in millimetres.
"""


def ornament_parts(name: str, result: dict, settings: dict, opts, sizes=None) -> dict[str, bytes]:
    """{"base": stl, "silhouette": stl, "lettering": stl (when there is any)}:
    the same ornament in pieces that touch but do not overlap."""
    z = _sizes(sizes)
    sh = ornament_shapes(name, result, settings, opts, z["hole_mm"], z["hang"])
    top = z["base_mm"] + z["raise_mm"]
    out = {"base": stl(prism(sh["base"], 0.0, z["base_mm"]), "Ornament base"),
           "silhouette": stl(prism(sh["silhouette"], z["base_mm"], top), "Silhouette")}
    if sh["lettering"] is not None:
        out["lettering"] = stl(prism(sh["lettering"], z["base_mm"], top), "Lettering")
    return out


def parts_zip(label: str, parts: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for kind, data in parts.items():
            zf.writestr(f"{label} - {kind}.stl", data)
        zf.writestr("How to use these files.txt", PARTS_NOTE)
    return buf.getvalue()


def picture(parts: list[tuple[np.ndarray, str]], width: int = 640) -> bytes:
    """A PNG of the model seen from above and a little to the front, each part
    (triangles, "#rrggbb") in its own colour. For the page: it shows what the
    file holds without opening a slicer."""
    import cv2

    a = np.deg2rad(32)
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    light = np.array([-0.35, 0.45, 0.82])
    light /= np.linalg.norm(light)
    tris, cols = [], []
    for v, colour in parts:
        if len(v):
            rgb = [int(colour[i:i + 2], 16) for i in (1, 3, 5)]
            tris.append(v.astype(np.float64))
            cols.append(np.tile(np.array(rgb[::-1], dtype=np.float64), (len(v), 1)))  # OpenCV wants blue first
    if not tris:
        return b""
    v, cols = np.vstack(tris), np.vstack(cols)
    q = v @ rot.T
    n = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    keep = n[:, 2] > 0  # the faces turned towards the viewer
    v, q, n, cols = v[keep], q[keep], n[keep], cols[keep]
    shade = 0.45 + 0.6 * np.clip(n @ light, 0, 1)
    xy = q[:, :, :2]
    lo, hi = xy.reshape(-1, 2).min(0), xy.reshape(-1, 2).max(0)
    k, pad = 2 * (width - 24) / (hi - lo)[0], 24  # drawn at twice the size, then shrunk: smooth edges
    canvas = np.zeros((int((hi - lo)[1] * k) + 2 * pad, 2 * width, 4), np.uint8)
    for i in np.lexsort((q[:, :, 2].mean(1), v[:, :, 2].max(1))):  # lower layers first, then back to front
        pts = np.column_stack([(xy[i, :, 0] - lo[0]) * k + pad, (hi[1] - xy[i, :, 1]) * k + pad]).astype(np.int32)
        c = np.clip(cols[i] * shade[i], 0, 255)
        cv2.fillConvexPoly(canvas, pts, (int(c[0]), int(c[1]), int(c[2]), 255))
    canvas = cv2.resize(canvas, (width, canvas.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", canvas)[1].tobytes()


def ornament_picture(name: str, result: dict, settings: dict, opts, sizes=None, base_colour: str = "#d9b27c",
                     raised_colour: str = "#111111") -> bytes:
    z = _sizes(sizes)
    sh = ornament_shapes(name, result, settings, opts, z["hole_mm"], z["hang"])
    raised = sh["silhouette"] if sh["lettering"] is None else _solid(sh["silhouette"].union(sh["lettering"]))
    return picture([(prism(sh["base"], 0.0, z["base_mm"]), base_colour),
                    (prism(raised, z["base_mm"], z["base_mm"] + z["raise_mm"]), raised_colour)])


def silhouette_picture(result: dict, settings: dict, sizes=None, colour: str = "#111111") -> bytes:
    return picture([(prism(silhouette_shape(result, settings), 0.0, _sizes(sizes)["thick_mm"]), colour)])
