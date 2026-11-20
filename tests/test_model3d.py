"""The 3D-print files: closed solids, at the size asked for, in millimetres."""
import numpy as np
import pytest

import engine
import model3d
from test_engine import bust

S = dict(engine.DEFAULT_SETTINGS)


def shape():
    return engine.outline(bust(), S)


def tris(data: bytes) -> np.ndarray:
    n = int(np.frombuffer(data[80:84], "<u4")[0])
    assert len(data) == 84 + 50 * n
    rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].astype(np.float64)


def open_edges(v: np.ndarray) -> int:
    """Edges that are not matched by a neighbour going the other way: 0 for a closed solid."""
    key = np.round(v * 1000).astype(np.int64)
    edges = {}
    for t in key:
        for i in range(3):
            e = (tuple(t[i]), tuple(t[(i + 1) % 3]))
            edges[e] = edges.get(e, 0) + 1
    return sum(1 for (a, b), c in edges.items() if edges.get((b, a), 0) != c)


def volume(v: np.ndarray) -> float:
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6)


def size(v: np.ndarray) -> np.ndarray:
    p = v.reshape(-1, 3)
    return p.max(0) - p.min(0)


def test_silhouette_alone_is_a_closed_solid_at_the_height_set():
    res = shape()
    v = tris(model3d.silhouette_stl(res, S, {"thick_mm": 2.0}))
    w_in, h_in = engine.size_in(res, S)
    assert open_edges(v) == 0
    assert np.allclose(size(v), [w_in * 25.4, h_in * 25.4, 2.0], atol=0.5)
    flat = model3d.silhouette_shape(res, S)
    assert volume(v) == pytest.approx(flat.area * 2.0, rel=1e-3)  # facing outwards, nothing missing


def test_flip_mirrors_the_3d_shape_too():
    from shapely.affinity import scale

    mask = bust()
    mask[400:470, 150:260] = 255  # a nose, so left and right differ
    res = engine.outline(mask, S)
    a = model3d.silhouette_shape(res, S)
    b = model3d.silhouette_shape(res, dict(S, flip=True))
    width = engine.size_in(res, S)[0] * 25.4
    mirrored = scale(a, xfact=-1, origin=(width / 2, 0))
    assert a.symmetric_difference(b).area > 0.01 * a.area  # it really changed
    assert b.symmetric_difference(mirrored).area < 0.002 * a.area  # into the mirror image


@pytest.mark.parametrize("kind", list(engine.ORNAMENTS))
def test_every_ornament_is_closed_and_the_right_size(kind):
    opts = {"shape": kind, "orn": 3.0, "orn_h": 4.0, "name_on": True, "line2": "2026"}
    sizes = {"base_mm": 3.0, "raise_mm": 1.0, "hang": "none"}
    sh = model3d.ornament_shapes("Ava Marie", shape(), dict(S, height_in=1.2), opts, hang="none")
    assert sh["lettering"] is not None and sh["hole"] is None
    ow, oh = engine.orn_size(engine._opts(opts))
    assert np.allclose(sh["base"].bounds, [0, -oh * 25.4, ow * 25.4, 0], atol=0.05)
    for name, data in model3d.ornament_parts("Ava Marie", shape(), dict(S, height_in=1.2), opts, sizes).items():
        v = tris(data)
        assert open_edges(v) == 0, name
        assert volume(v) > 0, name
    v = tris(model3d.ornament_stl("Ava Marie", shape(), dict(S, height_in=1.2), opts, sizes))
    assert size(v)[2] == pytest.approx(4.0)


def test_parts_touch_but_do_not_overlap():
    sh = model3d.ornament_shapes("Ava", shape(), S, {"shape": "circle", "orn": 3.0, "name_on": True, "place": "arc"})
    assert sh["silhouette"].intersection(sh["lettering"]).area < 1e-6
    parts = model3d.ornament_parts("Ava", shape(), S, {"orn": 3.0, "name_on": True}, {"base_mm": 2.0, "raise_mm": 0.8})
    base, sil = tris(parts["base"]), tris(parts["silhouette"])
    assert base[:, :, 2].max() == pytest.approx(2.0) and sil[:, :, 2].min() == pytest.approx(2.0)
    assert sil[:, :, 2].max() == pytest.approx(2.8)


def test_hanging_hole_loop_or_nothing():
    res, opts = shape(), {"shape": "circle", "orn": 3.0}
    small = dict(S, height_in=1.5)  # leaves room near the top
    none = model3d.ornament_shapes("", res, small, opts, hang="none")
    hole = model3d.ornament_shapes("", res, small, opts, 4.0, "hole")
    loop = model3d.ornament_shapes("", res, small, opts, 4.0, "loop")
    assert none["hole"] is None and len(none["base"].interiors) == 0
    assert not hole["loop"] and len(hole["base"].interiors) == 1  # a hole through the ornament
    assert hole["base"].bounds == pytest.approx(none["base"].bounds)  # and nothing sticking out
    assert hole["hole"].area == pytest.approx(np.pi * 2.0**2, rel=0.02)
    assert loop["loop"] and loop["base"].bounds[3] > none["base"].bounds[3] + 3  # a ring standing on top
    # automatic: a hole when it clears the silhouette, a loop when the head is in the way
    assert not model3d.ornament_shapes("", res, small, opts, 4.0, "auto")["loop"]
    assert model3d.ornament_shapes("", res, dict(S, height_in=2.9), opts, 4.0, "auto")["loop"]
    # a hole that has to go through the silhouette stays open
    forced = model3d.ornament_shapes("", res, dict(S, height_in=2.9), opts, 4.0, "hole")
    assert forced["silhouette"].intersection(forced["hole"]).area < 1e-6


def test_letters_keep_their_holes():
    lay = engine._layout(shape(), S, {"orn": 3.0, "name_on": True, "font": "sans"}, "BOB", framed=True)
    letters = model3d.lettering_shape(lay)
    polys = list(letters.geoms) if letters.geom_type == "MultiPolygon" else [letters]
    assert len(polys) == 3 and sum(len(p.interiors) for p in polys) == 5  # B has two holes, O one


@pytest.mark.parametrize("place", ["below", "arc"])
def test_letters_built_from_parts_are_lettered_too(place):
    # An accented letter is usually made of two other glyphs in the font; so is
    # Georgia's middle dot. Those used to stop the 3D files (0.5.0).
    for font in ("serif", "sans"):
        sh = model3d.ornament_shapes("Zoé", shape(), S, {"orn": 3.0, "name_on": True, "line2": "2026",
                                                        "place": place, "font": font})
        assert sh["lettering"] is not None and sh["lettering"].area > 1


def test_a_picture_of_the_model():
    png = model3d.ornament_picture("Ava", shape(), S, {"orn": 3.0, "name_on": True})
    assert png[:4] == b"\x89PNG" and len(png) > 2000
