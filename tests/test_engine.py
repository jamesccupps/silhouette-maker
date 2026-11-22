"""The outline, sizing and file-writing code. No model is needed: the masks are drawn."""
import re

import cv2
import numpy as np
import pytest

import engine


def bust(w=900, h=1200):
    """Head, neck and shoulders running off the bottom edge."""
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, int(h * 0.36)), (int(w * 0.27), int(h * 0.25)), 0, 0, 360, 255, -1)
    cv2.rectangle(m, (int(w * 0.43), int(h * 0.55)), (int(w * 0.57), int(h * 0.8)), 255, -1)
    cv2.ellipse(m, (w // 2, h), (int(w * 0.45), int(h * 0.22)), 0, 0, 360, 255, -1)
    return m


def settings(mask, **over):
    return dict(engine.DEFAULT_SETTINGS, cut=engine.default_cut(mask), **over)


def test_outline_is_one_closed_shape_at_the_asked_height():
    m = bust()
    s = settings(m, height_in=2.25)
    r = engine.outline(m, s)
    assert len(r["loops"]) == 1
    w_in, h_in = engine.size_in(r, s)
    assert h_in == 2.25 and 1.0 < w_in < 2.25
    svg = engine.svg_single(r, s, "A & <B>")
    assert f'height="{h_in:.4f}in"' in svg and "A &amp; &lt;B&gt;" in svg
    # numbers are 72 per inch, which is how Cricut Design Space reads them
    vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    assert float(vb.group(2)) == pytest.approx(2.25 * 72, abs=0.01)
    d = re.search(r' d="([^"]+)"', svg).group(1)
    assert d.startswith("M") and d.endswith("Z")
    ys = [float(v) for v in re.findall(r"[\d.-]+ ([\d.-]+)", d)]  # includes control points
    assert min(ys) > -4 and max(ys) < 2.25 * 72 + 4


def test_neckline_cuts_the_shoulders_off():
    m = bust()
    with_cut = engine.outline(m, settings(m))
    without = engine.outline(m, settings(m, cut_on=False))
    assert with_cut["bbox"][3] < without["bbox"][3] - 50
    assert without["bbox"][3] == pytest.approx(m.shape[0], abs=3)


def test_empty_mask_gives_no_shape():
    r = engine.outline(np.zeros((300, 300), np.uint8), dict(engine.DEFAULT_SETTINGS))
    assert r["loops"] == [] and r["bbox"] is None


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_upright_turns_puts_the_body_at_the_bottom(k):
    m = bust()
    turned = np.rot90(m, k)  # the photo as taken: body on some other edge
    back = engine.upright_turns(turned)
    assert (k + back) % 4 == 0
    assert np.array_equal(np.rot90(turned, back), m)


def test_upright_turns_leaves_a_floating_shape_alone():
    m = np.zeros((600, 600), np.uint8)
    cv2.circle(m, (300, 300), 150, 255, -1)
    assert engine.upright_turns(m) == 0


def test_straighten_undoes_a_tilt():
    m = np.zeros((1200, 1200), np.uint8)
    cv2.ellipse(m, (600, 600), (150, 400), 0, 0, 360, 255, -1)  # tall, clear of the edges
    s = dict(engine.DEFAULT_SETTINGS, cut_on=False)
    upright = engine.outline(m, s)
    # tilt the photo 20 degrees counter-clockwise, as a phone held crooked would
    rot = cv2.getRotationMatrix2D((m.shape[1] / 2, m.shape[0] / 2), 20, 1.0)
    crooked = cv2.warpAffine(m, rot, (m.shape[1], m.shape[0]))
    fixed = engine.outline(crooked, dict(s, tilt_deg=20))
    wrong = engine.outline(crooked, dict(s, tilt_deg=-20))

    def width(r):
        return r["bbox"][2] - r["bbox"][0]

    assert width(fixed) == pytest.approx(width(upright), rel=0.04)
    assert width(wrong) > width(upright) * 1.15


def test_flip_mirrors_the_path():
    m = bust()
    m[300:340, 100:260] = 255  # a nose, so left and right differ
    s = settings(m)
    r = engine.outline(m, s)
    a, b = engine.placed_d(r, s), engine.placed_d(r, dict(s, flip=True))
    assert a != b
    xs = lambda d: [float(v) for v in re.findall(r"([\d.-]+) [\d.-]+", d)]
    width = engine.size_in(r, s)[0] * engine.UNITS_PER_INCH
    assert sorted(xs(b)) == pytest.approx(sorted(width - x for x in xs(a)), abs=0.03)


def test_border_options_change_the_piece():
    m = bust()
    s = settings(m)
    r = engine.outline(m, s)
    sw, sh = engine.size_in(r, s)
    assert engine.layout(r, s) == (sw, sh, 0.0, 0.0)
    assert engine.layout(r, s, {"space": 0.5})[:2] == (sw + 1.0, sh + 1.0)
    w, h, ox, oy = engine.layout(r, s, {"border": "circle", "orn": 3.0})
    assert (w, h) == (3.0, 3.0) and ox == pytest.approx((3 - sw) / 2)
    assert "<ellipse" in engine.svg_single(r, s, "x", {"border": "ornament", "orn": 3.0})
    assert "<ellipse" in engine.svg_single(r, s, "x", {"border": "circle", "orn": 3.0})  # the older name for it
    assert "<rect" in engine.svg_single(r, s, "x", {"border": "rect", "space": 0.25})
    assert "<ellipse" not in engine.svg_single(r, s, "x") and "<rect" not in engine.svg_single(r, s, "x")
    png = engine.png_cutout(r, s, {"border": "circle", "orn": 3.0})
    assert png.size[0] == png.size[1] and png.getpixel((0, 0))[3] == 0  # corner outside the circle
    tight = engine.png_cutout(r, s)
    assert tight.size[1] / tight.info["dpi"][1] == pytest.approx(2.25, abs=0.01)


def test_sheets_keep_every_piece_on_the_mat():
    m = bust()
    s = settings(m)
    r = engine.outline(m, s)
    sheets = engine.svg_sheets([(f"kid {i}", r, s) for i in range(40)], 11.5, 11.5)
    assert len(sheets) > 1
    assert sum(sheet.count("<path") for sheet in sheets) == 40
    for sheet in sheets:
        for d in re.findall(r' d="([^"]+)"', sheet):
            nums = [float(v) for v in re.findall(r"[\d.-]+", d)]
            assert min(nums) > -4 and max(nums) < 11.5 * 72 + 4  # control points bow out a little


def test_specks_inside_are_filled_but_real_gaps_stay():
    m = bust()
    s = settings(m, cut_on=False)
    clean = engine.outline(m, s)
    speckled = m.copy()
    for x, y in ((430, 380), (470, 420), (400, 460)):
        cv2.circle(speckled, (x, y), 6, 0, -1)  # what a cut-out model leaves inside a head
    assert len(engine.outline(speckled, s)["loops"]) > 1  # left alone, each speck would be cut
    assert len(engine.outline(engine.clean_cutout(speckled), s)["loops"]) == len(clean["loops"]) == 1
    gap = m.copy()
    cv2.circle(gap, (450, 430), 70, 0, -1)  # a real opening, like between a ponytail and the neck
    assert len(engine.outline(engine.clean_cutout(gap), s)["loops"]) == 2
    # a small gap erased by hand after the cut-out is the user's choice and stays
    by_hand = engine.clean_cutout(m)
    cv2.circle(by_hand, (450, 430), 12, 0, -1)
    assert len(engine.outline(by_hand, s)["loops"]) == 2


def test_idle_models_are_let_go():
    engine._sessions["pretend"] = object()
    engine.release_models()
    assert engine._sessions == {}


def test_iphone_heic_photo_opens(tmp_path):
    pytest.importorskip("pillow_heif")
    from PIL import Image

    f = tmp_path / "IMG_0001.HEIC"
    Image.new("RGB", (640, 480), (180, 170, 160)).save(f, format="HEIF")
    img = engine.load_photo(f)
    assert img.mode == "RGB" and img.size == (640, 480)


needs_font = pytest.mark.skipif(not engine.fonts(), reason="no usable font on this computer")


def piece():
    m = bust()
    s = settings(m)
    return engine.outline(m, s), s


def path_xs(d):
    return [float(v) for v in re.findall(r"(-?[\d.]+) -?[\d.]+", d)]


@needs_font
def test_a_name_becomes_outlines_not_text():
    key = engine.fonts()[0]["key"]
    r, s = piece()
    plain = engine.svg_single(r, s, "Ava")
    named = engine.svg_single(r, s, "Ava", {"name_on": True, "font": key, "tsize": 0.3})
    both = engine.svg_single(r, s, "Ava", {"name_on": True, "line2": "2026", "font": key, "tsize": 0.3})
    only_year = engine.svg_single(r, s, "Ava", {"line2": "2026", "font": key})
    assert "<text" not in named  # Design Space ignores <text>; it must be a path
    assert named.count("<path") == plain.count("<path") + 1 and "Ava lettering" in named
    h = lambda svg: float(re.search(r'height="([\d.]+)in"', svg).group(1))
    assert h(plain) == 2.25 and h(named) == pytest.approx(2.25 + 0.12 + 0.98 * 0.3, abs=0.01) and h(both) > h(named)
    assert only_year.count("<path") == plain.count("<path") + 1  # the second line does not need the name
    # nothing to letter: nothing is added
    assert engine.svg_single(r, s, "  ", {"name_on": True, "font": key}) == engine.svg_single(r, s, "  ")


@needs_font
def test_a_long_name_wraps_then_shrinks_to_stay_on_the_ornament():
    key = engine.fonts()[0]["key"]
    r, s = piece()
    o = {"name_on": True, "font": key, "tsize": 0.3, "orn": 3.0}
    short = engine._layout(r, s, o, "Ava", framed=True)
    assert [(t, size) for t, size, *_ in short["texts"]] == [("Ava", 0.3)] and short["fits"]

    two_words = engine._layout(r, s, o, "Annabelle Montgomery", framed=True)
    assert [t for t, *_ in two_words["texts"]] == ["Annabelle", "Montgomery"]  # wrapped at the space

    camera = engine._layout(r, s, o, "IMG_20261119_190204", framed=True)  # no space to wrap at
    (text, size, cx, base, _), = camera["texts"]
    assert size < 0.3 and cx == pytest.approx(camera["w"] / 2)
    for lay in (short, two_words, camera):
        assert (lay["w"], lay["h"]) == (3.0, pytest.approx(3.0, abs=0.75))
        for t, size, cx, base, f in lay["texts"]:
            half = engine.text_width_em(t, f) * size / 2
            y = base - 0.35 * size - lay["cy"]
            assert half * 2 <= 0.93 * max(0.75, 3.0 * (1 - (y / 1.5) ** 2) ** 0.5 if abs(y) < 1.5 else 0.75)


@needs_font
def test_lettering_can_curve_along_the_bottom_of_a_round_ornament():
    key = engine.fonts()[0]["key"]
    r, s = piece()
    o = {"name_on": True, "line2": "2026", "place": "arc", "font": key, "tsize": 0.3, "orn": 3.5}
    lay = engine._layout(r, s, o, "Ava")
    assert lay["arc"] and not lay["texts"] and (lay["w"], lay["h"]) == (3.5, 3.5)
    assert [t.strip(" \u00b7-") for t, _, _ in lay["arc"]["segments"]] == ["Ava", "2026"]
    u = 72.0
    d = engine.text_arc_d("Ava 2026", key, 0.3 * u, 100.0, 100.0, 1.4 * u)
    pts = np.array(re.findall(r"(-?[\d.]+) (-?[\d.]+)", d), dtype=float)
    dist = np.hypot(pts[:, 0] - 100, pts[:, 1] - 100)
    assert dist.min() > (1.4 - 0.32) * u and dist.max() < (1.4 + 0.12) * u  # every letter hugs the circle
    assert pts[:, 1].min() > 100 and abs(pts[:, 0].mean() - 100) < 0.2 * u  # along the bottom, centred
    # a very long line shrinks instead of wrapping right round
    long_lay = engine._layout(r, s, dict(o, tsize=0.7, line2="Sunshine Room Preschool Class of 2026"),
                              "Annabelle Montgomery")
    assert long_lay["arc"]["segments"][0][2] < 0.7
    # other shapes have no curve to follow: the lines go underneath
    square = engine._layout(r, s, dict(o, shape="square"), "Ava")
    assert square["arc"] is None and len(square["texts"]) == 2


@pytest.mark.parametrize("shape", list(engine.ORNAMENTS))
def test_every_ornament_shape_fills_its_box_and_exports(shape):
    pts = engine.ornament_points(shape, 3.0, 4.0)
    assert pts[:, 0].min() == pytest.approx(-1.5) and pts[:, 0].max() == pytest.approx(1.5)
    assert pts[:, 1].min() == pytest.approx(-2.0) and pts[:, 1].max() == pytest.approx(2.0)
    r, s = piece()
    o = {"border": "ornament", "shape": shape, "orn": 4.0, "orn_h": 5.0}
    svg = engine.svg_single(r, s, "Ava", o)
    tall = shape in ("oval", "rect")  # only these take a separate height
    assert f'width="4.0000in" height="{5 if tall else 4}.0000in"' in svg and "Ava ornament" in svg
    png = engine.png_cutout(r, s, o)
    assert png.size[0] / png.info["dpi"][0] == pytest.approx(4.0, abs=0.02)
    on = engine.on_ornament("Ava", r, s, o)
    assert on["w"] == 4.0 and "Ava ornament" not in on["body"]  # the preview draws its own outline
    assert engine.on_ornament("Ava", None, s, o)["fits"]  # nothing cut yet: just the ornament


def test_a_shape_too_big_for_the_ornament_is_reported():
    r, s = piece()
    assert engine.on_ornament("x", r, s, {"orn": 3.0})["fits"]
    assert not engine.on_ornament("x", r, s, {"orn": 1.5})["fits"]
    assert not engine.on_ornament("x", r, s, {"orn": 3.0, "shape": "star"})["fits"]


@needs_font
def test_name_and_second_line_have_their_own_font_and_size():
    keys = [f["key"] for f in engine.fonts()]
    a, b = keys[0], keys[-1]
    r, s = piece()
    o = {"name_on": True, "line2": "2026", "font": a, "tsize": 0.34, "font2": b, "tsize2": 0.18, "orn": 4.0}
    (t1, s1, _, base1, f1), (t2, s2, _, base2, f2) = engine._layout(r, s, o, "Ava", framed=True)["texts"]
    assert (t1, s1, f1) == ("Ava", 0.34, a) and (t2, s2, f2) == ("2026", 0.18, b) and base2 > base1
    # left alone, the second line uses the name's font, a little smaller
    (_, s1, _, _, f1), (_, s2, _, _, f2) = engine._layout(r, s, {"name_on": True, "line2": "2026", "font": a, "tsize": 0.3,
                                                                "orn": 4.0}, "Ava", framed=True)["texts"]
    assert f2 == f1 == a and s2 == pytest.approx(0.24)
    # curved: both parts keep their own font and their sizes stay in proportion when squeezed
    arc = engine._layout(r, s, dict(o, place="arc", orn=3.0, tsize=0.6, tsize2=0.3), "Annabelle Montgomery")["arc"]
    (_, fa, sa), (_, fb, sb) = arc["segments"]
    assert (fa, fb) == (a, b) and sa < 0.6 and sa / sb == pytest.approx(2.0, rel=0.01)
    if len(keys) > 1:  # a different font really gives different letters
        same = engine.svg_single(r, s, "Ava", dict(o, font2=a))
        assert engine.svg_single(r, s, "Ava", o) != same


@needs_font
def test_fonts_are_listed_by_name_and_old_choices_still_work():
    fonts = engine.fonts()
    labels = [f["label"] for f in fonts]
    assert labels == sorted(labels, key=str.lower) and len(set(labels)) == len(labels)
    assert all(set(f) == {"key", "label", "suggested"} for f in fonts)
    keys = {f["key"] for f in fonts}
    for old in ("serif", "sans", "bold", "script", "not a font", None, ""):
        assert engine.resolve_font(old) in keys  # always something usable
    assert engine.resolve_font(fonts[-1]["key"]) == fonts[-1]["key"]
    for f in fonts[:: max(1, len(fonts) // 12)]:  # a spread of them can all letter a name
        assert len(engine.text_d("Ava 2026", f["key"], 20.0, 0.0, 0.0)) > 50


def test_print_pages_pack_everyone_at_real_size():
    mask = bust()
    S = settings(mask)
    res = engine.outline(mask, S)
    items = [(f"Child {i}", res, S) for i in range(7)]
    pages = engine.print_pages(items, {"orn": 3.0}, "letter")
    assert len(pages) == 2 and all('width="8.5000in" height="11.0000in"' in p for p in pages)
    assert sum(p.count("in tall") for p in pages) == 7
    assert all(f'width="{2 * 72:.2f}" height=' in p for p in pages)  # every page has the 2 inch bar
    for p in pages:  # nothing is placed outside the half-inch margin
        for x, y in re.findall(r'translate\(([\d.]+) ([\d.]+)\)', p):
            assert 36 <= float(x) <= (8.5 - 0.5 - 3.0) * 72 + 0.01 and 36 <= float(y) <= (11 - 0.5 - 3.0) * 72 + 0.01
    one = engine.print_pages(items[:1], {"orn": 3.0}, "a4", show="shape", ink="outline", names=False, copies=3)
    assert len(one) == 1 and one[0].count("<g transform") == 3 and "in tall" not in one[0]
    assert 'stroke-width="0.7"' in one[0] and "stroke-dasharray" not in one[0]


@needs_font
def test_the_font_list_is_kept_between_starts(tmp_path, monkeypatch):
    import fontTools.ttLib

    monkeypatch.setattr(engine, "FONT_CACHE", tmp_path / "fonts.json")
    engine._scan.cache_clear()
    try:
        first = engine._registry()
        assert (tmp_path / "fonts.json").exists()
        engine._scan.cache_clear()                              # as after starting the app again

        def no_look(*a, **k):
            raise AssertionError("looked through the fonts again")
        with monkeypatch.context() as m:
            m.setattr(fontTools.ttLib, "TTFont", no_look)
            m.setattr(fontTools.ttLib, "TTCollection", no_look)
            assert engine._registry() == first                  # read from the kept list
        engine._scan.cache_clear()
        monkeypatch.setattr(engine, "_font_signature", lambda files: "a font was added")
        assert engine._registry() == first                      # looked through again, same fonts
        assert "a font was added" in (tmp_path / "fonts.json").read_text(encoding="utf-8")
    finally:
        engine._scan.cache_clear()


def test_print_pages_count_what_is_on_each_page():
    r, s = engine.outline(bust(), engine.DEFAULT_SETTINGS), dict(engine.DEFAULT_SETTINGS, height_in=2.0)
    counts = []
    pages = engine.print_pages([("A", r, s)] * 15, {"orn": 3.0}, "letter", counts=counts)
    assert len(counts) == len(pages) > 1 and sum(counts) == 15 and counts[0] >= counts[-1]
