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
    assert "<circle" in engine.svg_single(r, s, "x", {"border": "circle", "orn": 3.0})
    assert "<rect" in engine.svg_single(r, s, "x", {"border": "rect", "space": 0.25})
    assert "<circle" not in engine.svg_single(r, s, "x") and "<rect" not in engine.svg_single(r, s, "x")
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
