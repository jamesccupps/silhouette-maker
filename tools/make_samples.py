"""Draw the sample profiles in samples/ (drawings, not real people).

They are what the README screenshots and a first try of the app use, so no
real person's photo ever needs to be in this repository. Each one is a
side-on portrait drawn from smooth curves: forehead, nose, lips, chin, an
ear, an eye, and hair in a different style, lit softly, on a plain wall.
Three are children and three are adults.

    python tools/make_samples.py            writes samples/
    python tools/make_samples.py some/dir   writes there instead
"""
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "samples"
W, H = 1200, 1600  # the finished picture
SS = 2  # drawn at twice the size, then reduced: smooth edges
U = 0.80 * SS  # pixels per unit; a head is about 1000 units from crown to chin
OX, OY = 590 * SS, 720 * SS  # where (0, 0), just above the ear, lands

# The face, from the hairline down to the base of the neck, facing left.
# x grows towards the back of the head, y grows downwards.
FACE = {
    "hairline": (-250, -430), "forehead": (-362, -255), "brow": (-352, -120), "nasion": (-330, -62),
    "bridge": (-362, 5), "tip": (-432, 78), "under_nose": (-378, 112), "lip_dip": (-372, 140),
    "upper_lip": (-402, 178), "mouth": (-378, 203), "lower_lip": (-394, 236), "chin_dip": (-362, 272),
    "chin": (-372, 332), "chin_low": (-325, 388), "jaw": (-215, 412), "throat": (-165, 445),
    "neck_front": (-160, 540), "neck_base": (-185, 660),
}
BACK = [(215, 660), (200, 500), (225, 350), (395, 130), (440, -110), (385, -345), (215, -495), (20, -535)]
ORDER = list(FACE)

# What makes each age look its age, as moves of the points above.
AGES = {
    "child": {"tip": (16, -6), "bridge": (8, 0), "brow": (6, 0), "forehead": (-8, 0), "upper_lip": (4, -8), "mouth": (2, -14),
              "lower_lip": (2, -18), "chin_dip": (0, -24), "chin": (10, -34), "chin_low": (6, -40), "jaw": (0, -34),
              "throat": (10, -30), "neck_front": (14, -10), "under_nose": (4, -4), "lip_dip": (2, -6)},
    "woman": {"tip": (2, 2), "chin": (2, 4), "chin_low": (0, 6)},
    "man": {"tip": (-20, 8), "bridge": (-12, 2), "brow": (-12, 0), "nasion": (-4, 0), "under_nose": (-4, 6), "chin": (-12, 16),
            "chin_low": (-12, 22), "jaw": (-6, 22), "throat": (-22, 18), "neck_front": (-30, 0), "neck_base": (-30, 0),
            "forehead": (6, 0), "lower_lip": (-2, 4), "chin_dip": (-2, 6)},
}

# name, age, skin, hair, shirt, wall (all RGB), hair style, and what makes this face its own
PEOPLE = [
    ("Ava", "child", (232, 190, 165), (98, 66, 42), (214, 120, 132), (226, 229, 232), "ponytail",
     {"tip": (-4, -4), "forehead": (-4, 0)}),
    ("Ben", "man", (214, 170, 140), (44, 36, 32), (74, 96, 126), (222, 228, 230), "short beard",
     {"tip": (-4, 4), "bridge": (-6, -2)}),
    ("Cleo", "woman", (240, 205, 184), (170, 112, 58), (150, 110, 170), (232, 229, 224), "bun",
     {"tip": (4, -8), "under_nose": (2, -4), "forehead": (6, 6), "upper_lip": (4, 0), "chin": (6, -4)}),
    ("Dev", "child", (150, 104, 76), (30, 24, 22), (212, 150, 60), (224, 230, 226), "curly",
     {"tip": (-6, 4), "upper_lip": (-8, 0), "lower_lip": (-10, 2), "forehead": (-4, 0)}),
    ("Eli", "child", (238, 198, 172), (200, 160, 98), (86, 140, 98), (228, 226, 232), "tousled",
     {"tip": (4, 0), "forehead": (-8, -6), "chin": (2, 4)}),
    ("Fay", "woman", (198, 150, 118), (38, 30, 32), (90, 170, 176), (228, 228, 226), "long",
     {"tip": (-12, 2), "bridge": (-6, 0), "forehead": (2, 4), "chin": (-4, 0)}),
]


# ---------------------------------------------------------------- curves and layers
def px(pts):
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    return np.stack([OX + p[:, 0] * U, OY + p[:, 1] * U], axis=1)


def ipx(pts):
    return np.round(px(pts)).astype(np.int32)


def smooth(pts, closed=True, steps=14):
    """A smooth curve through the points (Catmull-Rom), in units."""
    p = np.asarray(pts, np.float64)
    n = len(p)
    out = []
    for i in range(n if closed else n - 1):
        p0, p1, p2, p3 = (p[(i + k - 1) % n] if closed else p[min(max(i + k - 1, 0), n - 1)] for k in range(4))
        for t in np.linspace(0, 1, steps, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    if not closed:
        out.append(p[-1])
    return np.array(out)


def blank():
    return np.zeros((H * SS, W * SS), np.uint8)


def fill(shape, pts, value=255):
    """Fill one outline, or a list of them, into the mask."""
    for one in (pts if isinstance(pts, list) else [pts]):
        cv2.fillPoly(shape, [ipx(one)], value, lineType=cv2.LINE_AA)
    return shape


def stroke(img, pts, colour, width):
    cv2.polylines(img, [ipx(pts)], False, [float(c) for c in colour], thickness=max(1, int(round(width * SS))), lineType=cv2.LINE_AA)


def lit(mask, rgb, depth=0.20, soft=26.0, light=(-0.75, -0.6)):
    """The colour of a rounded thing: brighter where its edge faces the light."""
    m = mask.astype(np.float32) / 255.0
    b = cv2.GaussianBlur(m, (0, 0), soft * SS)
    gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=5)
    k = -(gx * light[0] + gy * light[1])  # the edge's outward direction against the light
    k = np.clip(k / (np.abs(k).max() + 1e-6), -1, 1)
    shade = 1.0 + depth * k - 0.10 * (1 - b) * m  # a little darker towards every edge
    return np.clip(np.asarray(rgb, np.float32)[None, None, :] * shade[..., None], 0, 255)


def put(img, mask, colour, alpha=1.0):
    a = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
    img[:] = img * (1 - a) + colour * a


def strands(mask, rgb, lines, rng, spread=0.22, width=1.6, strength=0.34):
    """Hair: the lit colour with many fine lighter and darker strands along `lines`."""
    col = lit(mask, rgb, depth=0.30, soft=22.0)
    tone = np.zeros(mask.shape, np.float32)
    for line in lines:
        layer = blank()
        cv2.polylines(layer, [ipx(line)], False, 255, thickness=max(1, int(round(width * SS * rng.uniform(0.6, 1.6)))),
                      lineType=cv2.LINE_AA)
        tone += layer.astype(np.float32) / 255.0 * rng.uniform(-spread, spread)
    tone = cv2.GaussianBlur(tone, (0, 0), 0.8 * SS)
    dark = 1.0 - float(np.mean(rgb)) / 255.0  # dark hair shows its strands less, so it gets more of them
    gain = strength * (1.0 + 1.6 * dark ** 2)
    return np.clip(col * (1.0 + gain * np.clip(tone, -1, 1))[..., None] + 26 * dark * np.clip(tone, 0, 1)[..., None], 0, 255)


def fan(centre, outer, rng, n=150, wobble=6.0, bend=26.0):
    """Strands from a parting at `centre` out to points along `outer`, each a little curved."""
    outer, centre = np.asarray(outer, np.float64), np.asarray(centre, np.float64)
    lines = []
    for _ in range(n):
        end = outer[rng.integers(len(outer))] + rng.normal(0, wobble, 2)
        start = centre + rng.normal(0, 28, 2)  # no single point where they all meet
        d = end - start
        perp = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
        mid = (start + end) / 2 + perp * rng.normal(0, bend)
        lines.append(smooth([start, mid, end], closed=False, steps=10))
    return lines


def lock(centre_line, half_width, rng, n=60, tip=0.96):
    """A lock of hair along a line: its outline and the strands inside it."""
    c = smooth(centre_line, closed=False, steps=12)
    d = np.gradient(c, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
    taper = np.sin(np.linspace(0.3, np.pi * tip, len(c)))[:, None]
    outline = np.vstack([c + nrm * half_width * taper, (c - nrm * half_width * taper)[::-1]])
    return outline, [c + nrm * half_width * taper * rng.uniform(-0.95, 0.95) for _ in range(n)]


def blob(cx, cy, r, rng, lumps=0.07, squash=1.0, n=36):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = r * (1 + lumps * np.sin(3 * t + rng.uniform(0, 6)) + lumps * 0.6 * np.sin(5 * t + rng.uniform(0, 6)))
    return np.stack([cx + rr * np.cos(t), cy + rr * np.sin(t) * squash], axis=1)


# ---------------------------------------------------------------- hair
CAP_OUT = [(-368, -240), (-340, -398), (-215, -522), (30, -582), (270, -537), (440, -377), (498, -140),
           (474, 80), (392, 230), (300, 285)]  # the hair's outer edge, from the fringe over the crown to the nape
CAP_IN = [(222, 300), (165, 205), (135, 95), (100, -25), (10, -70), (-70, -20), (-78, 45), (-125, -30),
          (-225, -150), (-300, -195)]  # and back along the ear and the temple
WHORL = (230.0, -390.0)


def hair_shapes(style, rng):
    """(on the head, behind the head): lists of (outline in units or None, strand lines)."""
    front, behind = [], []
    out, inn = list(CAP_OUT), list(CAP_IN)

    if "ponytail" in style:
        front.append((smooth(out + inn), fan((430, -215), smooth(out, False), rng) + fan((430, -215), smooth(inn, False), rng, 90)))
        behind.append(lock([(425, -215), (545, -250), (640, -110), (638, 130), (590, 320), (592, 430)], 92, rng, 90, tip=0.86))
    elif "bun" in style:
        front.append((smooth(out + inn), fan((400, -330), smooth(out, False), rng) + fan((400, -330), smooth(inn, False), rng, 90)))
        swirl = [np.stack([445 + (15 + 138 * u) * np.cos(a0 + 4.2 * u), -425 + (15 + 128 * u) * np.sin(a0 + 4.2 * u)], axis=1)
                 for a0 in np.linspace(0, 2 * np.pi, 40, endpoint=False) for u in [np.linspace(0, 1, 14)]]
        behind.append((blob(445, -425, 152, rng, 0.04, 0.95), swirl))
    elif "curly" in style:
        front.append((smooth(out + inn), []))
        ring = smooth([(-340, -235), (-335, -400), (-215, -545), (30, -610), (285, -565), (470, -400), (535, -150),
                       (510, 95), (425, 265), (315, 330)], closed=False, steps=4)
        for p in ring:  # soft round curls all around the edge, no two alike
            front.append((blob(p[0] + rng.normal(0, 24), p[1] + rng.normal(0, 24), rng.uniform(66, 118), rng, 0.08), []))
        for p in ring[::2]:
            q = p * 0.9 + np.array([16.0, -20.0])
            front.append((blob(q[0] + rng.normal(0, 26), q[1] + rng.normal(0, 26), rng.uniform(70, 110), rng, 0.08), []))
        for _ in range(48):  # and the same, smaller, all over
            a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, 1) ** 0.5
            p = np.array([165 + 300 * d * np.cos(a), -215 + 300 * d * np.sin(a)])
            if p[0] < 110 and p[1] > -110:
                continue  # leave the ear and the face clear
            front.append((blob(p[0], p[1], rng.uniform(48, 70), rng, 0.06), None))
    elif "long" in style:
        o = [(-374, -170), (-350, -392), (-215, -526), (30, -586), (275, -541), (450, -381), (512, -140), (520, 130),
             (500, 420), (465, 700), (400, 905), (300, 965), (190, 955)]
        i = [(110, 900), (120, 640), (95, 420), (-10, 395), (-95, 330), (-165, 220), (-215, 90), (-255, -40), (-318, -110)]
        front.append((smooth(o + i), fan((140, -450), smooth(o[5:] + i[:3], False), rng, 220, 6, 34)
                      + fan((140, -450), smooth(o[:3] + i[3:], False), rng, 110, 6)))
    else:  # a short cut with the ear showing
        if "tousled" in style:
            o = [(-388, -248), (-372, -360), (-318, -452), (-262, -500), (-205, -548), (-110, -560), (-30, -606), (70, -590),
                 (160, -612), (290, -556), (372, -500), (440, -400), (502, -300), (500, -130), (474, 80), (392, 230), (300, 285)]
            i = inn[:8] + [(-225, -150), (-285, -150), (-340, -180)]
            wob = 12
        else:
            o, i, wob = out, inn, 6
        cap = [smooth(list(o) + list(i))]
        if "tousled" in style:  # short tufts that stand off the head, part of the same mop
            edge = smooth(o[:14], False, steps=6)
            for k in range(4, len(edge) - 4, 5):
                along_edge = edge[k + 3] - edge[k - 3]
                along_edge /= np.linalg.norm(along_edge) + 1e-6
                outward = np.array([along_edge[1], -along_edge[0]])
                tip = edge[k] + outward * rng.uniform(26, 52) + along_edge * rng.uniform(-34, 34)
                cap.append(np.array([edge[k - 3] - outward * 20, tip, edge[k + 3] - outward * 20]))
        front.append((cap, fan(WHORL, smooth(o, False), rng, 170, wob) + fan(WHORL, smooth(i, False), rng, 90, wob)))
    return front, behind


def beard(f, rng):
    """A short full beard: from the ear along the jaw, proud of the chin, and a moustache.
    [(outline, strands)]"""
    jaw = smooth([(-80, 20), (-118, 120), (-205, 172), (-300, 176), f["mouth"] + (74, 6), f["mouth"] + (40, 34),
                  f["chin_dip"] + (-2, -10), f["chin"] + (-20, 2), f["chin_low"] + (-24, 28), f["jaw"] + (-8, 36),
                  f["throat"] + (56, 26), (-50, 398), (-8, 300), (4, 170), (-22, 40)])
    lip = smooth([f["under_nose"] + (10, 8), f["lip_dip"] + (-7, 2), f["upper_lip"] + (-9, -10), f["upper_lip"] + (14, 4),
                  f["mouth"] + (40, -8), f["mouth"] + (78, 2), (-296, 168), (-318, 132), f["under_nose"] + (40, 14)])
    lines = []
    for _ in range(420):
        q = np.array([rng.uniform(-440, 60), rng.uniform(20, 470)])
        lines.append(np.stack([q, q + (rng.normal(-6, 8), rng.uniform(14, 30))]))
    return [(jaw, lines), (lip, lines)]


# ---------------------------------------------------------------- one portrait
def face_points(age, tweak):
    f = {k: np.array(v, np.float64) for k, v in FACE.items()}
    for moves in (AGES[age], tweak):
        for k, (dx, dy) in moves.items():
            f[k] = f[k] + (dx, dy)
    return f


def draw(name, age, skin, hair, shirt, wall, style, tweak, seed):
    rng = np.random.default_rng(seed)
    f = face_points(age, tweak)
    skin, hair, shirt = (np.asarray(c, np.float32) for c in (skin, hair, shirt))
    yy, xx = np.mgrid[0:H * SS, 0:W * SS].astype(np.float32)

    # the wall: a plain colour, a little darker away from the window
    img = np.asarray(wall, np.float32)[None, None, :] * (1.04 - 0.10 * (xx / (W * SS)) - 0.05 * (yy / (H * SS)))[..., None]

    grow = 1.035 if age == "child" else 1.0  # a child's head is big for the face
    back = [(x * (grow if y < 300 else 1.0), y * (grow if y < 300 else 1.0)) for x, y in BACK]
    if age == "man":
        back[0], back[1] = (255, 660), (235, 500)  # a thicker neck
    head = smooth([f[k] for k in ORDER] + back)
    wide = {"child": 0.86, "woman": 0.96, "man": 1.10}[age]
    body = smooth([(-190, 590), (-245, 670), (-360 * wide, 730), (-560 * wide, 830), (-690 * wide, 1010), (-760 * wide, 1500),
                   (1200, 1500), (1060 * wide, 1040), (860 * wide, 850), (560 * wide, 745), (330, 680), (235, 600)])
    front, behind = hair_shapes(style, rng)
    whiskers = beard(f, rng) if "beard" in style else None

    everything = blank()
    for shape in [head, body] + [o for o, _ in front + behind + (whiskers or []) if o is not None]:
        fill(everything, shape)
    shadow = cv2.GaussianBlur(np.roll(everything, (18 * SS, 26 * SS), (0, 1)).astype(np.float32) / 255.0, (0, 0), 30 * SS)
    img *= (1 - 0.16 * shadow)[..., None]  # a soft shadow on the wall

    m_body = fill(blank(), body)
    put(img, m_body, lit(m_body, shirt, depth=0.16, soft=40.0))
    for outline, lines in behind:  # a ponytail or a bun, behind the head
        m = fill(blank(), outline)
        put(img, m, strands(m, hair, lines, rng))

    m_head = fill(blank(), head)
    skin_col = lit(m_head, skin, depth=0.17, soft=34.0)
    under = fill(blank(), smooth([f["chin_low"] + (10, 12), f["jaw"] + (0, 14), (-60, 430), (150, 380), (200, 520),
                                  (215, 660), f["neck_base"], f["neck_front"], f["throat"] + (0, 10)]))
    under = cv2.GaussianBlur(under, (0, 0), 14 * SS)
    skin_col *= (1 - 0.17 * (under.astype(np.float32) / 255.0))[..., None]  # the neck sits in the chin's shade
    cheek = np.exp(-(((xx - (OX - 215 * U)) / (95 * U)) ** 2 + ((yy - (OY + 150 * U)) / (70 * U)) ** 2))
    blush = 0.20 if age != "man" else 0.08
    skin_col = skin_col * (1 - blush * cheek[..., None]) + np.array([228, 120, 120], np.float32) * blush * cheek[..., None]
    put(img, m_head, skin_col)

    # the neck of the shirt, in front of the neck
    neck = smooth([(-215, 600), (-150, 640), (10, 668), (190, 645), (250, 600), (255, 690), (20, 730), (-225, 690)])
    m_neck = fill(blank(), neck)
    put(img, m_neck, lit(m_neck, shirt * 0.86, depth=0.12, soft=10.0))

    dark = skin * 0.62
    # the ear
    m_ear = fill(blank(), smooth([(40, -30), (88, -48), (118, -5), (112, 60), (84, 118), (52, 130), (40, 90), (28, 40)]))
    put(img, m_ear, lit(m_ear, skin * 0.97, depth=0.22, soft=8.0))
    stroke(img, smooth([(52, -8), (86, -22), (100, 15), (92, 62), (70, 96)], False), dark, 2)
    stroke(img, smooth([(60, 30), (76, 40), (70, 70), (56, 82)], False), dark, 2)

    # the eye, seen from the side, with its lashes and brow
    big = 1.12 if age == "child" else 1.0
    ex, ey = f["nasion"][0] + 62, f["nasion"][1] + 22

    def eye(pts):
        return [(ex + x * big, ey + y * big) for x, y in pts]
    white = fill(blank(), smooth(eye([(-34, 2), (-14, -18), (18, -12), (34, 6), (8, 14), (-18, 12)])))
    put(img, white, np.array([244, 240, 236], np.float32))
    iris = fill(blank(), smooth(eye([(-26, -2), (-16, -16), (-2, -6), (-6, 10), (-20, 10)])))
    put(img, cv2.bitwise_and(iris, white), hair * 0.5 + np.array([30, 22, 16], np.float32))
    lash = hair * 0.45
    stroke(img, smooth(eye([(-40, 2), (-16, -20), (18, -14), (40, 6)]), False), lash, 4)
    if age != "man":
        for dx, up in ((-40, -8), (-33, -20), (-22, -29)):
            stroke(img, eye([(dx + 6, -4 + up * 0.3), (dx - 8, up)]), lash, 2)
    stroke(img, smooth(eye([(-52, -52), (-20, -70), (30, -62), (58, -44)]), False), hair * 0.75 + dark * 0.25, 10 if age == "man" else 7)

    # the nostril, the lips and the line of the mouth
    nx, ny = f["under_nose"]
    cv2.ellipse(img, tuple(int(v) for v in ipx([(nx + 30, ny - 14)])[0]), (int(13 * U), int(6 * U)), 20, 0, 360,
                [float(c) for c in dark], -1, cv2.LINE_AA)
    lips = smooth([f["lip_dip"] + (4, 14), f["upper_lip"] + (3, 0), f["mouth"] + (2, 0), f["lower_lip"] + (3, 0), f["chin_dip"] + (6, -12),
                   f["lower_lip"] + (48, -6), f["mouth"] + (66, 2), f["upper_lip"] + (46, -6)])
    m_lips = cv2.bitwise_and(fill(blank(), lips), m_head)
    rosy = 0.45 if age != "man" else 0.25
    put(img, cv2.GaussianBlur(m_lips, (0, 0), 2 * SS), skin * (1 - rosy) * 0.95 + np.array([190, 84, 88], np.float32) * rosy, 0.85)
    stroke(img, smooth([f["mouth"] + (4, 0), f["mouth"] + (36, 3), f["mouth"] + (66, 0)], False), dark * 0.8, 2)

    for outline, lines in whiskers or []:
        m = fill(blank(), outline)
        put(img, cv2.GaussianBlur(m, (0, 0), 1.6 * SS), strands(m, hair * 1.1 + 8, lines, rng, 0.3, 1.4, 0.5), 0.95)

    for outline, lines in front:  # the hair on the head, over the top of the ear
        m = fill(blank(), outline)
        if lines is None:  # a curl on top of the others: only its lit edge shows
            put(img, m, lit(m, hair * 1.25 + 6, depth=0.5, soft=7.0), 0.55)
        else:
            put(img, m, strands(m, hair, lines, rng) if lines else lit(m, hair, depth=0.34, soft=12.0))
    if "ponytail" in style:  # the band that holds it
        put(img, fill(blank(), smooth([(428, -268), (468, -256), (476, -172), (436, -162)])), np.array([212, 214, 220], np.float32))

    out = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA)
    out = cv2.GaussianBlur(out, (0, 0), 0.7)  # a camera is never this sharp
    out = np.clip(out.astype(np.float32) + rng.normal(0, 2.2, out.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, *look) in enumerate(PEOPLE):
        cv2.imwrite(str(out_dir / f"{name}.jpg"), draw(name, *look, seed=100 + i), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {len(PEOPLE)} samples to {out_dir}")
