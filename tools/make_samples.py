"""Draw the sample profiles in samples/ (cartoons, not real people).

They are what the README screenshots and a first try of the app use, so no
real child's photo ever needs to be in this repository.

    python tools/make_samples.py
"""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "samples"
W, H = 1500, 2000

# name, hair (BGR), skin, shirt, wall, style, nose push, chin push
KIDS = [
    ("Ava", (40, 55, 80), (150, 175, 215), (160, 90, 60), (225, 228, 232), "ponytail", 0, 0),
    ("Ben", (30, 30, 35), (120, 150, 200), (70, 140, 80), (215, 225, 230), "short", 14, 10),
    ("Cleo", (70, 130, 200), (170, 195, 230), (150, 80, 150), (230, 228, 222), "bun", -8, -6),
    ("Dev", (25, 35, 60), (95, 125, 175), (60, 60, 190), (222, 230, 226), "curly", 6, 4),
    ("Eli", (110, 170, 215), (165, 190, 228), (190, 150, 60), (228, 224, 230), "short", -4, 12),
    ("Fay", (20, 25, 30), (140, 165, 205), (90, 170, 200), (226, 226, 226), "pigtail", 10, -4),
]


def draw(hair, skin, shirt, wall, style, nose, chin):
    img = np.full((H, W, 3), wall, np.uint8)
    cv2.ellipse(img, (800, 2050), (620, 520), 0, 0, 360, shirt, -1)  # shoulders
    cv2.rectangle(img, (690, 1250), (900, 1650), skin, -1)  # neck
    cv2.ellipse(img, (760, 900), (400, 470), 0, 0, 360, skin, -1)  # head, facing left
    face = np.array([[420, 700], [370, 860], [300 - nose, 960], [350, 1000], [345, 1060], [375, 1090],
                     [350, 1130], [390 - chin, 1190], [430 - chin, 1290], [600, 1330], [700, 1000]], np.int32)
    cv2.fillPoly(img, [face], skin)
    cv2.ellipse(img, (840, 800), (420, 440), 0, 150, 420, hair, -1)
    if style == "ponytail":
        cv2.ellipse(img, (1230, 1050), (110, 260), -20, 0, 360, hair, -1)
    elif style == "bun":
        cv2.circle(img, (1150, 470), 165, hair, -1)
    elif style == "pigtail":
        cv2.ellipse(img, (1120, 1330), (95, 230), 12, 0, 360, hair, -1)
    elif style == "curly":
        for a in range(150, 425, 22):
            t = np.deg2rad(a)
            cv2.circle(img, (int(840 + 430 * np.cos(t)), int(800 + 450 * np.sin(t))), 95, hair, -1)
    cv2.circle(img, (470, 890), 14, (30, 30, 30), -1)  # eye
    return cv2.resize(cv2.GaussianBlur(img, (0, 0), 2), (1200, 1600), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, *look in KIDS:
        cv2.imwrite(str(OUT / f"{name}.jpg"), draw(*look), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {len(KIDS)} samples to {OUT}")
