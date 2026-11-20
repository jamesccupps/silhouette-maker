"""Draw the app icon (static/icon.ico): a profile on a wooden disc.

    python tools/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "static" / "icon.ico"
S = 1024  # drawn large, shrunk for smooth edges


def draw() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((16, 16, S - 16, S - 16), fill="#d9b27c", outline="#b08850", width=18)
    d.ellipse((S / 2 - 30, 60, S / 2 + 30, 120), fill="#5b4526")  # the hanging hole
    k = 0.56  # the sample profile, scaled into the disc

    def p(x, y):
        return (47 + x * k, -20 + y * k)


    black = "#111111"
    d.ellipse((*p(360, 430), *p(1160, 1370)), fill=black)  # head
    d.polygon([p(420, 700), p(370, 860), p(300, 960), p(350, 1000), p(345, 1060), p(375, 1090), p(350, 1130),
               p(390, 1190), p(430, 1290), p(600, 1330), p(700, 1000)], fill=black)  # face
    d.ellipse((*p(1100, 800), *p(1360, 1300)), fill=black)  # ponytail
    d.rectangle((*p(690, 1250), *p(900, 1560)), fill=black)  # neck
    d.pieslice((*p(380, 1480), *p(1210, 1800)), 180, 360, fill=black)  # bust
    return img


if __name__ == "__main__":
    draw().resize((256, 256), Image.LANCZOS).save(OUT, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote", OUT)
