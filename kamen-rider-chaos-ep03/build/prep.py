"""Source prep: upscale crops, cut out riders/blade, extract the chest-emblem 3.

Run from this directory after fetching models/ and fonts/ (see README.md).
"""
import os
import subprocess
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "work")
U = os.path.join(HERE, "input", "umbrazor-sheet.jpg")
R = os.path.join(HERE, "input", "rider-sheet.webp")

# (source, crop box, output) -- Real-ESRGAN x4
CROPS = [
    (U, "295,497,718,674", "sr_uform.png"),     # Umbrazor form-change close-up
    (U, "15,505,290,652", "sr_uweapon.png"),    # Umbrazor Blade
    (R, "390,0,762,775", "sr_g34.png"),         # rider, three-quarter view
    (R, "1150,930,1448,1086", "sr_g3.png"),     # chest emblem "3"
]


def run(*args):
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def main():
    os.makedirs(W, exist_ok=True)
    for src, box, out in CROPS:
        dst = os.path.join(W, out)
        if not os.path.exists(dst):
            run("sr.py", src, box, dst)
        run("repair.py", src, box, dst)  # ncnn on CPU occasionally returns a garbage tile

    # blank the caption printed beside the blade on the character sheet
    im = Image.open(os.path.join(W, "sr_uweapon.png")).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle((540, 240, 1100, 588), fill=(0, 0, 0))
    d.rectangle((0, 560, 1100, 588), fill=(0, 0, 0))
    im.save(os.path.join(W, "sr_uweapon_clean.png"))

    # BiRefNet cut-outs, one process each (several sessions in one process crash onnxruntime here)
    for src, out, box in [("sr_uform.png", "m_uform_full.png", None),
                          ("sr_g34.png", "m_g34.png", "0,0,1488,2400"),
                          ("sr_uweapon_clean.png", "m_uweapon2.png", None)]:
        if not os.path.exists(os.path.join(W, out)):
            run("seg1.py", os.path.join(W, src), os.path.join(W, out), *([box] if box else []))

    # emblem 3: the glowing engraved outline, isolated by its green cast
    a = np.asarray(Image.open(os.path.join(W, "sr_g3.png")).convert("RGB")).astype(np.float32) / 255
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = np.clip((g - np.maximum(r, b) - 0.12) * 5, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 2.5)
    m = np.clip((m - 0.35) * 4, 0, 1)
    n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0.5).astype(np.uint8), 8)
    big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    x, y, w, h = st[big, :4]
    m = m * (cv2.dilate((lab == big).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)
    m = m[max(0, y - 20):y + h + 20, max(0, x - 20):x + w + 20]
    Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(W, "m_three.png"))
    print("prep done")


if __name__ == "__main__":
    main()
