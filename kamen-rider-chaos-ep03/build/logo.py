"""Hand-built blade-stroke katakana title: カオス, rendered as chrome with a sliced edge."""
import math
import numpy as np
import cv2
from PIL import Image

SS = 3  # supersample
WEIGHT = 1.28


def bez(p0, p1, p2, n=80):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(np.array, (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2, t[:, 0]


def blade(path_pts, widths, taper_in=0.12, taper_out=0.35, cut_in=0.0):
    """Tapered stroke polygon along a quadratic bezier; widths=(start, mid, end) in glyph units."""
    pts, t = bez(*path_pts)
    w0, wm, w1 = (v * WEIGHT for v in widths)
    w = np.where(t < 0.5, w0 + (wm - w0) * (t / 0.5), wm + (w1 - wm) * ((t - 0.5) / 0.5))
    if taper_in > 0:
        w = w * np.clip(t / taper_in, 0, 1) ** 0.6
    if taper_out > 0:
        w = w * np.clip((1 - t) / taper_out, 0, 1) ** 0.75
    d = np.gradient(pts, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    left = pts + nrm * (w / 2)[:, None]
    right = pts - nrm * (w / 2)[:, None]
    if cut_in:
        # angled blunt start: slide the first left point forward along the stroke
        left[0] = left[0] + d[0] * cut_in
    return np.concatenate([left, right[::-1]], 0)


# glyph strokes in a unit box (x right, y down)
GLYPHS = {
    "カ": [
        blade([(0.00, 0.38), (0.45, 0.33), (0.90, 0.30)], (0.10, 0.12, 0.13), taper_in=0.25, taper_out=0),
        blade([(0.85, 0.27), (0.90, 0.72), (0.60, 1.00)], (0.15, 0.13, 0.08), taper_in=0, taper_out=0.4),
        blade([(0.50, -0.02), (0.44, 0.60), (0.00, 1.00)], (0.14, 0.12, 0.06), taper_in=0, taper_out=0.55, cut_in=0.06),
    ],
    "オ": [
        blade([(0.00, 0.36), (0.50, 0.31), (1.00, 0.27)], (0.10, 0.12, 0.12), taper_in=0.25, taper_out=0.12),
        blade([(0.64, -0.02), (0.66, 0.70), (0.44, 0.96)], (0.15, 0.14, 0.09), taper_in=0, taper_out=0.35, cut_in=0.06),
        blade([(0.58, 0.40), (0.34, 0.74), (0.00, 0.90)], (0.12, 0.10, 0.06), taper_in=0, taper_out=0.6),
    ],
    "ス": [
        blade([(0.06, 0.13), (0.45, 0.11), (0.84, 0.09)], (0.10, 0.12, 0.13), taper_in=0.25, taper_out=0),
        blade([(0.80, 0.06), (0.62, 0.66), (0.00, 1.00)], (0.15, 0.13, 0.06), taper_in=0, taper_out=0.55),
        blade([(0.47, 0.55), (0.78, 0.74), (1.02, 1.00)], (0.05, 0.11, 0.15), taper_in=0.3, taper_out=0.12),
    ],
}


def logo_mask(text="カオス", height=300, gap=0.04, shear=-0.16, xscale=0.94):
    H = height * SS
    pad = int(H * 0.35)
    adv = H * (xscale + gap)
    Wd = int(adv * len(text) + pad * 2)
    m = np.zeros((H + pad * 2, Wd), np.float32)
    for i, ch in enumerate(text):
        for poly in GLYPHS[ch]:
            p = poly.copy() * H
            p[:, 0] = p[:, 0] * xscale + pad + i * adv - p[:, 1] * shear  # italic lean forward
            p[:, 1] += pad
            cv2.fillPoly(m, [np.round(p * 8).astype(np.int32)], 1.0, cv2.LINE_AA, shift=3)
    return m, pad


def chrome(m, slash_angle=-0.55, violet=(0.55, 0.25, 1.0), jade=(0.2, 1.0, 0.6)):
    """Return RGBA float image of the chrome logo at supersampled scale."""
    h, w = m.shape
    mb = (m > 0.5).astype(np.uint8)
    # slice: a thin diagonal cut through the middle glyph, offset the lower part
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.5
    s = (xx - cx) * math.sin(slash_angle) - (yy - cy) * math.cos(slash_angle)
    gap = np.abs(s) < 3.2 * SS
    m = m * (~gap)
    mb = (m > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
    dn = np.clip(dist / (h * 0.045), 0, 1)
    gy, gx = np.gradient(cv2.GaussianBlur(dist, (0, 0), 2))
    # bevel lighting from upper-left
    lx, ly = -0.6, -0.8
    shade = np.clip(0.5 + 1.4 * (gx * lx + gy * ly) * (1 - dn), 0, 1)
    # chrome gradient: sky / horizon / ground reflections
    ys, ye = np.where(mb.any(1))[0][[0, -1]]
    t = np.clip((yy - ys) / max(1, ye - ys), 0, 1)
    ramp = np.interp(t, [0, 0.38, 0.47, 0.5, 0.62, 1.0], [0.98, 0.78, 0.98, 0.18, 0.34, 0.86])
    base = ramp * (0.65 + 0.55 * shade)
    rgb = np.stack([base * 0.96, base * 0.98, base * 1.03], -1)
    # colour edges: violet on the left glyph, jade on the right
    side = np.clip(xx / w * 1.6 - 0.3, 0, 1)[..., None]
    edgecol = np.array(violet) * (1 - side) + np.array(jade) * side
    rim = (np.clip(1 - dist / (2.5 * SS), 0, 1) * mb)[..., None]
    rgb = rgb * (1 - rim * 0.65) + edgecol * rim * 0.95
    # brushed grain
    rng = np.random.default_rng(5)
    br = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=18, sigmaY=0.6)
    rgb = rgb * (1 + br[..., None] * 0.18)
    # black keyline + outer coloured glow
    m_s = cv2.GaussianBlur(m, (0, 0), 0.8)
    outer = cv2.dilate(mb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13 * SS, 13 * SS))).astype(np.float32)
    outer = cv2.GaussianBlur(outer, (0, 0), 1.0)
    glow = cv2.GaussianBlur(outer, (0, 0), 10 * SS)
    a = np.clip(np.maximum(outer, glow * 0.85), 0, 1)
    col = np.zeros((h, w, 3), np.float32)
    col += (glow * 0.9)[..., None] * edgecol * 0.6
    col = col * (1 - outer[..., None])  # keyline is black
    col = col * (1 - m_s[..., None]) + np.clip(rgb, 0, 1.2) * m_s[..., None]
    # slash light: thin hot line along the cut
    sl = np.exp(-(s / (1.2 * SS)) ** 2) * np.clip(1 - np.abs((xx - cx) / (w * 0.5)), 0, 1) ** 0.5
    sl *= cv2.GaussianBlur(cv2.dilate(mb, np.ones((5 * SS, 5 * SS), np.uint8)).astype(np.float32), (0, 0), 2 * SS)
    sl *= np.exp(-((xx - cx) / (w * 0.13)) ** 4)
    col += sl[..., None] * np.array([1.0, 1.0, 1.0]) * 1.2
    a = np.maximum(a, np.clip(sl * 1.2, 0, 1))
    return np.dstack([np.clip(col, 0, 1), a])


def render(height=300):
    m, pad = logo_mask(height=height)
    rgba = chrome(m)
    h, w = m.shape
    out = cv2.resize(rgba, (w // SS, h // SS), interpolation=cv2.INTER_AREA)
    return out


if __name__ == "__main__":
    out = render(300)
    bg = np.full(out.shape[:2] + (3,), 0.25, np.float32)
    comp = bg * (1 - out[..., 3:]) + out[..., :3] * out[..., 3:]
    Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).save(
        "work/t_logo.png")
    print(out.shape)
