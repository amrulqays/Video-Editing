"""Kamen Rider Chaos - Episode 3, 21:9 key art (3360x1440)."""
import math
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import fx

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "input") + "/"
WD = os.path.join(HERE, "work", "banner") + "/"
FD = os.path.join(HERE, "fonts") + "/"
W, H = 3360, 1440

# ---------------------------------------------------------------- copy (edit here)
TAGLINE_JP = "光があるから、俺は闇になる。"
TAGLINE_EN = "BECAUSE THERE IS LIGHT,  I BECOME THE DARK"
EP_TITLE_JP = "闇ノ刃、翠ノ拳"
EP_TITLE_EN = "BLADE OF DARKNESS · FIST OF JADE"
EPISODE = "episode 3"
DATE = "5.8.2026"
TIME = "5.30pm"
BILLING = "KAMEN RIDER CHAOS   VS   UMBRAZOR"

GREEN = np.array([0.22, 1.0, 0.62], np.float32)
VIOLET = np.array([0.58, 0.30, 1.0], np.float32)
WHITE = np.array([0.96, 0.96, 0.95], np.float32)

# ---------------------------------------------------------------- placement
CH_S, CH_X0, CH_EYE_Y = 1.75, 0, 470          # Chaos helmet (orig px -> canvas)
UM_S, UM_FACE = 1.85, (2690, 500)             # Umbrazor, face centre on canvas
UM_FACE_SRC = (410, 330)                      # in crop (380,0,1800,821) coords
TN_S, TN_CX, TN_FEET = 0.88, 1680, 905        # tunnel rider scale / centre / feet line
TN_RIDER_SRC = (925, 770)
LOGO_W = 960


def load(path, mode="RGB"):
    return np.asarray(Image.open(path).convert(mode)).astype(np.float32) / 255


def lum(x):
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def affine_place(rgb, a, M, size=(W, H)):
    pre = rgb * a[..., None]
    o = cv2.warpAffine(pre, M, size, flags=cv2.INTER_AREA if M[0, 0] < 1 else cv2.INTER_LANCZOS4)
    oa = cv2.warpAffine(a, M, size, flags=cv2.INTER_AREA if M[0, 0] < 1 else cv2.INTER_LANCZOS4)
    return np.clip(o, 0, None), np.clip(oa, 0, 1)


def scale_to(rgb, a, s):
    h, w = a.shape
    nw, nh = int(round(w * s)), int(round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4
    return cv2.resize(rgb, (nw, nh), interpolation=interp), cv2.resize(a, (nw, nh), interpolation=interp)


def put(canvas, rgb, a, x, y):
    """Composite a premultiplied-free layer at integer offset (clipped)."""
    h, w = a.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return canvas
    sr, sa = rgb[y0 - y:y1 - y, x0 - x:x1 - x], a[y0 - y:y1 - y, x0 - x:x1 - x, None]
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - sa) + sr * sa
    return canvas


def layer(rgb, a, x, y):
    """Full-canvas premultiplied rgb + alpha from a positioned layer."""
    L = np.zeros((H, W, 3), np.float32)
    A = np.zeros((H, W), np.float32)
    h, w = a.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    L[y0:y1, x0:x1] = rgb[y0 - y:y1 - y, x0 - x:x1 - x] * a[y0 - y:y1 - y, x0 - x:x1 - x, None]
    A[y0:y1, x0:x1] = a[y0 - y:y1 - y, x0 - x:x1 - x]
    return L, A


def over(canvas, L, A):
    return canvas * (1 - A[..., None]) + L


def grade(rgb, contrast=1.35, pivot=0.25, black=0.02, gain=1.0, desat=0.0, keep_glow=True, sharpen=0.5):
    l = lum(rgb)[..., None]
    mx, mn = rgb.max(-1, keepdims=True), rgb.min(-1, keepdims=True)
    sat = (mx - mn) / (mx + 1e-4)
    glow = np.clip((sat - 0.35) * 3, 0, 1) * np.clip((mx - 0.3) * 3, 0, 1) if keep_glow else 0
    out = rgb * (1 - desat * (1 - glow)) + l * desat * (1 - glow)
    out = (out - pivot) * contrast + pivot
    out = np.clip((out - black) / (1 - black), 0, None) * gain
    if sharpen:
        out = out + (out - cv2.GaussianBlur(out, (0, 0), 1.6)) * sharpen
    return np.clip(out, 0, 1.2)


def rim(alpha, src, color, strength, radius, width=12):
    a = cv2.GaussianBlur(alpha, (0, 0), width * 0.6)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=5)
    nrm = np.sqrt(gx * gx + gy * gy) + 1e-6
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = src[0] - xx, src[1] - yy
    d = np.sqrt(dx * dx + dy * dy) + 1e-6
    facing = np.clip((-gx * dx - gy * dy) / (nrm * d), 0, 1)
    band = np.clip(nrm / (np.percentile(nrm, 99.5) + 1e-6), 0, 1)
    inner = alpha * (1 - cv2.erode(alpha, np.ones((width, width), np.uint8)))
    e = cv2.GaussianBlur(band * facing ** 2 * np.exp(-(d / radius) ** 2), (0, 0), 2) * np.clip(inner * 2, 0, 1)
    return e[..., None] * np.array(color, np.float32) * strength


def soft_rect_fade(h, w, x0, y0, feather):
    """1 everywhere except the region x>x0 & y>y0 (removed), with a feathered rounded corner."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - x0, yy - y0
    inside = np.minimum(dx, dy)
    return np.clip(0.5 - inside / feather, 0, 1)


def text_mask(draw_fn):
    im = Image.new("L", (W, H), 0)
    draw_fn(ImageDraw.Draw(im))
    return np.asarray(im).astype(np.float32) / 255


def tracked(d, text, x, y, fnt, track, anchor="l"):
    widths = [d.textlength(c, font=fnt) for c in text]
    total = sum(widths) + track * (len(text) - 1)
    if anchor == "c":
        x -= total / 2
    elif anchor == "r":
        x -= total
    for c, w in zip(text, widths):
        d.text((x, y), c, font=fnt, fill=255)
        x += w + track
    return total


def font(name, size, wght=None):
    f = ImageFont.truetype(FD + name, size)
    if wght is not None:
        f.set_variation_by_axes([wght])
    return f


def ink_text(canvas, m, color, shadow=0.8, sblur=8, glow=None, gamt=0.0, gblur=16):
    if shadow:
        s = cv2.GaussianBlur(m, (0, 0), sblur)
        canvas = canvas * (1 - np.clip(s * 1.8, 0, shadow)[..., None])
    if glow is not None and gamt:
        g = cv2.GaussianBlur(m, (0, 0), gblur)
        canvas = fx.screen(canvas, np.clip(g[..., None] * glow * gamt, 0, 1))
    return canvas * (1 - m[..., None]) + np.array(color, np.float32) * m[..., None]


def main():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # ------------------------------------------------------------ void
    smoke = fx.warped_smoke(H, W, seed=21, base=4, warp=0.2)
    side = np.clip((xx - W / 2) / (W * 0.35), -1, 1)
    base = 0.010 + 0.05 * smoke ** 2
    canvas = np.stack([base, base, base * 1.15], -1)
    canvas += (np.clip(-side, 0, 1) * smoke ** 2)[..., None] * GREEN * 0.05
    canvas += (np.clip(side, 0, 1) * smoke ** 2)[..., None] * VIOLET * 0.07

    # ------------------------------------------------------------ HUD crest, a hologram in the dark behind the rider
    crest = load(IMG + "hud-crest.webp")
    ch, cw = crest.shape[:2]
    cy_, cx_ = np.mgrid[0:ch, 0:cw].astype(np.float32)
    rad = np.sqrt((cx_ - cw / 2) ** 2 + (cy_ - ch / 2) ** 2) / (cw / 2)
    crest = crest * np.clip((0.86 - rad) / 0.12, 0, 1)[..., None]  # drop the corner readouts
    cs = 1080 / cw
    crest = cv2.resize(crest, (int(cw * cs), int(ch * cs)), interpolation=cv2.INTER_AREA)
    holo = np.zeros((H, W, 3), np.float32)
    hx, hy = int(1680 - crest.shape[1] / 2), int(560 - crest.shape[0] / 2)
    y0, y1 = max(0, hy), min(H, hy + crest.shape[0])
    holo[y0:y1, hx:hx + crest.shape[1]] = crest[y0 - hy:y1 - hy]
    holo = holo * np.array([0.6, 1.0, 0.85], np.float32)
    canvas = fx.screen(canvas, np.clip(holo * 0.30 + cv2.GaussianBlur(holo, (0, 0), 14) * 0.25, 0, 1))

    # ------------------------------------------------------------ tunnel: rider walking out with his machine
    tun = load(IMG + "tunnel.webp")
    tm = load(WD + "m_tunnel.png", "L")
    tbg = tun ** 1.9 * 0.55                       # crush the concrete, keep the lamps
    tbg = cv2.GaussianBlur(tbg, (0, 0), 1.5)
    tfg = grade(tun, contrast=1.45, pivot=0.32, black=0.03, gain=1.0, desat=0.15, sharpen=0.6)
    th, tw = tm.shape
    ty_, tx_ = np.mgrid[0:th, 0:tw].astype(np.float32)
    # darken the car body relative to the rider so he leads
    rider_zone = np.exp(-(((tx_ - TN_RIDER_SRC[0]) / 190) ** 2))
    tfg = tfg * (0.55 + 0.45 * rider_zone)[..., None]
    tun_rgb = tbg * (1 - tm[..., None]) + tfg * tm[..., None]
    tun_rgb, _ = scale_to(tun_rgb, tm, TN_S)
    th2, tw2 = tun_rgb.shape[:2]
    tx0 = int(TN_CX - TN_RIDER_SRC[0] * TN_S)
    ty0 = int(TN_FEET - TN_RIDER_SRC[1] * TN_S)
    vy, vx = np.mgrid[0:th2, 0:tw2].astype(np.float32)
    vig = np.clip(1 - (np.abs((vx - TN_RIDER_SRC[0] * TN_S) / (tw2 * 0.42))) ** 3, 0, 1)
    vig *= np.clip(vy / (th2 * 0.25), 0, 1) ** 1.2 * np.clip((th2 - vy) / (th2 * 0.12), 0, 1)
    tm2 = cv2.resize(tm, (tw2, th2), interpolation=cv2.INTER_AREA)
    tun_a = np.maximum(vig, tm2 * np.clip(vig * 3, 0, 1))
    L, A = layer(tun_rgb, tun_a, tx0, ty0)
    canvas = over(canvas, L, A)
    _, rider_a = layer(tun_rgb, tm2 * np.clip(vig * 3, 0, 1), tx0, ty0)

    # ------------------------------------------------------------ Umbrazor (right), shadow side
    um = load(WD + "sr_umb.png")
    uma = cv2.resize(load(WD + "m_umb.png", "L"), (um.shape[1], um.shape[0]), interpolation=cv2.INTER_LINEAR)
    um = grade(um, contrast=1.5, pivot=0.12, black=0.025, gain=1.15, desat=0.25, sharpen=0.4)
    s = UM_S / 4
    um, uma = scale_to(um, uma, s)
    uh, uw = uma.shape
    ux0 = int(UM_FACE[0] - UM_FACE_SRC[0] * UM_S)
    uy0 = int(UM_FACE[1] - UM_FACE_SRC[1] * UM_S)
    uy_, ux_ = np.mgrid[0:uh, 0:uw].astype(np.float32)
    # dissolve the lower body into the void
    n = fx.fbm(uh, uw, base=8, octaves=6, seed=33)
    fade = np.clip(((uh - uy_) / (uh * 0.28) + (n - 0.5) * 0.7), 0, 1)
    uma = uma * fade
    L, A = layer(um, uma, ux0, uy0)
    canvas = over(canvas, L, A)
    canvas += rim(A, (1680, 520), VIOLET * 0.9 + 0.1, 0.9, 1400) * A[..., None]

    # ------------------------------------------------------------ Kamen Rider Chaos (left), light side
    cr = load(WD + "sr_chaos.png")
    cra = cv2.resize(load(WD + "m_chaos.png", "L"), (cr.shape[1], cr.shape[0]), interpolation=cv2.INTER_LINEAR)
    hh, ww = cra.shape
    k = ww / 760.0
    # cut away the episode-1 lettering that sits over the helmet's lower right
    cra = cra * soft_rect_fade(hh, ww, 505 * k, 590 * k, 70 * k) * soft_rect_fade(hh, ww, 690 * k, 450 * k, 50 * k)
    gy_, gx_ = np.mgrid[0:hh, 0:ww].astype(np.float32)
    cra = cra * np.clip((745 * k - gx_) / (40 * k), 0, 1)
    cr = grade(cr, contrast=1.4, pivot=0.2, black=0.03, gain=1.05, desat=0.2, sharpen=0.3)
    cr, cra = scale_to(cr, cra, CH_S / 4)
    ch2 = cra.shape[0]
    cy0 = int(CH_EYE_Y - 300 * CH_S)
    n = fx.fbm(cra.shape[0], cra.shape[1], base=8, octaves=6, seed=34)
    ry = np.mgrid[0:cra.shape[0], 0:cra.shape[1]][0].astype(np.float32)
    cra = cra * np.clip(((ch2 - ry) / (ch2 * 0.18) + (n - 0.5) * 0.6), 0, 1)
    L, A = layer(cr, cra, CH_X0, cy0)
    canvas = over(canvas, L, A)
    canvas += rim(A, (1680, 520), GREEN * 0.8 + 0.2, 0.7, 1400) * A[..., None]

    # ------------------------------------------------------------ the clash: jade and violet mist meeting behind him
    mist = fx.warped_smoke(H, W, seed=44, base=3, warp=0.3)
    mist2 = fx.warped_smoke(H, W, seed=45, base=5, warp=0.25)
    band = np.exp(-((yy - 560) / 430) ** 2)
    gm = np.clip((1720 - xx) / 260, 0, 1) * np.clip((xx - 1080) / 260, 0, 1)
    vm = np.clip((xx - 1640) / 260, 0, 1) * np.clip((2300 - xx) / 300, 0, 1)
    tex = np.clip(mist * 1.3 - 0.35, 0, 1) ** 1.6 * (0.6 + 0.6 * mist2)
    keep = (1 - rider_a * 0.92)
    canvas = fx.screen(canvas, np.clip((tex * band * gm * keep)[..., None] * GREEN * 0.5, 0, 1))
    canvas = fx.screen(canvas, np.clip((tex * band * vm * keep)[..., None] * VIOLET * 0.6, 0, 1))
    # where the two meet: a thin seam of white light behind the rider
    seam = fx.gauss_glow(H, W, 1680, 520, 1, sx=7, sy=330) * 0.55 + fx.gauss_glow(H, W, 1680, 520, 1, sx=60, sy=420) * 0.12
    seam *= np.clip((yy - 200) / 160, 0, 1)  # keep the tagline clean
    canvas = fx.screen(canvas, np.clip((seam * keep)[..., None] * np.array([0.85, 0.95, 1.0]), 0, 1))

    # ------------------------------------------------------------ embers drifting through the dark
    rng = np.random.default_rng(12)
    emb = np.zeros((H, W, 3), np.float32)
    for i in range(170):
        x = rng.uniform(900, 2500)
        y = rng.uniform(120, 1150)
        col = GREEN if (x + rng.normal(0, 260)) < 1680 else VIOLET
        r = rng.uniform(0.8, 2.6) if rng.random() < 0.85 else rng.uniform(5, 12)
        dot = np.zeros((H, W), np.float32)
        cv2.circle(dot, (int(x * 4), int(y * 4)), max(2, int(r * 4)), 1.0, -1, cv2.LINE_AA, shift=2)
        blur = 0.6 if r < 3 else r * 0.7
        amp = rng.uniform(0.35, 1.0) if r < 3 else rng.uniform(0.12, 0.25)
        emb += (cv2.GaussianBlur(dot, (0, 0), blur) * amp)[..., None] * (col * 0.8 + 0.2)
    canvas = fx.screen(np.clip(canvas, 0, 1), np.clip(emb + cv2.GaussianBlur(emb, (0, 0), 6) * 0.8, 0, 1))

    # ------------------------------------------------------------ light: eye flares, bloom
    flare = np.zeros((H, W, 3), np.float32)
    ue = (ux0 + 450 * UM_S, uy0 + 380 * UM_S)
    for (ex, ey, col, ln) in [(780, CH_EYE_Y - 10, GREEN, 1.0), (ue[0], ue[1], VIOLET, 1.0)]:
        f = fx.gauss_glow(H, W, ex, ey, 5, sx=78, sy=1) * 0.55 + fx.gauss_glow(H, W, ex, ey, 60) * 0.25
        flare += f[..., None] * col * ln
    canvas = fx.screen(np.clip(canvas, 0, 1), np.clip(flare, 0, 1))
    hi = np.clip(canvas - 0.55, 0, None)
    mx = canvas.max(-1, keepdims=True)
    sat = (mx - canvas.min(-1, keepdims=True)) / (mx + 1e-4)
    hi = hi + canvas * np.clip((sat - 0.45) * 2, 0, 1) * np.clip((mx - 0.3) * 2, 0, 1) * 0.7
    bloom = cv2.GaussianBlur(hi, (0, 0), 10) * 0.5 + cv2.GaussianBlur(hi, (0, 0), 48) * 0.55
    canvas = fx.screen(canvas, np.clip(bloom, 0, 1))

    # ------------------------------------------------------------ title block (bottom centre)
    pool = fx.gauss_glow(H, W, 1680, 1210, 1, sx=620, sy=260)
    canvas = canvas * (1 - pool[..., None] * 0.72)
    lg = load(WD + "logo_src.png")
    lga = load(WD + "m_logo.png", "L")
    lg = grade(lg, contrast=1.15, pivot=0.3, black=0.0, gain=1.05, sharpen=0.35)
    ls = LOGO_W / lg.shape[1]
    lg, lga = scale_to(lg, lga, ls)
    lh, lw = lga.shape
    lx0, ly0 = int(1680 - lw / 2), H - lh - 108
    # contact shadow + faint green aura so the chrome sits in the scene
    L, A = layer(np.zeros_like(lg), lga, lx0, ly0 + 14)
    canvas = canvas * (1 - cv2.GaussianBlur(A, (0, 0), 16)[..., None] * 0.85)
    L, A = layer(lg, lga, lx0, ly0)
    aura = cv2.GaussianBlur(A, (0, 0), 30)
    canvas = fx.screen(canvas, np.clip(aura[..., None] * GREEN * 0.22, 0, 1))
    canvas = over(canvas, L, A)

    # ------------------------------------------------------------ grade before type
    canvas = np.clip(canvas, 0, 1)
    v = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.5) / (H * 0.78)) ** 2)
    canvas *= np.clip(1.08 - v ** 2.2 * 0.75, 0.25, 1)[..., None]
    canvas = np.clip(canvas, 0, 1)
    canvas = canvas * canvas * (3 - 2 * canvas) * 0.55 + canvas * 0.45   # S-curve: high contrast
    canvas += fx.grain(H, W, 0.022, seed=5)[..., None]
    canvas = np.clip(canvas, 0, 1)

    # ------------------------------------------------------------ typography
    kt = font("KaiseiTokumin-ExtraBold.ttf", 46)
    m = text_mask(lambda d: tracked(d, TAGLINE_JP, W / 2, 64, kt, 10, "c"))
    canvas = ink_text(canvas, m, WHITE, glow=VIOLET, gamt=0.35)
    cz = font("Cinzel[wght].ttf", 19, 600)
    m = text_mask(lambda d: tracked(d, TAGLINE_EN, W / 2, 134, cz, 9, "c"))
    canvas = ink_text(canvas, m, (0.72, 0.72, 0.74), shadow=0.6, sblur=5)

    cg = font("CormorantGaramond[wght].ttf", 66, 600)
    m = text_mask(lambda d: tracked(d, EPISODE, W / 2, H - 128, cg, 3, "c"))
    canvas = ink_text(canvas, m, WHITE, shadow=0.85, sblur=8, glow=GREEN, gamt=0.18)

    # flanks: episode title left of the mark, date right of it
    flank_y = ly0 + int(lh * 0.38)
    zom = font("ZenOldMincho-Black.ttf", 50)
    xl = lx0 - 40
    m = text_mask(lambda d: tracked(d, EP_TITLE_JP, xl, flank_y, zom, 8, "r"))
    canvas = ink_text(canvas, m, WHITE, glow=VIOLET, gamt=0.25)
    cz2 = font("Cinzel[wght].ttf", 17, 600)
    m = text_mask(lambda d: tracked(d, EP_TITLE_EN, xl, flank_y + 78, cz2, 6, "r"))
    canvas = ink_text(canvas, m, (0.7, 0.7, 0.72), shadow=0.6, sblur=4)

    xr = lx0 + lw + 40
    cgd = font("CormorantGaramond[wght].ttf", 60, 600)
    m = text_mask(lambda d: tracked(d, DATE, xr, flank_y - 8, cgd, 3, "l"))
    canvas = ink_text(canvas, m, WHITE, glow=GREEN, gamt=0.2)
    cgt = font("CormorantGaramond[wght].ttf", 44, 500)
    m = text_mask(lambda d: tracked(d, TIME, xr + 2, flank_y + 62, cgt, 3, "l"))
    canvas = ink_text(canvas, m, (0.86, 0.86, 0.86), shadow=0.7, sblur=6)

    bf = font("Oswald[wght].ttf", 15, 300)
    m = text_mask(lambda d: tracked(d, BILLING, W / 2, H - 44, bf, 5, "c"))
    canvas = ink_text(canvas, m, (0.45, 0.45, 0.47), shadow=0.0)

    out = Image.fromarray((np.clip(canvas, 0, 1) * 255 + 0.5).astype(np.uint8))
    out.save(os.path.join(HERE, "..", "banner-21x9.png"))
    out.resize((W // 2, H // 2), Image.LANCZOS).save(WD + "banner_half.png")
    print("ok")


if __name__ == "__main__":
    main()
