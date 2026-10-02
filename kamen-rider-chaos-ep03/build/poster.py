"""Kamen Rider Chaos - Episode 3 poster compositor."""
import math
import os
import sys
import numpy as np
import cv2
from PIL import Image
import fx

WD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work") + "/"
W, H = 1800, 2546
C = np.array([985.0, 1100.0])          # impact point
DIAG = math.atan2(1290 - 820, 1290 - 620)  # clash axis, upper-left -> lower-right
ARM_POLY = [(286, 1075), (350, 1050), (424, 1062), (426, 1180), (420, 1330), (408, 1470), (362, 1580),
            (334, 1700), (326, 1860), (312, 2170), (110, 2170), (110, 1880), (160, 1770), (188, 1600),
            (238, 1470), (276, 1250)]
ARM_PIVOT = (388, 1085)
ARM_ANG = -173


def load_rgba(img, mask, crop=None, flip=False):
    im = Image.open(WD + img).convert("RGB")
    mk = Image.open(WD + mask).convert("L")
    if crop:
        im = im.crop(crop)
    if mk.size != im.size:
        mk = mk.resize(im.size, Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255
    m = np.asarray(mk).astype(np.float32) / 255
    if flip:
        a, m = a[:, ::-1].copy(), m[:, ::-1].copy()
    return a, m


def place(rgb, alpha, scale, angle_deg, src_pt, dst_pt):
    """Scale/rotate (CCW positive) a layer about src_pt and drop it at dst_pt on the canvas."""
    if scale < 1:
        h, w = alpha.shape
        nw, nh = int(round(w * scale)), int(round(h * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)
        src_pt = (src_pt[0] * nw / w, src_pt[1] * nh / h)
        scale = 1.0
    M = cv2.getRotationMatrix2D(tuple(map(float, src_pt)), angle_deg, scale)
    M[:, 2] += np.array(dst_pt) - np.array(src_pt)
    pre = rgb * alpha[..., None]
    o_rgb = cv2.warpAffine(pre, M, (W, H), flags=cv2.INTER_LANCZOS4, borderValue=0)
    o_a = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LANCZOS4, borderValue=0)
    o_a = np.clip(o_a, 0, 1)
    o_rgb = np.clip(o_rgb, 0, None)
    return o_rgb, o_a, M


def ragged_fade(alpha, axis_vals, start, end, seed, rough=0.55):
    """Fade alpha out between start->end along a coordinate field, with an ink-like ragged edge."""
    h, w = alpha.shape
    t = np.clip((axis_vals - end) / (start - end), 0, 1)  # 1 where kept, 0 where gone
    n = fx.fbm(h, w, base=10, octaves=6, seed=seed)
    k = np.clip((t + (n - 0.5) * rough - 0.25) * 3.2, 0, 1)
    return alpha * k


def over(dst, rgb_pre, a):
    return dst * (1 - a[..., None]) + rgb_pre


def lum(x):
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def rim_light(alpha, center, color, strength, radius, width=10):
    """Light the edges of a placed layer that face the impact point."""
    a = cv2.GaussianBlur(alpha, (0, 0), width * 0.6)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=5)
    nrm = np.sqrt(gx * gx + gy * gy) + 1e-6
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = center[0] - xx, center[1] - yy
    d = np.sqrt(dx * dx + dy * dy) + 1e-6
    # outward normal is -grad(alpha); light hits edges whose outward normal faces the source
    facing = np.clip((-gx * dx - gy * dy) / (nrm * d), 0, 1)
    band = np.clip(nrm / (nrm.max() * 0.25), 0, 1)
    edge = band * facing ** 1.5 * np.exp(-(d / radius) ** 2)
    inner = alpha * (1 - cv2.erode(alpha, np.ones((width, width), np.uint8)))
    edge = cv2.GaussianBlur(edge, (0, 0), 2) * np.clip(inner * 2 + 0.25, 0, 1)
    return edge[..., None] * np.array(color, np.float32)[None, None] * strength


def keep_largest(a, thr=0.3):
    n, lab, stats, _ = cv2.connectedComponentsWithStats((a > thr).astype(np.uint8), 8)
    if n <= 2:
        return a
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    keep = cv2.dilate((lab == big).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    return a * keep


def grade_fig(rgb, contrast=1.3, pivot=0.3, desat=0.3, glow_keep=True, sharpen=0.6, gain=1.0,
              tint=(1, 1, 1), clahe=0.0, glow_sat=1.0):
    if clahe:
        lab = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        cl = cv2.createCLAHE(clipLimit=clahe, tileGridSize=(8, 8))
        lab[..., 0] = cl.apply(lab[..., 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255
    l = lum(rgb)
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / (mx + 1e-4)
    glow = np.clip((sat - 0.35) * 3, 0, 1) * np.clip((mx - 0.3) * 3, 0, 1) if glow_keep else 0
    gray = l[..., None]
    k = (desat * (1 - glow))[..., None] if glow_keep else desat
    out = rgb * (1 - k) + gray * k
    out = (out - pivot) * contrast + pivot
    out = out * gain * np.array(tint, np.float32)
    if sharpen:
        bl = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = out + (out - bl) * sharpen
    return np.clip(out, 0, 1)


def main(preview=False):
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = C

    # ---------------------------------------------------------------- atmosphere
    smoke = fx.warped_smoke(H, W, seed=4, base=3, warp=0.22)
    smoke2 = fx.warped_smoke(H, W, seed=9, base=6, warp=0.15)
    d = np.sqrt((xx - cx) ** 2 + ((yy - cy) * 0.85) ** 2)
    light = np.exp(-(d / 900) ** 2) * 0.75 + np.exp(-(d / 380) ** 2) * 0.35
    base = 0.2 + 0.56 * light
    tone = base * (0.55 + 0.75 * smoke) + 0.10 * (smoke2 - 0.5)
    # vertical falloff: darker top & bottom bands (title area reads cleanly)
    v = yy / H
    tone *= 0.72 + 0.28 * np.exp(-((v - 0.45) / 0.42) ** 2)
    tone = np.clip(tone, 0, 1)
    bg = np.stack([tone * 0.92, tone * 0.95, tone * 1.04], -1)
    # energy spill: violet on the shadow side, jade on the other
    side = np.clip(((xx - cx) * math.cos(DIAG) + (yy - cy) * math.sin(DIAG)) / 900, -1, 1)
    spill = np.exp(-(d / 1100) ** 2) * smoke
    violet = np.array([0.42, 0.16, 0.85], np.float32)
    jade = np.array([0.10, 0.85, 0.48], np.float32)
    bg += (np.clip(-side, 0, 1) * spill)[..., None] * violet * 0.22
    bg += (np.clip(side, 0, 1) * spill)[..., None] * jade * 0.16

    # ---------------------------------------------------------------- back ink
    ink = np.zeros((H, W), np.float32)
    def away(x, y):
        return math.atan2(y - cy, x - cx)
    for (x, y, sz, sd, st) in [(150, 230, 300, 1, 22), (1680, 260, 250, 2, 18), (20, 1180, 300, 3, 16),
                               (90, 1900, 170, 14, 3), (1760, 1720, 200, 5, 12), (560, 120, 120, 12, 10),
                               (1240, 90, 110, 13, 9)]:
        ink = np.maximum(ink, fx.ink_splat2(H, W, x, y, sz, away(x, y), spread=1.5, seed=sd, streaks=st,
                                            trains=8, droplets=100, mist=1100, drop_scale=0.7))
    # ink flung outward by the impact along the clash axis
    ink = np.maximum(ink, fx.ink_splat2(H, W, cx, cy, 170, DIAG + math.pi, spread=1.2, seed=6, body=False,
                                        streaks=10, trains=14, droplets=150, mist=1600, drop_scale=0.45) * 0.9)
    ink = np.maximum(ink, fx.ink_splat2(H, W, cx, cy, 150, DIAG, spread=1.2, seed=16, body=False,
                                        streaks=8, trains=10, droplets=120, mist=1300, drop_scale=0.45) * 0.9)
    ink = np.maximum(ink, fx.drips(H, W, 20, 420, 0, 9, 230, seed=7))
    ink = np.maximum(ink, fx.drips(H, W, 1420, 1800, 0, 7, 180, seed=8))
    # keep the core of the flash clean
    ink *= 1 - np.exp(-(d / 260) ** 2)
    canvas = bg * (1 - ink[..., None] * 0.94)

    # ---------------------------------------------------------------- energy behind
    for i, (tgt, col, sd) in enumerate([((420, 560), violet, 21), ((260, 1250), violet, 22),
                                        ((1700, 1380), jade, 24), ((1770, 1210), jade, 26)]):
        core, glow = fx.lightning(H, W, tuple(C), tgt, seed=sd, depth=8, disp=0.32, branches=3, width=1.6)
        canvas = fx.screen(canvas, np.clip(glow[..., None] * col * 0.4 + core[..., None] * col * 0.5
                                           + core[..., None] * 0.35, 0, 1))

    # ---------------------------------------------------------------- Umbrazor
    u_rgb, u_a = load_rgba("sr_uform.png", "m_uform_full.png", flip=True)
    hu, wu = u_a.shape
    uy, ux = np.mgrid[0:hu, 0:wu].astype(np.float32)
    u_a = ragged_fade(u_a, uy, hu - 170, hu - 4, seed=31)
    # grade: deepen blacks, keep violet glow
    u_rgb = grade_fig(u_rgb, contrast=1.3, pivot=0.16, desat=0.45, glow_keep=True, sharpen=0.9, clahe=2.6,
                      gain=1.12)
    u_rgb_p, u_a_p, _ = place(u_rgb, u_a, 1.22, -12, (971, 330), (610, 830))

    # ---------------------------------------------------------------- Umbrazor Blade
    w_rgb, w_a = load_rgba("sr_uweapon_clean.png", "m_uweapon2.png", flip=True)
    w_rgb = grade_fig(w_rgb, contrast=1.25, pivot=0.2, desat=0.35, glow_keep=True, sharpen=0.8, clahe=2.0)
    w_rgb_p, w_a_p, _ = place(w_rgb, w_a, 1.25, 39, (1097, 430), tuple(C + np.array([-6, 4])))

    # ---------------------------------------------------------------- green rider
    g_rgb, g_a = load_rgba("sr_g34.png", "m_g34.png", crop=(0, 0, 1488, 2400))
    hg, wg = g_a.shape
    gy, gx_ = np.mgrid[0:hg, 0:wg].astype(np.float32)
    # lift the near arm off the body so it can be raised into a block against the blade
    arm_m = np.zeros_like(g_a)
    cv2.fillPoly(arm_m, [np.array(ARM_POLY, np.int32)], 1.0, cv2.LINE_AA)
    arm_m = cv2.GaussianBlur(arm_m, (0, 0), 1.2)
    arm_a = keep_largest(g_a * arm_m)
    g_a = g_a * (1 - arm_m)
    # soft contact shadow where the arm used to sit against the ribs
    cut = cv2.GaussianBlur(arm_m, (0, 0), 14) * g_a
    g_rgb = g_rgb * (1 - 0.55 * cut[..., None])
    g_a = ragged_fade(g_a, gy, 1560, 2160, seed=32, rough=0.7)
    g_rgb = grade_fig(g_rgb, contrast=1.3, pivot=0.45, desat=0.3, glow_keep=True, sharpen=0.6,
                      gain=0.74, tint=(0.88, 0.93, 1.0), clahe=1.6)
    G_S, G_ROT, G_HEAD_SRC, G_HEAD_DST = 0.86, 9, np.array([700, 520]), np.array([1300, 1330])
    g_rgb_p, g_a_p, _ = place(g_rgb, g_a, G_S, G_ROT, tuple(G_HEAD_SRC), tuple(G_HEAD_DST))
    th = math.radians(G_ROT)
    v = (np.array(ARM_PIVOT) - G_HEAD_SRC) * G_S
    pivot_dst = G_HEAD_DST + np.array([v[0] * math.cos(th) + v[1] * math.sin(th),
                                       -v[0] * math.sin(th) + v[1] * math.cos(th)])
    arm_rgb_p, arm_a_p, _ = place(g_rgb, arm_a, G_S, G_ROT + ARM_ANG, ARM_PIVOT, tuple(pivot_dst))

    # lighting falloff on figures: brighter toward the impact
    fl = 0.45 + 0.75 * np.exp(-(d / 800) ** 2)

    # afterimages (motion trails) behind each rider
    for rgbp, ap, ang, L, op in [(u_rgb_p, u_a_p, math.degrees(DIAG), 90, 0.35),
                                 (g_rgb_p, np.maximum(g_a_p, arm_a_p), math.degrees(DIAG), 90, 0.35)]:
        tr = fx.motion_blur(ap, L, -ang)
        canvas = canvas * (1 - (tr * op)[..., None])

    canvas = over(canvas, arm_rgb_p * fl[..., None], arm_a_p)
    canvas += rim_light(arm_a_p, C, (0.75, 1.0, 0.85), 1.4, 700) * arm_a_p[..., None]
    canvas = over(canvas, g_rgb_p * fl[..., None], g_a_p)
    canvas += rim_light(g_a_p, C, (0.75, 1.0, 0.85), 1.4, 900) * g_a_p[..., None]
    canvas = over(canvas, u_rgb_p * fl[..., None], u_a_p)
    canvas += rim_light(u_a_p, C, (0.85, 0.75, 1.0), 1.6, 900) * u_a_p[..., None]
    canvas = over(canvas, w_rgb_p * fl[..., None], w_a_p)
    canvas += rim_light(w_a_p, C, (0.9, 0.85, 1.0), 1.4, 700) * w_a_p[..., None]

    # ---------------------------------------------------------------- impact
    burst = (fx.gauss_glow(H, W, cx, cy, 16) * 1.6 + fx.gauss_glow(H, W, cx, cy, 55) * 0.9
             + fx.gauss_glow(H, W, cx, cy, 170) * 0.45 + fx.gauss_glow(H, W, cx, cy, 420) * 0.2)
    streak = fx.gauss_glow(H, W, cx, cy, 14, sx=14, sy=1, angle=DIAG + math.pi / 2) * 0.55
    streak += fx.gauss_glow(H, W, cx, cy, 7, sx=38, sy=1, angle=-0.08) * 0.3
    rays = fx.sparks(H, W, cx, cy, 70, 10, 120, seed=41, length=(60, 360))
    rays = cv2.GaussianBlur(rays, (0, 0), 1.2)
    sp = fx.sparks(H, W, cx, cy, 280, 50, 620, seed=42, length=(6, 64))
    tint = np.ones((H, W, 3), np.float32)
    tint += (np.clip(-side * 3, 0, 1))[..., None] * (violet - 1) * 0.6
    tint += (np.clip(side * 3, 0, 1))[..., None] * (jade - 1) * 0.6
    light_add = (burst + streak + rays * 0.7)[..., None] * tint
    light_add[..., :] += (fx.gauss_glow(H, W, cx, cy, 30) * 1.2)[..., None]  # white-hot core
    canvas = fx.screen(canvas, np.clip(light_add, 0, 1)) + np.clip(light_add - 1, 0, None) * 0.3
    canvas = fx.screen(canvas, np.clip((cv2.GaussianBlur(sp, (0, 0), 0.8) * 1.4)[..., None] * tint, 0, 1))
    canvas = fx.screen(canvas, np.clip(cv2.GaussianBlur(sp, (0, 0), 5)[..., None] * tint * 0.6, 0, 1))

    # debris
    fill, edge = fx.shards(H, W, cx, cy, 90, 140, 900, seed=43, smin=4, smax=24)
    fill = fx.motion_blur(fill, 9, -math.degrees(DIAG))
    canvas = canvas * (1 - fill[..., None] * 0.85)
    canvas = fx.screen(canvas, (cv2.GaussianBlur(edge, (0, 0), 0.6) * 0.6)[..., None] * tint)

    # foreground ink specks over everything
    fg = fx.ink_splat2(H, W, cx, cy, 260, DIAG + math.pi * 0.75, spread=1.0, seed=51, body=False, streaks=0,
                       trains=0, droplets=0, mist=260)
    canvas = canvas * (1 - fg[..., None] * 0.9)

    # ---------------------------------------------------------------- bloom
    canvas = np.clip(canvas, 0, 1.5)
    hi = np.clip(canvas - 0.62, 0, None)
    mx = canvas.max(-1, keepdims=True)
    sat = (mx - canvas.min(-1, keepdims=True)) / (mx + 1e-4)
    hi = hi + canvas * np.clip((sat - 0.45) * 2, 0, 1) * np.clip((mx - 0.35) * 2, 0, 1) * 0.6
    bloom = cv2.GaussianBlur(hi, (0, 0), 9) * 0.55 + cv2.GaussianBlur(hi, (0, 0), 40) * 0.45
    canvas = fx.screen(np.clip(canvas, 0, 1), np.clip(bloom, 0, 1))

    # ---------------------------------------------------------------- grade
    canvas = np.clip(canvas, 0, 1)
    vig = np.clip(1 - (np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.47) / (H * 0.6)) ** 2)) ** 2.4, 0, 1)
    canvas *= (0.5 + 0.5 * vig)[..., None]
    # bottom fog/shade for the title block
    shade = np.clip((yy - H * 0.62) / (H * 0.38), 0, 1) ** 1.3
    canvas *= (1 - shade * 0.55)[..., None]
    canvas = np.clip(canvas, 0, 1)
    # filmic curve
    canvas = canvas ** 0.95
    canvas = canvas * canvas * (3 - 2 * canvas) * 0.5 + canvas * 0.5
    # split tone: cool graphite shadows, neutral highlights
    lm = lum(canvas)[..., None]
    canvas = canvas + (1 - lm) ** 2 * np.array([-0.012, 0.0, 0.022], np.float32)
    canvas += fx.grain(H, W, 0.028)[..., None]
    out = Image.fromarray((np.clip(canvas, 0, 1) * 255 + 0.5).astype(np.uint8))
    out.save(WD + "draft.png")
    out.resize((W // 2, H // 2), Image.LANCZOS).save(WD + "draft_half.png")
    np.save(WD + "art.npy", canvas.astype(np.float16))
    print("ok")


if __name__ == "__main__":
    main()
