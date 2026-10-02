"""Typography + title pass over the rendered artwork."""
import math
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import fx
import logo

SP = os.path.dirname(os.path.abspath(__file__)) + "/"
WD = SP + "work/"
FD = SP + "fonts/"
CF = FD

VIOLET = np.array([0.62, 0.36, 1.0], np.float32)
JADE = np.array([0.30, 1.0, 0.62], np.float32)
WHITE = np.array([0.97, 0.96, 0.95], np.float32)

SLOGAN_L = ["光があるから、", "俺は闇になる。"]
SLOGAN_R = ["闇があるから、", "俺は光になる。"]
EP_TITLE = "第3話「闇ノ刃、翠ノ拳」"
TAGLINE = "闇に呑まれるか、闇を断つか。"


def font(name, size, wght=None):
    f = ImageFont.truetype(name, size)
    if wght is not None:
        try:
            f.set_variation_by_axes([wght])
        except Exception:
            pass
    return f


def blank(W, H):
    return Image.new("L", (W, H), 0)


def vertical(mask_img, cols, x_right, y_top, fnt, size, col_gap, adv=1.04):
    """Japanese vertical setting: columns read right-to-left; 、。 sit in the upper right of their cell."""
    d = ImageDraw.Draw(mask_img)
    for ci, line in enumerate(cols):
        cx = x_right - ci * col_gap
        y = y_top
        for ch in line:
            bb = d.textbbox((0, 0), ch, font=fnt)
            cw, chh = bb[2] - bb[0], bb[3] - bb[1]
            if ch in "、。":
                d.text((cx + size * 0.5 - cw - bb[0] - size * 0.06, y - bb[1] + size * 0.02), ch, font=fnt, fill=255)
                y += size * 0.62
            else:
                d.text((cx - cw / 2 - bb[0], y + (size - chh) / 2 - bb[1]), ch, font=fnt, fill=255)
                y += size * adv


def tracked(mask_img, text, x, y, fnt, track, anchor="l"):
    d = ImageDraw.Draw(mask_img)
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


def comp_text(canvas, mask_img, color, shadow=0.75, shadow_blur=10, glow=None, glow_amt=0.0, glow_blur=18):
    m = np.asarray(mask_img).astype(np.float32) / 255
    if shadow:
        s = cv2.GaussianBlur(m, (0, 0), shadow_blur)
        canvas = canvas * (1 - np.clip(s * shadow * 1.6, 0, shadow)[..., None])
    if glow is not None and glow_amt:
        g = cv2.GaussianBlur(m, (0, 0), glow_blur)
        canvas = fx.screen(canvas, np.clip(g[..., None] * glow * glow_amt, 0, 1))
    canvas = canvas * (1 - m[..., None]) + np.array(color, np.float32) * m[..., None]
    return canvas


def paste_rgba(canvas, rgba, x, y):
    h, w = rgba.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = sub[..., 3:]
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - a) + sub[..., :3] * a
    return canvas


def neon_three(height):
    m = np.asarray(Image.open(WD + "m_three.png").convert("L")).astype(np.float32) / 255
    h, w = m.shape
    s = height / h
    m = cv2.resize(m, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    pad = 60
    m = np.pad(m, pad)
    line = np.clip((m - 0.15) * 1.6, 0, 1)
    core = np.clip((cv2.GaussianBlur(m, (0, 0), 1.2) - 0.55) * 3, 0, 1)
    g1 = cv2.GaussianBlur(line, (0, 0), 6)
    g2 = cv2.GaussianBlur(line, (0, 0), 22)
    rgb = (line[..., None] * JADE * 0.95 + core[..., None] * 0.85 + g1[..., None] * JADE * 0.9
           + g2[..., None] * JADE * 0.55)
    a = np.clip(np.maximum.reduce([line, g1 * 1.1, g2 * 0.8]), 0, 1)
    rgb = np.clip(rgb / np.maximum(a[..., None], 1e-3), 0, 1)
    return np.dstack([rgb, a]), pad


def main():
    art = np.load(WD + "art.npy").astype(np.float32)
    H, W = art.shape[:2]
    cv_ = art.copy()

    # ------------------------------------------------ vertical slogans (top corners)
    kt = font(FD + "KaiseiTokumin-ExtraBold.ttf", 90)
    mL = blank(W, H)
    vertical(mL, SLOGAN_L, 270, 140, kt, 90, 110)
    cv_ = comp_text(cv_, mL, WHITE, shadow=0.8, shadow_blur=12, glow=VIOLET, glow_amt=0.55, glow_blur=26)
    mR = blank(W, H)
    vertical(mR, SLOGAN_R, W - 160, 140, kt, 90, 110)
    cv_ = comp_text(cv_, mR, WHITE, shadow=0.8, shadow_blur=12, glow=JADE, glow_amt=0.45, glow_blur=26)

    # ------------------------------------------------ title block
    lg = logo.render(height=300)
    lh, lw = lg.shape[:2]
    # locate the glyph extents inside the padded logo image
    ys, xs = np.where(lg[..., 3] > 0.6)
    gx0, gx1, gy0, gy1 = xs.min(), xs.max(), ys.min(), ys.max()
    title_cx, title_top = 840, 1700
    lx = int(title_cx - (gx0 + gx1) / 2)
    ly = int(title_top - gy0)
    # dark pool under the title for legibility
    shade = np.zeros((H, W), np.float32)
    cv2.ellipse(shade, (title_cx, title_top + 190), (720, 300), 0, 0, 360, 1.0, -1)
    shade = cv2.GaussianBlur(shade, (0, 0), 90)
    cv_ = cv_ * (1 - shade[..., None] * 0.55)
    cv_ = paste_rgba(cv_, lg, lx, ly)
    gl, gr = lx + gx0, lx + gx1
    gt, gb = ly + gy0, ly + gy1

    zom = font(FD + "ZenOldMincho-Black.ttf", 62)
    m = blank(W, H)
    tracked(m, "仮面ライダー", gl + 6, gt - 92, zom, 10)
    cv_ = comp_text(cv_, m, WHITE, shadow=0.85, shadow_blur=8, glow=VIOLET, glow_amt=0.35)

    cin = font(FD + "Cinzel[wght].ttf", 30, wght=700)
    m = blank(W, H)
    tracked(m, "KAMEN RIDER CHAOS", (gl + gr) / 2, gb + 42, cin, 15, anchor="c")
    cv_ = comp_text(cv_, m, (0.86, 0.86, 0.88), shadow=0.7, shadow_blur=6)

    # episode line with hairline rules
    ep_font = font(FD + "ZenOldMincho-Black.ttf", 52)
    m = blank(W, H)
    ep_y = gb + 104
    tw = tracked(m, EP_TITLE, (gl + gr) / 2, ep_y, ep_font, 4, anchor="c")
    d = ImageDraw.Draw(m)
    cy_rule = ep_y + 34
    for sgn in (-1, 1):
        x_in = (gl + gr) / 2 + sgn * (tw / 2 + 28)
        x_out = x_in + sgn * 120
        d.line([(x_in, cy_rule), (x_out, cy_rule)], fill=200, width=2)
    cv_ = comp_text(cv_, m, WHITE, shadow=0.85, shadow_blur=8, glow=np.array([0.8, 0.8, 1.0]), glow_amt=0.15)

    # ------------------------------------------------ episode numeral (date slot, lower right)
    three, pad = neon_three(250)
    th_, tw_ = three.shape[:2]
    tx, ty = W - 150 - (tw_ - pad * 2) - pad, 2030 - pad
    cv_ = paste_rgba(cv_, three, tx, ty)
    m = blank(W, H)
    epf = font(FD + "Cinzel[wght].ttf", 30, wght=700)
    tracked(m, "EPISODE", tx + pad + (tw_ - pad * 2) / 2, ty + pad + 250 + 26, epf, 12, anchor="c")
    cv_ = comp_text(cv_, m, (0.9, 0.92, 0.9), shadow=0.7, shadow_blur=6)

    # ------------------------------------------------ tagline + fine print
    sm = font(FD + "ShipporiMinchoB1-Bold.ttf", 44)
    m = blank(W, H)
    tracked(m, TAGLINE, W / 2, H - 132, sm, 6, anchor="c")
    cv_ = comp_text(cv_, m, WHITE, shadow=0.85, shadow_blur=8)
    fp = font(CF + "WorkSans-Regular.ttf", 17)
    m = blank(W, H)
    tracked(m, "KAMEN RIDER CHAOS  ·  EPISODE 03  ·  BLADE OF DARKNESS, FIST OF JADE", W / 2, H - 62, fp, 5,
            anchor="c")
    cv_ = comp_text(cv_, m, (0.62, 0.62, 0.64), shadow=0.5, shadow_blur=4)

    # ------------------------------------------------ unify
    cv_ = np.clip(cv_, 0, 1)
    cv_ += fx.grain(H, W, 0.012, seed=17)[..., None]
    out = Image.fromarray((np.clip(cv_, 0, 1) * 255 + 0.5).astype(np.uint8))
    out.save(os.path.join(SP, "..", "poster.png"))
    out.resize((W // 2, H // 2), Image.LANCZOS).save(WD + "poster_half.png")
    print("ok")


if __name__ == "__main__":
    main()
