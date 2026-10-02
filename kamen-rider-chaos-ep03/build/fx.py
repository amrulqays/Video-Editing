"""Procedural effects for the poster: smoke, ink, light, lightning, sparks, grain."""
import math
import numpy as np
import cv2

RNG = np.random.default_rng(7)


def fbm(h, w, base=4, octaves=6, persistence=0.55, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else RNG
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        cells = base * (2 ** o)
        gh, gw = max(2, int(cells * h / max(h, w)) + 2), max(2, int(cells * w / max(h, w)) + 2)
        g = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= persistence
    out /= tot
    out -= out.min()
    out /= max(out.max(), 1e-6)
    return out


def warped_smoke(h, w, seed=0, base=3, warp=0.18):
    """Domain-warped fBm, reads as billowing smoke."""
    rng = np.random.default_rng(seed)
    qx = fbm(h, w, base=base, octaves=5, seed=rng.integers(1e9))
    qy = fbm(h, w, base=base, octaves=5, seed=rng.integers(1e9))
    n = fbm(h, w, base=base + 1, octaves=7, seed=rng.integers(1e9))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = xx + (qx - 0.5) * warp * w
    my = yy + (qy - 0.5) * warp * h
    s = cv2.remap(n, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    s -= s.min()
    return s / max(s.max(), 1e-6)


def radial(h, w, cx, cy, r):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    return d


def gauss_glow(h, w, cx, cy, sigma, sx=1.0, sy=1.0, angle=0.0):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    ca, sa = math.cos(angle), math.sin(angle)
    u = (dx * ca + dy * sa) / sx
    v = (-dx * sa + dy * ca) / sy
    return np.exp(-(u * u + v * v) / (2 * sigma * sigma)).astype(np.float32)


# ---------------------------------------------------------------- ink

def _blob(mask, cx, cy, r, rng, rough=0.35, n=90):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    k = rng.normal(0, 1, 7)
    rr = r * (1 + rough * sum(k[i] * np.sin((i + 2) * ang + rng.random() * 6.28) / (i + 2) for i in range(7)))
    rr *= 1 + 0.08 * rng.normal(0, 1, n)
    pts = np.stack([cx + rr * np.cos(ang), cy + rr * np.sin(ang)], 1)
    cv2.fillPoly(mask, [np.round(pts * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)


def _taper(mask, x0, y0, x1, y1, w0, w1):
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy) + 1e-6
    nx, ny = -dy / L, dx / L
    pts = np.array([[x0 + nx * w0, y0 + ny * w0], [x1 + nx * w1, y1 + ny * w1],
                    [x1 - nx * w1, y1 - ny * w1], [x0 - nx * w0, y0 - ny * w0]])
    cv2.fillPoly(mask, [np.round(pts * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)


def ink_splat(h, w, cx, cy, size, seed=0, direction=None, spread=math.pi * 2,
              spikes=26, droplets=160, mist=900, core=True):
    """Ink splatter mask in [0,1]. direction biases the throw (radians)."""
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.float32)
    if core:
        _blob(m, cx, cy, size * 0.42, rng, rough=0.5)
        for _ in range(9):
            a = rng.random() * 6.28
            d = size * (0.2 + 0.35 * rng.random())
            _blob(m, cx + d * math.cos(a), cy + d * math.sin(a), size * (0.12 + 0.18 * rng.random()), rng)

    def pick_angle():
        if direction is None:
            return rng.random() * 6.28
        return direction + (rng.random() - 0.5) * spread

    for _ in range(spikes):
        a = pick_angle()
        L = size * (0.6 + 1.3 * rng.random() ** 1.6)
        w0 = size * (0.03 + 0.07 * rng.random())
        x1, y1 = cx + L * math.cos(a), cy + L * math.sin(a)
        _taper(m, cx, cy, x1, y1, w0, w0 * 0.15)
        cv2.circle(m, (int(x1 * 4), int(y1 * 4)), int(w0 * 0.9 * 4), 1.0, -1, cv2.LINE_AA, shift=2)
    for _ in range(droplets):
        a = pick_angle()
        d = size * (0.5 + 2.4 * rng.random() ** 1.3)
        r = size * 0.075 * (1 - d / (size * 3.1)) * (0.25 + rng.random())
        if r < 0.6:
            r = 0.6
        x, y = cx + d * math.cos(a), cy + d * math.sin(a)
        cv2.ellipse(m, (int(x * 4), int(y * 4)), (int(r * 4 * (1 + 0.6 * rng.random())), int(r * 4)),
                    math.degrees(a), 0, 360, 1.0, -1, cv2.LINE_AA, shift=2)
    for _ in range(mist):
        a = pick_angle()
        d = size * (0.3 + 3.2 * rng.random())
        r = 0.5 + 2.2 * rng.random() ** 3
        cv2.circle(m, (int((cx + d * math.cos(a)) * 4), int((cy + d * math.sin(a)) * 4)),
                   int(r * 4), 1.0, -1, cv2.LINE_AA, shift=2)
    # metaball merge for organic, wet edges
    b = cv2.GaussianBlur(m, (0, 0), max(1.0, size * 0.012))
    m = np.clip((b - 0.42) * 6 + 0.5, 0, 1)
    return np.maximum(m, 0)


def ink_splat2(h, w, cx, cy, size, direction, spread=1.6, seed=0, body=True, streaks=18,
               trains=10, droplets=140, mist=900, drop_scale=1.0):
    """Realistic thrown-ink splatter: noisy pooled body, tapered streaks, droplet trains, mist."""
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.float32)
    x0, y0 = int(max(0, cx - size * 4)), int(max(0, cy - size * 4))
    x1, y1 = int(min(w, cx + size * 4)), int(min(h, cy + size * 4))
    if body and x1 > x0 and y1 > y0:
        hh, ww = y1 - y0, x1 - x0
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        ca, sa = math.cos(direction), math.sin(direction)
        u = dx * ca + dy * sa
        v = -dx * sa + dy * ca
        # body elongated along the throw direction
        r = np.sqrt((u / 1.35) ** 2 + v ** 2) / size
        n = fbm(hh, ww, base=max(3, int(ww / size * 2.2)), octaves=6, seed=rng.integers(1e9))
        n2 = fbm(hh, ww, base=max(6, int(ww / size * 6)), octaves=4, seed=rng.integers(1e9))
        f = np.exp(-r ** 2 * 1.6) + (n - 0.5) * 0.9 + (n2 - 0.5) * 0.35
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], (f > 0.42).astype(np.float32))

    def ang():
        return direction + rng.normal(0, spread * 0.45)

    for _ in range(streaks):
        a = ang()
        start = size * (0.3 + 0.4 * rng.random())
        L = size * (0.5 + 2.2 * rng.random() ** 1.8)
        w0 = size * (0.015 + 0.06 * rng.random() ** 1.5)
        sx, sy = cx + start * math.cos(a), cy + start * math.sin(a)
        ex, ey = sx + L * math.cos(a), sy + L * math.sin(a)
        _taper(m, sx, sy, ex, ey, w0, w0 * 0.08)
        if rng.random() < 0.5:
            cv2.ellipse(m, (int((ex + 6 * math.cos(a)) * 4), int((ey + 6 * math.sin(a)) * 4)),
                        (int(w0 * 2.2 * 4), int(w0 * 1.1 * 4)), math.degrees(a), 0, 360, 1.0, -1,
                        cv2.LINE_AA, shift=2)
    for _ in range(trains):
        a = ang()
        d = size * (0.8 + 1.5 * rng.random())
        r0 = drop_scale * size * (0.02 + 0.04 * rng.random())
        for k in range(rng.integers(4, 12)):
            d += r0 * (2.5 + 3 * rng.random())
            r0 *= 0.86
            x, y = cx + d * math.cos(a), cy + d * math.sin(a)
            cv2.ellipse(m, (int(x * 4), int(y * 4)), (max(2, int(r0 * 1.8 * 4)), max(2, int(r0 * 4))),
                        math.degrees(a), 0, 360, 1.0, -1, cv2.LINE_AA, shift=2)
    for _ in range(droplets):
        a = ang() if rng.random() < 0.8 else rng.random() * 6.28
        d = size * (0.7 + 3.0 * rng.random() ** 1.4)
        r = max(0.8, drop_scale * size * 0.05 * (0.15 + rng.random() ** 2) * (1.6 - d / (size * 4)))
        x, y = cx + d * math.cos(a), cy + d * math.sin(a)
        el = 1 + 1.4 * rng.random()
        cv2.ellipse(m, (int(x * 4), int(y * 4)), (max(2, int(r * el * 4)), max(2, int(r * 4))),
                    math.degrees(a), 0, 360, 1.0, -1, cv2.LINE_AA, shift=2)
    for _ in range(mist):
        a = ang() if rng.random() < 0.7 else rng.random() * 6.28
        d = size * (0.5 + 4.0 * rng.random())
        r = 0.5 + 1.6 * rng.random() ** 3
        cv2.circle(m, (int((cx + d * math.cos(a)) * 4), int((cy + d * math.sin(a)) * 4)),
                   max(2, int(r * 4)), 1.0, -1, cv2.LINE_AA, shift=2)
    b = cv2.GaussianBlur(m, (0, 0), max(0.8, size * 0.006))
    return np.clip((b - 0.4) * 5 + 0.5, 0, 1)


def drips(h, w, x0, x1, y, count, length, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.float32)
    for _ in range(count):
        x = x0 + (x1 - x0) * rng.random()
        L = length * (0.2 + rng.random())
        wd = 2 + 6 * rng.random()
        _taper(m, x, y, x, y + L, wd, wd * 0.7)
        cv2.circle(m, (int(x * 4), int((y + L) * 4)), int(wd * 1.25 * 4), 1.0, -1, cv2.LINE_AA, shift=2)
    return cv2.GaussianBlur(m, (0, 0), 1.0)


# ---------------------------------------------------------------- light

def lightning(h, w, p0, p1, seed=0, depth=7, disp=0.28, branches=5, width=2.2):
    """Return (core, glow) masks for a branching bolt from p0 to p1."""
    rng = np.random.default_rng(seed)
    core = np.zeros((h, w), np.float32)

    def bolt(a, b, d, wd):
        pts = [np.array(a, float), np.array(b, float)]
        for _ in range(d):
            new = [pts[0]]
            for i in range(len(pts) - 1):
                p, q = pts[i], pts[i + 1]
                mid = (p + q) / 2
                v = q - p
                nrm = np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-6)
                mid = mid + nrm * rng.normal(0, 1) * disp * np.linalg.norm(v) * 0.5
                new += [mid, q]
            pts = new
        n = len(pts)
        for i in range(n - 1):
            t = i / n
            th = max(1, int(round(wd * (1 - 0.7 * t) * 4)))
            cv2.line(core, tuple(np.round(pts[i] * 4).astype(int)), tuple(np.round(pts[i + 1] * 4).astype(int)),
                     1.0, th, cv2.LINE_AA, shift=2)
        return pts

    main = bolt(p0, p1, depth, width)
    for _ in range(branches):
        i = rng.integers(len(main) // 6, len(main) * 5 // 6)
        s = main[i]
        v = np.array(p1, float) - np.array(p0, float)
        ang = math.atan2(v[1], v[0]) + rng.normal(0, 0.7)
        L = np.linalg.norm(v) * (0.15 + 0.3 * rng.random())
        e = s + L * np.array([math.cos(ang), math.sin(ang)])
        bolt(tuple(s), tuple(e), depth - 2, width * 0.55)
    glow = cv2.GaussianBlur(core, (0, 0), 6) * 0.9 + cv2.GaussianBlur(core, (0, 0), 22) * 1.4
    return core, glow


def sparks(h, w, cx, cy, n, rmin, rmax, seed=0, angle_bias=None, spread=6.28, length=(8, 60)):
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.float32)
    for _ in range(n):
        a = rng.random() * 6.28 if angle_bias is None else angle_bias + (rng.random() - 0.5) * spread
        d = rmin + (rmax - rmin) * rng.random() ** 1.7
        L = length[0] + (length[1] - length[0]) * rng.random() ** 2
        x0, y0 = cx + d * math.cos(a), cy + d * math.sin(a)
        x1, y1 = x0 + L * math.cos(a), y0 + L * math.sin(a)
        wd = 0.6 + 1.8 * rng.random() ** 2
        _taper(m, x1, y1, x0, y0, wd, wd * 0.15)
    return m


def shards(h, w, cx, cy, n, rmin, rmax, seed=0, smin=4, smax=26):
    """Dark angular debris flying outward; returns (fill, edge) masks."""
    rng = np.random.default_rng(seed)
    fill = np.zeros((h, w), np.float32)
    edge = np.zeros((h, w), np.float32)
    for _ in range(n):
        a = rng.random() * 6.28
        d = rmin + (rmax - rmin) * rng.random() ** 1.2
        s = smin + (smax - smin) * rng.random() ** 2.5
        x, y = cx + d * math.cos(a), cy + d * math.sin(a)
        k = rng.integers(3, 6)
        angs = np.sort(rng.random(k) * 6.28)
        rr = s * (0.4 + rng.random(k))
        pts = np.stack([x + rr * np.cos(angs), y + rr * np.sin(angs)], 1)
        ip = np.round(pts * 4).astype(np.int32)
        cv2.fillPoly(fill, [ip], 1.0, cv2.LINE_AA, shift=2)
        j = rng.integers(k)
        cv2.line(edge, tuple(ip[j]), tuple(ip[(j + 1) % k]), 1.0, 4, cv2.LINE_AA, shift=2)
    return fill, edge


def motion_blur(img, length, angle_deg):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1)
    k = cv2.warpAffine(k, M, (length, length))
    k /= k.sum()
    return cv2.filter2D(img, -1, k)


def grain(h, w, amount=0.035, seed=3):
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 1, (h, w)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 0.7)
    return g * amount


def screen(a, b):
    return 1 - (1 - a) * (1 - b)
