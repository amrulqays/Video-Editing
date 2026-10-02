"""Render the UMBRAZOR fight edit (video only) to an intermediate file.

Pipeline per output frame:
  retime (motion-blur for speed-ups, optical-flow interpolation for slow-mo)
  -> zoom / punch / shake transform -> chromatic aberration
  -> grade (contrast curve, split-tone, saturation) -> bloom -> flash -> fades
  -> vignette -> upscale -> film grain
then a title card with embers, light streak and RGB-split slam-in.
"""
import os, sys, subprocess, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edl import SEGMENTS, TITLE_FRAMES, FPS, build_timeline, speed_at

OUT_W, OUT_H = 1920, 822
SRC_W, SRC_H = 1470, 630
PREVIEW = '--preview' in sys.argv
OUTFILE = 'video_preview.mp4' if PREVIEW else 'video_main.mp4'
if PREVIEW:
    OUT_W, OUT_H = 960, 412

BLACK_GAP = 8          # frames of hard black before the title slam
rng = np.random.default_rng(7)


def load(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'rawvideo',
                          '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, SRC_H, SRC_W, 3)


print('loading sources...', flush=True)
SRC = {'A': load('A.mp4'), 'B': load('B.mp4')}

# ---------------------------------------------------------------- retiming
_flow_cache = {}
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def flows(src, i):
    key = (src, i)
    if key not in _flow_cache:
        f0 = cv2.cvtColor(SRC[src][i], cv2.COLOR_RGB2GRAY)
        f1 = cv2.cvtColor(SRC[src][i + 1], cv2.COLOR_RGB2GRAY)
        s0 = cv2.resize(f0, (SRC_W // 2, SRC_H // 2), interpolation=cv2.INTER_AREA)
        s1 = cv2.resize(f1, (SRC_W // 2, SRC_H // 2), interpolation=cv2.INTER_AREA)
        fw = dis.calc(s0, s1, None)
        bw = dis.calc(s1, s0, None)
        fw = cv2.resize(fw, (SRC_W, SRC_H)) * 2.0
        bw = cv2.resize(bw, (SRC_W, SRC_H)) * 2.0
        _flow_cache[key] = (fw, bw)
    return _flow_cache[key]


GX, GY = np.meshgrid(np.arange(SRC_W, dtype=np.float32), np.arange(SRC_H, dtype=np.float32))


def interp_frame(src, s, b):
    i = int(math.floor(s))
    t = s - i
    frames = SRC[src]
    if t < 0.03 or i + 1 > b:
        return frames[min(i, b)].astype(np.float32)
    if t > 0.97:
        return frames[i + 1].astype(np.float32)
    fw, bw = flows(src, i)
    f0 = frames[i].astype(np.float32)
    f1 = frames[i + 1].astype(np.float32)
    w0 = cv2.remap(f0, GX - t * fw[..., 0], GY - t * fw[..., 1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REFLECT)
    w1 = cv2.remap(f1, GX - (1 - t) * bw[..., 0], GY - (1 - t) * bw[..., 1], cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REFLECT)
    return (1 - t) * w0 + t * w1


def fetch(seg, s):
    src, a, b = seg['src'], seg['a'], seg['b']
    v = speed_at(seg, s)
    if v > 1.08:
        # shutter blur: average the source frames swept during this output frame
        n = int(math.ceil(v))
        acc = np.zeros((SRC_H, SRC_W, 3), np.float32)
        for k in range(n):
            idx = int(round(min(s + k * v / n, b)))
            acc += SRC[src][idx]
        return acc / n
    return interp_frame(src, s, b)


# ---------------------------------------------------------------- grading
def make_curve():
    x = np.linspace(0, 1, 1024)
    # filmic S-curve: slight toe crush, rolled shoulder
    s = 1 / (1 + np.exp(-(x - 0.40) * 6.0))
    s = (s - s[0]) / (s[-1] - s[0])
    y = 0.6 * s + 0.4 * x
    y = 0.012 + (1 - 0.012) * y          # keep a hint of detail in the blacks
    y = y ** 0.94                        # lift mids a touch
    return x, y


CX, CY = make_curve()
LUT = np.interp(np.linspace(0, 1, 4096), CX, CY).astype(np.float32)


def grade(img, desat=0.0):
    x = np.clip(img / 255.0, 0, 1)
    x = LUT[(x * 4095).astype(np.int32)]
    lum = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lum3 = lum[..., None]
    # split tone: cool/teal shadows, warm highlights (pushes violet energy to pop)
    sh = np.clip(1 - lum3 * 2.2, 0, 1)
    hi = np.clip(lum3 * 1.6 - 0.6, 0, 1)
    x = x + sh * np.array([-0.012, 0.004, 0.03], np.float32) + hi * np.array([0.012, 0.008, -0.01], np.float32)
    sat = 1.04 - desat
    x = lum3 + (x - lum3) * sat
    return np.clip(x, 0, 1)


def bloom(x, strength=0.2):
    small = cv2.resize(x, (x.shape[1] // 4, x.shape[0] // 4), interpolation=cv2.INTER_AREA)
    lum = small @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    mask = np.clip((lum - 0.66) / 0.34, 0, 1)[..., None]
    hl = small * mask
    b1 = cv2.GaussianBlur(hl, (0, 0), 6)
    b2 = cv2.GaussianBlur(hl, (0, 0), 20)
    bl = cv2.resize(0.6 * b1 + 0.6 * b2, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_LINEAR)
    return 1 - (1 - x) * (1 - np.clip(bl * strength * 2, 0, 1))


def vignette_mask(w, h, amt=0.38):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w / 2) / (w / 2)
    ny = (yy - h / 2) / (h / 2)
    r = np.sqrt(nx ** 2 * 0.75 + ny ** 2)
    m = 1 - amt * np.clip(r - 0.35, 0, None) ** 1.6
    return np.clip(m, 0, 1)[..., None].astype(np.float32)


VIG = vignette_mask(SRC_W, SRC_H)

GRAIN = [rng.normal(0, 1, (OUT_H, OUT_W, 1)).astype(np.float32) for _ in range(6)]
for g in GRAIN:
    g[:] = cv2.GaussianBlur(g, (0, 0), 0.6)[..., None] if g.ndim == 3 else g


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def env(local_frame, ev_frame_out, decay):
    d = local_frame - ev_frame_out
    if d < 0 or d > decay * 3:
        return 0.0
    return math.exp(-d / max(decay, 1) * 2.2)


def transform(img, zoom, dx, dy, rot):
    M = cv2.getRotationMatrix2D((SRC_W / 2, SRC_H / 2), rot, zoom)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (SRC_W, SRC_H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def chroma(img, px):
    if px < 0.3:
        return img
    out = img.copy()
    for ch, sgn in ((0, 1), (2, -1)):
        s = 1 + sgn * px / (SRC_W / 2)
        M = cv2.getRotationMatrix2D((SRC_W / 2, SRC_H / 2), 0, s)
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (SRC_W, SRC_H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
    return out


# ---------------------------------------------------------------- encoder
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s', f'{OUT_W}x{OUT_H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-preset', 'ultrafast' if PREVIEW else 'slow',
                        '-crf', '26' if PREVIEW else '15', '-pix_fmt', 'yuv420p',
                        '-tune', 'film', OUTFILE], stdin=subprocess.PIPE)


def emit(x, frame_no):
    """x: float RGB [0,1] at source res -> upscale, grain, write."""
    up = cv2.resize(x, (OUT_W, OUT_H), interpolation=cv2.INTER_LANCZOS4 if not PREVIEW else cv2.INTER_AREA)
    if not PREVIEW:
        # light unsharp to counter upscale softness
        blur = cv2.GaussianBlur(up, (0, 0), 1.2)
        up = up + 0.35 * (up - blur)
    lum = up @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    gw = (0.012 + 0.03 * lum * (1 - lum))[..., None]
    up = up + GRAIN[frame_no % len(GRAIN)][:OUT_H, :OUT_W] * gw
    enc.stdin.write((np.clip(up, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())


# ---------------------------------------------------------------- main edit
tl, total = build_timeline()
frame_no = 0
last_frame = None
for i, start, pos in tl:
    seg = SEGMENTS[i]
    n = len(pos)
    fx = seg.get('fx', [])
    # map event source frames to local output frame indices
    p = np.asarray(pos)
    ev = []
    for e in fx:
        kind, f = e[0], e[1]
        k = int(np.searchsorted(p, f - 1e-6))
        ev.append((kind, k) + tuple(e[2:]))
    z0, z1 = seg.get('zoom', (1.0, 1.0))
    print(f"[{seg['name']}] {n} frames", flush=True)
    for j, s in enumerate(pos):
        img = fetch(seg, s)
        u = j / max(n - 1, 1)
        zoom = z0 + (z1 - z0) * ease(u)
        dx = dy = rot = 0.0
        ca = 0.0
        flash = np.zeros(3, np.float32)
        for e in ev:
            kind, k = e[0], e[1]
            if kind == 'punch':
                zoom += e[2] * env(j, k, e[3])
            elif kind == 'shake':
                a = e[2] * env(j, k, e[3])
                if a > 0:
                    dx += a * rng.uniform(-1, 1)
                    dy += a * rng.uniform(-0.7, 0.7)
                    rot += a * 0.04 * rng.uniform(-1, 1)
            elif kind == 'ca':
                ca += e[2] * env(j, k, e[3])
            elif kind == 'flash':
                flash += e[2] * env(j, k, e[3]) * np.array(e[4], np.float32)
        # zoom margin so shake never reveals edges
        margin = 1 + 2.2 * (abs(dx) + abs(dy)) / SRC_W
        img = transform(img, max(zoom, 1.0) * margin, dx, dy, rot)
        img = chroma(img, ca)
        x = grade(img, seg.get('desat', 0.0))
        x = bloom(x)
        if flash.any():
            x = 1 - (1 - x) * (1 - np.clip(flash, 0, 1))
        fi = seg.get('fade_in', 0)
        if fi and j < fi:
            x = x * ease(j / fi) ** 1.5
        x = x * VIG
        emit(x, frame_no)
        frame_no += 1
        last_frame = x
    # drop cached flows for this segment to keep memory bounded
    _flow_cache.clear()

# ---------------------------------------------------------------- black gap
for k in range(BLACK_GAP):
    emit(np.zeros((SRC_H, SRC_W, 3), np.float32), frame_no)
    frame_no += 1

# ---------------------------------------------------------------- title card
print('[title]', flush=True)
TW, TH = SRC_W, SRC_H
font = ImageFont.truetype(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'SourceSansPro-Black.ttf'), 178)
text = 'UMBRAZOR'
track = 44
# measure with tracking
widths = [font.getbbox(c)[2] - font.getbbox(c)[0] for c in text]
total_w = sum(widths) + track * (len(text) - 1)
asc, desc = font.getmetrics()


def text_layer(scale=1.0):
    im = Image.new('L', (TW, TH), 0)
    d = ImageDraw.Draw(im)
    x = (TW - total_w) / 2
    y = TH / 2 - (asc + desc) / 2 - 8
    for c, w in zip(text, widths):
        d.text((x - font.getbbox(c)[0], y), c, font=font, fill=255)
        x += w + track
    a = np.asarray(im, np.float32) / 255.0
    if scale != 1.0:
        M = cv2.getRotationMatrix2D((TW / 2, TH / 2), 0, scale)
        a = cv2.warpAffine(a, M, (TW, TH), flags=cv2.INTER_CUBIC)
    return a


BASE_TXT = text_layer()
ys, xs = np.nonzero(BASE_TXT > 0.5)
ty0, ty1 = ys.min(), ys.max()
# chrome gradient fill: bright top, steel middle, violet-tinted bottom
grad = np.zeros((TH, TW, 3), np.float32)
for yy in range(TH):
    u = np.clip((yy - ty0) / max(ty1 - ty0, 1), 0, 1)
    if u < 0.48:
        c = np.array([1.0, 1.0, 1.0]) * (1 - 0.35 * u / 0.48)
    elif u < 0.52:
        c = np.array([0.42, 0.40, 0.48])
    else:
        v = (u - 0.52) / 0.48
        c = np.array([0.55, 0.50, 0.62]) + v * np.array([0.25, 0.05, 0.38])
    grad[yy] = c
embers = [dict(x=rng.uniform(0, TW), y=rng.uniform(TH * 0.3, TH * 1.1), vx=rng.uniform(-0.6, 0.6),
               vy=rng.uniform(-3.2, -1.0), r=rng.uniform(0.8, 2.4), hue=rng.uniform(0, 1),
               ph=rng.uniform(0, 6.28)) for _ in range(110)]
cy = (ty0 + ty1) / 2

for k in range(TITLE_FRAMES):
    t = k / FPS
    slam = ease(k / 7.0)
    scale = 1.22 - 0.22 * slam + 0.045 * (k / TITLE_FRAMES)
    a = text_layer(scale)
    # rgb split decays after the slam
    split = 14 * math.exp(-k / 5.0)
    col = np.zeros((TH, TW, 3), np.float32)
    M = cv2.getRotationMatrix2D((TW / 2, TH / 2), 0, scale)
    g = cv2.warpAffine(grad, M, (TW, TH))
    fill = g * a[..., None]
    if split > 0.4:
        shiftR = np.float32([[1, 0, split], [0, 1, 0]])
        shiftB = np.float32([[1, 0, -split], [0, 1, 0]])
        fill[..., 0] = cv2.warpAffine(fill[..., 0], shiftR, (TW, TH))
        fill[..., 2] = cv2.warpAffine(fill[..., 2], shiftB, (TW, TH))
    # violet glow behind letters
    glow = cv2.GaussianBlur(a, (0, 0), 18) * 1.6 + cv2.GaussianBlur(a, (0, 0), 50) * 1.2
    glow_col = glow[..., None] * np.array([0.55, 0.18, 0.95], np.float32)
    # backdrop: dark radial violet haze, breathing
    yy, xx = np.mgrid[0:TH, 0:TW].astype(np.float32)
    rr = np.sqrt(((xx - TW / 2) / (TW * 0.55)) ** 2 + ((yy - cy) / (TH * 0.45)) ** 2)
    haze = np.clip(1 - rr, 0, 1) ** 2 * (0.16 + 0.03 * math.sin(t * 3))
    col += haze[..., None] * np.array([0.35, 0.08, 0.6], np.float32)
    # embers
    layer = np.zeros((TH, TW, 3), np.float32)
    for e in embers:
        ex = e['x'] + e['vx'] * k + 6 * math.sin(e['ph'] + k * 0.15)
        ey = e['y'] + e['vy'] * k
        if 0 <= ex < TW and 0 <= ey < TH:
            c = (1.0, 0.45, 0.15) if e['hue'] < 0.45 else (0.75, 0.35, 1.0)
            flick = 0.6 + 0.4 * math.sin(e['ph'] + k * 0.7)
            cv2.circle(layer, (int(ex), int(ey)), int(math.ceil(e['r'])),
                       tuple(float(v * flick) for v in c), -1, lineType=cv2.LINE_AA)
    layer = layer + cv2.GaussianBlur(layer, (0, 0), 4) * 1.5
    col += layer * min(1.0, k / 6.0)
    col = 1 - (1 - col) * (1 - np.clip(glow_col, 0, 1))
    col = col * (1 - a[..., None]) + fill
    # anamorphic light streak through the title on the slam
    streak_a = math.exp(-k / 9.0)
    sy = np.exp(-((yy - cy) / 4.0) ** 2) * np.exp(-((xx - TW / 2) / (TW * 0.42)) ** 2)
    col += (sy * streak_a * 1.4)[..., None] * np.array([0.8, 0.6, 1.0], np.float32)
    # slam flash
    fl = 0.9 * math.exp(-k / 2.5)
    col = 1 - (1 - col) * (1 - fl * np.array([0.9, 0.8, 1.0], np.float32))
    # camera shake on slam
    sh = 10 * math.exp(-k / 6.0)
    if sh > 0.3:
        col = transform(col, 1 + 2.2 * sh / TW * 2, rng.uniform(-sh, sh), rng.uniform(-sh, sh) * 0.6, 0)
    # fade out
    fo = 20
    if k > TITLE_FRAMES - fo:
        col = col * ease((TITLE_FRAMES - k) / fo)
    emit(np.clip(col, 0, 1) * VIG, frame_no)
    frame_no += 1

enc.stdin.close()
enc.wait()
print('done frames', frame_no, 'seconds', frame_no / FPS)
