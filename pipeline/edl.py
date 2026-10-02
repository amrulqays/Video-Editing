"""Edit decision list + retiming for the UMBRAZOR fight edit.

Frames are source frame indices at 24 fps (inclusive ranges).
speed: list of (src_frame, speed) keypoints, linearly interpolated.
zoom: (start, end) slow push over the shot.
fx: visual events keyed by source frame:
    ('flash', frame, strength, decay_frames, color)
    ('shake', frame, amp_px, decay_frames)
    ('punch', frame, zoom_amt, decay_frames)
    ('ca', frame, px, decay_frames)
"""
import numpy as np

FPS = 24
WHITE = (1.0, 1.0, 1.0)
VIOLET = (0.85, 0.55, 1.0)

SEGMENTS = [
    # ---------------- INTRO: the hunter in the alley ----------------
    dict(name='walk',      src='B', a=10,  b=36,  speed=1.0, zoom=(1.00, 1.06), fade_in=14),
    dict(name='face',      src='B', a=37,  b=43,  speed=0.85, zoom=(1.06, 1.10)),
    dict(name='grip',      src='B', a=44,  b=56,  speed=0.8, zoom=(1.00, 1.07),
         fx=[('punch', 50, 0.025, 8)]),
    dict(name='monster',   src='B', a=57,  b=92,  speed=1.15, zoom=(1.00, 1.06),
         fx=[('flash', 57, 0.35, 6, (1.0, 0.4, 0.4)), ('shake', 57, 6, 10),
             ('shake', 70, 4, 6), ('shake', 78, 7, 8), ('shake', 86, 5, 6)]),
    dict(name='crouch',    src='B', a=93,  b=115, speed=1.25, zoom=(1.00, 1.05)),
    dict(name='lunge',     src='A', a=100, b=110, speed=1.0, zoom=(1.02, 1.08),
         fx=[('shake', 100, 6, 8)]),
    dict(name='eyes',      src='A', a=111, b=120, speed=0.5, zoom=(1.00, 1.14),
         desat=0.35),
    dict(name='summon',    src='B', a=116, b=129, speed=1.0, zoom=(1.03, 1.0),
         fx=[('flash', 116, 0.9, 7, VIOLET), ('shake', 116, 12, 12),
             ('ca', 116, 10, 10), ('punch', 116, 0.06, 10)]),
    dict(name='whip',      src='B', a=130, b=138, speed=1.0, zoom=(1.0, 1.0),
         fx=[('ca', 130, 8, 9)]),
    # ---------------- HENSHIN ----------------
    dict(name='belt',      src='B', a=139, b=160, speed=1.0, zoom=(1.00, 1.08),
         fx=[('flash', 143, 0.55, 8, VIOLET), ('punch', 143, 0.05, 10), ('ca', 143, 6, 8)]),
    dict(name='portalA',   src='A', a=147, b=163, speed=1.0, zoom=(1.04, 1.00),
         fx=[('flash', 147, 0.6, 6, VIOLET), ('shake', 147, 8, 10)]),
    dict(name='run',       src='B', a=161, b=199, speed=[(161, 1.6), (199, 1.25)], zoom=(1.00, 1.05)),
    dict(name='dive',      src='B', a=200, b=232, speed=[(200, 1.0), (205, 0.5), (226, 0.62), (232, 1.0)],
         zoom=(1.00, 1.10), fx=[('flash', 203, 0.45, 8, VIOLET), ('ca', 205, 6, 14)]),
    dict(name='emerge',    src='A', a=196, b=224, speed=[(196, 1.0), (208, 0.75), (224, 1.0)],
         zoom=(1.00, 1.06), fx=[('flash', 196, 0.4, 6, VIOLET)]),
    dict(name='helmet',    src='B', a=236, b=247, speed=0.5, zoom=(1.00, 1.12),
         fx=[('flash', 241, 0.25, 10, VIOLET)]),
    # ---------------- FIGHT (drop) ----------------
    dict(name='leap',      src='B', a=248, b=279, speed=1.3, zoom=(1.10, 1.02), drop=True,
         fx=[('flash', 248, 1.0, 8, WHITE), ('shake', 248, 16, 14), ('ca', 248, 12, 10),
             ('punch', 248, 0.08, 12)]),
    dict(name='slash',     src='B', a=280, b=295, speed=[(280, 1.0), (286, 0.6), (295, 0.8)],
         zoom=(1.02, 1.08),
         fx=[('flash', 289, 0.5, 6, VIOLET), ('shake', 289, 10, 10), ('ca', 289, 8, 8)]),
    dict(name='clash',     src='B', a=296, b=323, speed=1.0, zoom=(1.08, 1.00),
         fx=[('flash', 296, 0.85, 7, WHITE), ('shake', 296, 18, 14), ('punch', 296, 0.07, 10),
             ('ca', 296, 12, 10), ('shake', 305, 8, 8), ('flash', 322, 0.5, 6, VIOLET),
             ('shake', 322, 12, 10), ('ca', 322, 8, 8)]),
    dict(name='blur',      src='B', a=324, b=328, speed=1.0, zoom=(1.0, 1.0),
         fx=[('ca', 324, 10, 5)]),
    dict(name='cape',      src='A', a=228, b=249, speed=1.1, zoom=(1.0, 1.06),
         fx=[('shake', 236, 6, 10), ('punch', 245, 0.03, 8)]),
    dict(name='wave',      src='A', a=250, b=266, speed=1.0, zoom=(1.0, 1.05),
         fx=[('shake', 250, 10, 12), ('ca', 250, 8, 10)]),
    dict(name='boom',      src='A', a=267, b=275, speed=0.45, zoom=(1.04, 1.12),
         fx=[('flash', 267, 1.0, 10, WHITE), ('shake', 267, 22, 18), ('ca', 267, 14, 14),
             ('punch', 267, 0.08, 14)]),
    dict(name='land',      src='B', a=329, b=356,
         speed=[(329, 1.1), (336, 1.0), (337, 0.45), (343, 0.6), (346, 1.0), (356, 1.0)],
         zoom=(1.0, 1.06),
         fx=[('shake', 337, 20, 16), ('punch', 337, 0.05, 12), ('flash', 337, 0.35, 8, VIOLET)]),
    # ---------------- FINALE ----------------
    dict(name='pose',      src='A', a=282, b=335, speed=[(282, 1.0), (292, 0.8), (300, 1.0)],
         zoom=(1.00, 1.10),
         fx=[('flash', 293, 0.6, 10, VIOLET), ('shake', 293, 14, 18), ('ca', 293, 8, 10),
             ('shake', 297, 8, 14)]),
    dict(name='wings',     src='B', a=390, b=432, speed=0.9, zoom=(1.00, 1.08),
         fx=[('shake', 395, 6, 12)], fade_out=0),
]

TITLE_FRAMES = 84


def speed_at(seg, f):
    sp = seg['speed']
    if np.isscalar(sp):
        return float(sp)
    xs = [k for k, _ in sp]
    ys = [v for _, v in sp]
    return float(np.interp(f, xs, ys))


def retime(seg):
    """Return list of float source frame positions for each output frame."""
    a, b = seg['a'], seg['b']
    out = []
    s = float(a)
    while s <= b + 0.5 - 1e-6:   # include the last frame's duration
        out.append(min(s, b))
        s += speed_at(seg, s)
    return out


def build_timeline():
    """Return list of (seg_index, out_start_frame, positions)."""
    tl = []
    t = 0
    for i, seg in enumerate(SEGMENTS):
        pos = retime(seg)
        tl.append((i, t, pos))
        t += len(pos)
    return tl, t


def src_to_out(tl, seg_index, src_frame):
    """Map a source frame inside a segment to output time in seconds."""
    i, start, pos = tl[seg_index]
    pos = np.asarray(pos)
    k = np.searchsorted(pos, src_frame - 1e-6)
    if k == 0:
        return start / FPS
    if k >= len(pos):
        return (start + len(pos)) / FPS
    # interpolate within output frames
    frac = (src_frame - pos[k - 1]) / max(pos[k] - pos[k - 1], 1e-6)
    return (start + k - 1 + frac) / FPS


def seg_index(name):
    for i, s in enumerate(SEGMENTS):
        if s['name'] == name:
            return i
    raise KeyError(name)


if __name__ == '__main__':
    tl, total = build_timeline()
    for i, start, pos in tl:
        s = SEGMENTS[i]
        print(f"{s['name']:8s} {s['src']} {s['a']:3d}-{s['b']:3d} out {start/FPS:6.3f}s "
              f"len {len(pos):3d}f ({len(pos)/FPS:.2f}s)")
    print('video total', total, 'frames', total / FPS, 's  + title', TITLE_FRAMES / FPS)
    print('grand total', (total + TITLE_FRAMES) / FPS)
