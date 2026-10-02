"""Instruments (music) and sound effects, all synthesized. Each returns mono float arrays
unless noted (stereo arrays have shape (n,2))."""
import numpy as np
from synth import *

R = rng()


# ================================================================ DRUMS / PERC
def kick(drive=2.2, tail=0.2):
    n = N(0.6)
    t = tt(n)
    f = 44 + 150 * np.exp(-t / 0.028) + 40 * np.exp(-t / 0.004)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / tail)
    click = hp(white(n), 2500) * np.exp(-t / 0.0025) * 0.35
    x = sat(body + click, drive)
    return fade(x, 0.0005, 0.02)


def taiko(pitch=1.0, size=1.0):
    n = N(1.4)
    t = tt(n)
    f = (64 + 95 * np.exp(-t / 0.03)) * pitch
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (0.32 * size))
    skin = bp(white(n), 160 * pitch, 1200 * pitch) * np.exp(-t / 0.05) * 1.3
    slap = hp(white(n), 1800) * np.exp(-t / 0.006) * 0.25
    x = sat(body * 1.0 + skin + slap, 1.6)
    return fade(x, 0.0005, 0.05)


def snare_big():
    n = N(0.9)
    t = tt(n)
    nz = bp(white(n), 900, 9000) * np.exp(-t / 0.15)
    tone = (np.sin(2 * np.pi * 188 * t) + 0.6 * np.sin(2 * np.pi * 330 * t)) * np.exp(-t / 0.07)
    crack = hp(white(n), 3500) * np.exp(-t / 0.012) * 0.9
    clap = np.zeros(n)
    for d in (0.0, 0.011, 0.023):
        k = N(d)
        seg = bp(white(n - k), 800, 4000) * np.exp(-tt(n - k) / 0.012)
        clap[k:] += seg * 0.5
    x = sat(nz * 0.9 + tone * 0.6 + crack + clap, 1.8)
    return fade(x, 0.0005, 0.05)


def snare_roll_hit(vel=1.0):
    n = N(0.25)
    t = tt(n)
    x = bp(white(n), 1200, 8000) * np.exp(-t / 0.06) + np.sin(2 * np.pi * 200 * t) * np.exp(-t / 0.03) * 0.5
    return fade(x * vel, 0.0005, 0.02)


def hat(open_=False, vel=1.0):
    n = N(0.5 if open_ else 0.12)
    t = tt(n)
    x = np.zeros(n)
    for fr in (205.3, 304.4, 369.6, 522.7, 540.0, 800.0):
        x += square(fr * 1.6, n)
    x = bp(x, 7000, 14000, 2) + 0.4 * hp(white(n), 9000)
    x *= np.exp(-t / (0.22 if open_ else 0.028))
    return fade(x * vel * 0.5, 0.0003, 0.01)


def crash(dur=3.0, bright=1.0):
    n = N(dur)
    t = tt(n)
    x = hp(white(n), 3500) * 0.6
    for _ in range(40):
        fr = R.uniform(3000, 12000)
        x += np.sin(2 * np.pi * fr * t + R.uniform(0, 6.28)) * R.uniform(0.02, 0.08) * np.exp(-t / R.uniform(0.3, 1.5))
    x = tv_filter(x, lambda f, tm: lp_gain(f, 4000 + 9000 * bright * np.exp(-tm / 0.6), 1))
    x *= np.exp(-t / (dur * 0.35)) * (1 - np.exp(-t / 0.001))
    return fade(x, 0.0005, 0.2)


def tom(pitch=1.0):
    n = N(0.8)
    t = tt(n)
    f = (90 + 70 * np.exp(-t / 0.04)) * pitch
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.25)
    x += bp(white(n), 200, 2000) * np.exp(-t / 0.03) * 0.4
    return fade(sat(x, 1.5), 0.0005, 0.05)


def tick(vel=1.0):
    n = N(0.03)
    t = tt(n)
    x = (np.sin(2 * np.pi * 3200 * t) + 0.5 * np.sin(2 * np.pi * 5100 * t)) * np.exp(-t / 0.004)
    return fade(x * vel, 0.0002, 0.005)


# ================================================================ TONAL
def bass_note(f, dur, accent=1.0, drive=2.6):
    n = N(dur + 0.03)
    t = tt(n)
    x = 0.6 * saw(f, n) + 0.6 * saw(f * cents(9), n) + 0.4 * sine(f, n) + 0.2 * square(f * 0.5, n)
    lo = lp(x, 180, 2)
    hi = lp(x, 1400 + 900 * accent, 2)
    e = np.exp(-t / (0.06 + 0.03 * accent)) * min(accent, 1.2)
    y = (1 - e) * lo + e * hi
    y = sat(y * 1.2, drive)
    mid = bp(sat(x * 3.0, 4.0), 350, 1600) * (0.35 + 0.4 * e)
    y = y + mid
    amp = np.ones(n)
    a = N(0.003)
    amp[:a] = np.linspace(0, 1, a)
    r = N(0.025)
    k = N(dur)
    amp[k:] = 0
    amp[max(0, k - r):k] *= np.linspace(1, 0, min(r, k))
    return y * amp


def spiccato(f, vel=1.0, dur=0.22):
    n = N(dur)
    t = tt(n)
    x = sum(saw(f * cents(c), n) for c in (-7, -2, 3, 8)) / 4
    x = lp(x, 2600 + 1500 * vel, 2)
    x += hp(white(n), 2500) * 0.04 * np.exp(-t / 0.02)
    x *= np.exp(-t / 0.075) * (1 - np.exp(-t / 0.0025))
    return fade(x * vel, 0.0005, 0.02)


def braam(freqs, dur=3.0, bright=1.0, drive=3.0, attack=0.025, growl=0.25):
    n = N(dur)
    t = tt(n)
    bend = cents(-45 * np.exp(-t / 0.07))
    x = np.zeros(n)
    for f in freqs:
        for c in (-15, -9, -4, 0, 4, 9, 15):
            x += saw(f * cents(c) * bend, n)
        if f > 50:
            x += 1.0 * sine(f * 0.5 * bend, n)
    x /= len(freqs) * 7
    am = 1 + growl * np.sin(2 * np.pi * 31 * t) * np.exp(-t / 0.8)
    x *= am
    x = tv_filter(x, lambda fr, tm: lp_gain(fr, 260 + bright * 2800 * np.exp(-tm / 0.45) + 600 * bright * np.exp(-tm / 2.5), 2))
    x = sat(x * 2.2, drive)
    x = lp(x, 5500, 2)
    amp = (1 - np.exp(-t / attack)) * np.exp(-t / (dur * 0.42))
    return fade(x * amp, 0.001, 0.3)


def pad(freqs, dur, attack=0.8, release=1.5, cutoff=900, voices=5, vib=0.003):
    n = N(dur)
    t = tt(n)
    x = np.zeros(n)
    for f in freqs:
        for k in range(voices):
            c = (k - (voices - 1) / 2) * 6
            fv = f * cents(c) * (1 + vib * np.sin(2 * np.pi * R.uniform(0.15, 0.4) * t + R.uniform(0, 6)))
            x += saw(fv, n)
    x /= len(freqs) * voices
    x = lp(x, cutoff, 2)
    amp = np.clip(t / attack, 0, 1) ** 1.5
    rs = N(max(dur - release, 0))
    amp[rs:] *= np.linspace(1, 0, n - rs) ** 1.5
    return x * amp


def choir(freqs, dur, attack=0.6, release=1.6, vowel='ah'):
    forms = {'ah': [(730, 90, 1.0), (1090, 110, 0.55), (2440, 160, 0.25), (3400, 250, 0.08)],
             'oh': [(570, 80, 1.0), (840, 100, 0.5), (2410, 160, 0.2), (3300, 250, 0.06)]}[vowel]
    n = N(dur)
    t = tt(n)
    src = np.zeros(n)
    for f in freqs:
        for k in range(4):
            vibr = 1 + 0.006 * np.sin(2 * np.pi * (5.0 + 0.4 * k) * t + R.uniform(0, 6)) + 0.002 * smooth_noise(n, 3)
            src += saw(f * cents((k - 1.5) * 7) * vibr, n)
    src /= len(freqs) * 4
    y = np.zeros(n)
    for fc, bw, g in forms:
        y += bp(src, fc - bw, fc + bw, 2) * g
    y += hp(white(n), 5000) * 0.004  # breath
    amp = np.clip(t / attack, 0, 1) ** 1.2
    rs = N(max(dur - release, 0))
    amp[rs:] *= np.linspace(1, 0, n - rs) ** 1.4
    return norm(y * amp, 0.8)


def drone(f, dur, cutoff0=180, cutoff1=600):
    n = N(dur)
    t = tt(n)
    x = 0.6 * saw(f, n) + 0.6 * saw(f * cents(7), n) + 0.6 * saw(f * 2 * cents(-5), n) + sine(f, n) * 0.5
    x = tv_filter(x, lambda fr, tm: lp_gain(fr, cutoff0 + (cutoff1 - cutoff0) * np.clip(tm / dur, 0, 1), 2))
    x *= 1 + 0.15 * np.sin(2 * np.pi * 0.23 * t)
    return x


# ================================================================ IMPACTS / FX
def impact(dur=3.0, f_hi=120, f_lo=30, crack=0.6, body=0.7, tau=0.75):
    n = N(dur)
    t = tt(n)
    f_lo = max(f_lo, 36)
    f = f_lo + (f_hi - f_lo) * np.exp(-t / 0.08)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / tau)
    bod = lp(brown(n), 280) * np.exp(-t / 0.22) * body
    thud = bp(white(n), 140, 900) * np.exp(-t / 0.07) * (0.5 + 0.5 * body)
    crk = bp(white(n), 1400, 8000) * np.exp(-t / 0.022) * crack
    x = sat(sub * 0.75 + bod * 0.6 + thud * 1.1 + crk * 1.2, 1.7)
    x *= 1 - np.exp(-t / 0.0007)
    return fade(x, 0.0003, 0.3)


def sub_drop(dur=2.5, f0=70, f1=24):
    n = N(dur)
    t = tt(n)
    f = f1 + (f0 - f1) * np.exp(-t / (dur * 0.35))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (dur * 0.4)) * (1 - np.exp(-t / 0.004))
    return fade(sat(x, 1.3), 0.001, 0.3)


def explosion(dur=4.0, size=1.0):
    n = N(dur)
    t = tt(n)
    x = impact(dur, 100 * (1.2 - 0.3 * size), 26, crack=0.9, body=1.0, tau=0.9 * size)
    turb = 0.6 + 0.4 * smooth_noise(n, 14)
    rumble = lp(brown(n), 160 + 80 * size) * np.exp(-t / (1.1 * size)) * turb * (1 - np.exp(-t / 0.01))
    roar = bp(white(n), 250, 1600) * np.exp(-t / (0.45 * size)) * turb * (1 - np.exp(-t / 0.006))
    deb = crackle(dur, density=90, decay=1.2 * size) * 0.35
    y = x * 0.9 + rumble * 0.9 + roar * 0.5 + deb
    return fade(sat(y, 1.4), 0.0005, 0.4)


def crackle(dur, density=60, decay=1.0, lo=1800, hi=9000):
    n = N(dur)
    t = tt(n)
    imp = np.zeros(n)
    dens = density * np.exp(-t / decay)
    p = dens / SR
    hits = R.uniform(size=n) < p
    imp[hits] = R.uniform(-1, 1, hits.sum()) * R.uniform(0.2, 1, hits.sum())
    # give each impulse a little ring
    k = np.exp(-tt(N(0.004)) / 0.0012)
    x = signal.fftconvolve(imp, k)[:n]
    return bp(x, lo, hi, 2) * 3


def whoosh(dur=0.6, f_lo=250, f_hi=2500, peak=0.55, pan_from=-0.6, pan_to=0.6, rough=0.3):
    n = N(dur)
    t = tt(n)
    u = t / dur
    shape = np.where(u < peak, (u / peak) ** 2, np.exp(-((u - peak) / (1 - peak)) * 3.5))
    fc = f_lo * (f_hi / f_lo) ** shape
    x = white(n)
    x = tv_filter(x, lambda fr, tm: band_gain(fr, np.interp(tm, t, fc), 0.55), nperseg=512)
    x *= shape * (1 + rough * smooth_noise(n, 30))
    x = norm(x, 1.0)
    return pan(fade(x, 0.002, 0.02), np.linspace(pan_from, pan_to, n))


def swish(dur=0.28, hi=7000, ring=True):
    n = N(dur + 0.5)
    t = tt(n)
    u = np.clip(t / dur, 0, 1)
    shape = np.where(t < dur, np.sin(np.pi * u) ** 3, 0)
    fc = 900 * (hi / 900) ** u
    x = tv_filter(white(n), lambda fr, tm: band_gain(fr, np.interp(tm, t, fc), 0.4), nperseg=256) * shape
    x = norm(x, 1.0)
    if ring:
        rr = np.zeros(n)
        k = N(dur * 0.55)
        tr = tt(n - k)
        for fr, a in ((3520, 0.18), (5274, 0.12), (7458, 0.06)):
            rr[k:] += np.sin(2 * np.pi * fr * tr) * a * np.exp(-tr / 0.35)
        x += rr
    return fade(x, 0.001, 0.05)


def zap(dur=0.5, f0=2200, f1=140):
    n = N(dur)
    t = tt(n)
    fc = f1 + (f0 - f1) * np.exp(-t / (dur * 0.18))
    mod = np.sin(2 * np.pi * np.cumsum(fc * 1.41) / SR) * 3.5
    x = np.sin(2 * np.pi * np.cumsum(fc) / SR + mod) * np.exp(-t / (dur * 0.35))
    x += crackle(dur, 140, dur * 0.4) * 0.25
    return fade(sat(x, 1.6), 0.001, 0.05)


def energy_slash(dur=0.55):
    w = whoosh(dur, 500, 6000, 0.35, -0.5, 0.5, 0.2)
    z = zap(dur, 2600, 180)
    out = w * 0.8
    out[:, 0] += z * 0.45
    out[:, 1] += z * 0.45
    return out


def clang(base=420, dur=1.6, bright=1.0):
    n = N(dur)
    t = tt(n)
    x = np.zeros(n)
    for r, a, d in ((1.0, 1.0, 1.1), (2.32, 0.7, 0.75), (4.25, 0.5, 0.45), (6.63, 0.38, 0.3),
                    (9.38, 0.25 * bright, 0.2), (13.1, 0.15 * bright, 0.12)):
        fr = base * r
        x += a * (np.sin(2 * np.pi * fr * t) + 0.5 * np.sin(2 * np.pi * fr * 1.004 * t)) * np.exp(-t / d)
    x += hp(white(n), 2000) * np.exp(-t / 0.008) * 1.5
    x *= 1 - np.exp(-t / 0.0004)
    return fade(sat(x * 0.6, 1.4), 0.0002, 0.2)


def shing(dur=2.2, base=2637):
    n = N(dur)
    t = tt(n)
    scr_d = 0.16
    u = np.clip(t / scr_d, 0, 1)
    scrape = tv_filter(white(n), lambda fr, tm: band_gain(fr, 3000 + 5000 * np.clip(tm / scr_d, 0, 1), 0.35), 256)
    scrape *= np.where(t < scr_d, u ** 2, np.exp(-(t - scr_d) / 0.03)) * 0.5
    ring = np.zeros(n)
    tr = np.clip(t - scr_d * 0.8, 0, None)
    on = t > scr_d * 0.8
    for r, a in ((1.0, 0.5), (1.5, 0.3), (2.0, 0.22), (2.67, 0.12)):
        fr = base * r
        ring += a * (np.sin(2 * np.pi * fr * tr) + np.sin(2 * np.pi * fr * 1.0025 * tr)) * np.exp(-tr / (0.9 / r ** 0.5))
    ring *= on
    return fade(scrape + ring * 0.6, 0.001, 0.2)


def creak(dur=0.5, grains=45):
    n = N(dur)
    x = np.zeros(n)
    for _ in range(grains):
        k = int(R.uniform(0, n - N(0.02)))
        g = N(R.uniform(0.004, 0.014))
        fc = R.uniform(500, 1800)
        gr = bp(white(g + 200), fc * 0.8, fc * 1.25)[200:] * np.hanning(g) * R.uniform(0.3, 1)
        x[k:k + g] += gr
    return x


def roar(dur=1.6, f0=82, bright=1.0):
    n = N(dur)
    t = tt(n)
    u = t / dur
    contour = f0 * (0.85 + 0.45 * np.sin(np.pi * np.clip(u * 1.2, 0, 1)) ** 0.7) * (1 + 0.03 * smooth_noise(n, 18))
    src = saw(contour, n) + 0.6 * square(contour * 0.5, n) + 0.5 * white(n) * (0.5 + 0.5 * smooth_noise(n, 25))
    src *= 1 + 0.45 * np.sin(2 * np.pi * 27 * t)

    def formant(fr, tm):
        v = np.clip(np.sin(np.pi * np.clip(tm / dur * 1.3, 0, 1)), 0, 1)
        F = [(320 + 420 * v, 120), (870 + 250 * v, 160), (2300 + 150 * v, 300), (3300, 400)]
        g = 0.04
        for k, (fc, bw) in enumerate(F):
            g = g + (0.9 ** k) * np.exp(-0.5 * ((fr - fc) / bw) ** 2)
        return g
    y = tv_filter(src, formant, 1024)
    y = sat(y * 3.0, 3.0)
    low = tv_filter(src, lambda fr, tm: lp_gain(fr, 300, 2), 1024)
    y = y + 0.8 * sat(low * 2, 2)
    amp = np.clip(t / 0.12, 0, 1) * np.where(u > 0.7, np.exp(-(u - 0.7) / 0.12), 1.0)
    return norm(fade(y * amp, 0.005, 0.1), 0.9)


def heartbeat():
    n = N(0.7)
    t = tt(n)
    x = np.zeros(n)
    for d, a in ((0.0, 1.0), (0.21, 0.7)):
        k = N(d)
        tr = tt(n - k)
        f = 42 + 38 * np.exp(-tr / 0.03)
        x[k:] += a * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tr / 0.085) * (1 - np.exp(-tr / 0.003))
    return lp(x, 180, 2)


def tinnitus(dur, f=5200):
    n = N(dur)
    t = tt(n)
    return np.sin(2 * np.pi * f * t) * np.sin(np.pi * t / dur) ** 2


def portal_hum(dur, root=55):
    n = N(dur)
    t = tt(n)
    trem = 0.7 + 0.3 * np.sin(2 * np.pi * 7 * t)
    x = (np.sin(2 * np.pi * root * t) + 0.5 * np.sin(2 * np.pi * root * 2 * t) + 0.25 * sat(saw(root * 3, n), 1.2)) * trem
    swirl = tv_filter(white(n), lambda fr, tm: band_gain(fr, 700 * 2 ** (0.9 * np.sin(2 * np.pi * 0.9 * tm)), 0.5))
    x = x * 0.6 + swirl * 0.5
    a = np.clip(t / 0.08, 0, 1) * np.clip((dur - t) / 0.2, 0, 1)
    return x * a


def powerup(dur=0.7, f0=220, f1=2600):
    n = N(dur)
    t = tt(n)
    u = t / dur
    f = f0 * (f1 / f0) ** (u ** 1.4)
    vib = 1 + 0.02 * np.sin(2 * np.pi * (6 + 30 * u) * t)
    x = np.sin(2 * np.pi * np.cumsum(f * vib) / SR) + 0.35 * lp(saw(f * vib * 0.5, n), 3000)
    x += crackle(dur, 30, 10, 2500, 9000) * u * 0.3
    return fade(x * (0.2 + 0.8 * u ** 1.5), 0.005, 0.01)


def mech_click():
    n = N(0.25)
    t = tt(n)
    x = np.zeros(n)
    for d, fr in ((0.0, 2400), (0.045, 1700)):
        k = N(d)
        tr = tt(n - k)
        x[k:] += (hp(white(n - k), 1500) * np.exp(-tr / 0.004) + np.sin(2 * np.pi * fr * tr) * np.exp(-tr / 0.02) * 0.6)
    return x


def shimmer(dur=1.5, density=40, rise=True):
    n = N(dur)
    x = np.zeros(n)
    count = int(density * dur)
    for _ in range(count):
        u = R.uniform() ** (0.6 if rise else 1.6)
        k = int(u * (n - N(0.4)))
        fr = R.uniform(2200, 9000)
        d = R.uniform(0.04, 0.3)
        m = min(N(d * 3), n - k)
        tr = tt(m)
        x[k:k + m] += np.sin(2 * np.pi * fr * tr) * np.exp(-tr / d) * R.uniform(0.2, 1)
    return x / max(1, np.sqrt(count) * 0.4)


def flame_roar(dur=1.2):
    n = N(dur)
    t = tt(n)
    turb = 0.5 + 0.5 * np.abs(smooth_noise(n, 16))
    x = lp(brown(n), 700) * turb + bp(white(n), 900, 3000) * turb * 0.25
    a = np.clip(t / 0.06, 0, 1) * np.exp(-t / (dur * 0.45))
    return x * a


def cloth_flap(dur=0.6):
    n = N(dur)
    t = tt(n)
    flutter = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 22 * t + 3 * smooth_noise(n, 8)))
    x = bp(white(n), 250, 1600) * flutter
    a = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5
    return x * a


def splash(dur=0.5):
    n = N(dur)
    t = tt(n)
    x = bp(white(n), 900, 6500) * np.exp(-t / 0.09) * (1 - np.exp(-t / 0.004))
    x += crackle(dur, 120, 0.15, 1500, 7000) * 0.4
    return x


def footstep_heavy():
    n = N(0.6)
    t = tt(n)
    f = 38 + 60 * np.exp(-t / 0.02)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.12) * (1 - np.exp(-t / 0.002))
    x += splash(0.6) * 0.35
    x += bp(white(n), 300, 1200) * np.exp(-t / 0.03) * 0.3
    return x


def debris(dur=1.2, density=35):
    n = N(dur)
    x = np.zeros(n)
    t = tt(n)
    cnt = int(density * dur)
    for _ in range(cnt):
        k = int(R.exponential(0.25) * SR)
        if k >= n - N(0.1):
            continue
        m = N(0.08)
        fc = R.uniform(500, 4000)
        x[k:k + m] += bp(white(m), fc * 0.7, fc * 1.4) * np.exp(-tt(m) / 0.015) * R.uniform(0.2, 1)
    return x


def riser(dur=4.0, f0=200, f1=9000):
    n = N(dur)
    t = tt(n)
    u = t / dur
    fc = f0 * (f1 / f0) ** u
    nz = tv_filter(white(n), lambda fr, tm: band_gain(fr, np.interp(tm, t, fc), 0.7), 1024)
    nz = norm(nz, 1.0)
    pitch = 110 * 2 ** (3 * u ** 1.3)
    tone = sum(lp(saw(pitch * m, n), 5000) * g for m, g in ((1, 0.5), (1.5, 0.25), (2, 0.25)))
    trem_rate = 3 + 22 * u ** 1.5
    trem = 1 - 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * np.cumsum(trem_rate) / SR))
    x = (nz * 0.8 + tone * 0.25) * trem * u ** 2.2
    return fade(x, 0.01, 0.003)


def reverse_swell(dur=1.5):
    c = crash(dur + 0.3, 1.2)[:N(dur)]
    x = c[::-1].copy()
    x += riser(dur, 600, 12000) * 0.4
    return fade(x, 0.02, 0.004)


def rain(dur):
    n = N(dur)
    out = np.zeros((n, 2))
    for ch in range(2):
        x = bp(pink(n), 400, 7500) * 0.25
        x *= 0.85 + 0.15 * smooth_noise(n, 0.5)
        drops = np.zeros(n)
        hits = R.uniform(size=n) < 70 / SR
        drops[hits] = R.uniform(0.1, 1, hits.sum())
        drops = signal.fftconvolve(drops, np.exp(-tt(N(0.01)) / 0.002) * np.sin(2 * np.pi * R.uniform(2500, 4500) * tt(N(0.01))))[:n]
        x += bp(drops, 1500, 7000) * 0.15
        x += lp(brown(n), 110) * 0.15
        out[:, ch] = x
    return out


def st(fn, *a, **k):
    """Render a mono generator twice (independent random phases) -> wide stereo."""
    return np.stack([fn(*a, **k), fn(*a, **k)], axis=1)


def stab(freqs, dur=0.16, bright=1.0, drive=2.2):
    n = N(dur + 0.3)
    t = tt(n)
    x = np.zeros(n)
    for f in freqs:
        for c in (-9, -3, 3, 9):
            x += saw(f * cents(c), n)
        x += 0.4 * square(f, n, 0.3)
    x /= len(freqs) * 4
    lo = lp(x, 450, 2)
    hi = lp(x, 2500 + 2500 * bright, 2)
    e = np.exp(-t / 0.07)
    y = (1 - e) * lo + e * hi
    amp = (1 - np.exp(-t / 0.002)) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.07))
    return fade(sat(y * amp * 1.6, drive), 0.001, 0.02)


def eerie(dur):
    n = N(dur)
    t = tt(n)
    x = np.zeros(n)
    for f, a in ((587.3, 0.5), (622.3, 0.35), (880.0, 0.25), (1174.7, 0.15)):
        trem = 0.6 + 0.4 * np.sin(2 * np.pi * (5.5 + R.uniform(-1, 1)) * t + R.uniform(0, 6))
        x += a * (0.7 * np.sin(2 * np.pi * f * t) + 0.3 * lp(saw(f * cents(R.uniform(-6, 6)), n), 2500)) * trem
    x += bp(white(n), 1100, 1500) * 0.12 * (0.5 + 0.5 * smooth_noise(n, 2))
    a = np.clip(t / (dur * 0.5), 0, 1) ** 2 * np.clip((dur - t) / 0.4, 0, 1)
    return x * a
