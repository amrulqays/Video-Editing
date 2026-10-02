"""Small numpy synthesis toolkit for score + sound design."""
import numpy as np
from scipy import signal

SR = 48000
_rng = np.random.default_rng(1234)


def rng():
    return _rng


def N(dur):
    return max(1, int(round(dur * SR)))


def tt(n):
    return np.arange(n) / SR


# ------------------------------------------------------------------ oscillators
def _as_arr(f, n):
    return np.full(n, float(f)) if np.isscalar(f) else np.asarray(f, float)


def phase_of(freq, n, ph0=0.0):
    f = _as_arr(freq, n)
    return (ph0 + np.cumsum(f) / SR) % 1.0, f / SR


def _blep(ph, dt):
    y = np.zeros_like(ph)
    m1 = ph < dt
    t1 = ph[m1] / dt[m1]
    y[m1] = t1 + t1 - t1 * t1 - 1
    m2 = ph > 1 - dt
    t2 = (ph[m2] - 1) / dt[m2]
    y[m2] = t2 * t2 + t2 + t2 + 1
    return y


def saw(freq, n, ph0=None):
    ph0 = _rng.uniform() if ph0 is None else ph0
    ph, dt = phase_of(freq, n, ph0)
    return 2 * ph - 1 - _blep(ph, dt)


def square(freq, n, pw=0.5, ph0=None):
    ph0 = _rng.uniform() if ph0 is None else ph0
    ph, dt = phase_of(freq, n, ph0)
    a = 2 * ph - 1 - _blep(ph, dt)
    ph2 = (ph + pw) % 1.0
    b = 2 * ph2 - 1 - _blep(ph2, dt)
    return 0.5 * (a - b)


def sine(freq, n, ph0=None):
    ph0 = _rng.uniform() if ph0 is None else ph0
    ph, _ = phase_of(freq, n, ph0)
    return np.sin(2 * np.pi * ph)


def white(n):
    return _rng.standard_normal(n)


def pink(n):
    w = np.fft.rfft(_rng.standard_normal(n))
    f = np.arange(len(w))
    f[0] = 1
    w = w / np.sqrt(f)
    x = np.fft.irfft(w, n)
    return x / (np.std(x) + 1e-12)


def brown(n):
    w = np.fft.rfft(_rng.standard_normal(n))
    f = np.arange(len(w)).astype(float)
    f[0] = 1
    w = w / f
    x = np.fft.irfft(w, n)
    return x / (np.std(x) + 1e-12)


# ------------------------------------------------------------------ filters
def _sos(kind, fc, order=2):
    nyq = SR / 2
    if kind == 'bandpass':
        lo, hi = fc
        return signal.butter(order, [max(lo, 10) / nyq, min(hi, nyq * 0.98) / nyq], 'bandpass', output='sos')
    return signal.butter(order, min(fc, nyq * 0.98) / nyq, kind, output='sos')


def lp(x, fc, order=2):
    return signal.sosfilt(_sos('lowpass', fc, order), x)


def hp(x, fc, order=2):
    return signal.sosfilt(_sos('highpass', fc, order), x)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(_sos('bandpass', (lo, hi), order), x)


def tv_filter(x, gain_fn, nperseg=1024):
    """Time-varying spectral filter. gain_fn(f[F,1], t[1,T]) -> gains."""
    nov = nperseg * 3 // 4
    f, t, Z = signal.stft(x, SR, nperseg=nperseg, noverlap=nov, boundary='even', padded=True)
    G = gain_fn(f[:, None], t[None, :])
    _, y = signal.istft(Z * G, SR, nperseg=nperseg, noverlap=nov, boundary=True)
    y = y[:len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def lp_gain(f, fc, order=2):
    return 1.0 / np.sqrt(1 + (f / np.maximum(fc, 1)) ** (2 * order))


def band_gain(f, fc, oct_bw):
    lf = np.log2(np.maximum(f, 1) / np.maximum(fc, 1))
    return np.exp(-0.5 * (lf / oct_bw) ** 2)


# ------------------------------------------------------------------ helpers
def env_exp(n, tau, attack=0.002):
    t = tt(n)
    e = np.exp(-t / tau)
    if attack > 0:
        e *= 1 - np.exp(-t / (attack / 3))
    return e


def fade(x, fi=0.003, fo=0.01):
    x = x.copy()
    a, b = N(fi), N(fo)
    if a > 1:
        x[:a] *= np.linspace(0, 1, a)
    if b > 1:
        x[-b:] *= np.linspace(1, 0, b)
    return x


def sat(x, drive=2.0):
    return np.tanh(x * drive) / np.tanh(drive)


def norm(x, peak=1.0):
    m = np.max(np.abs(x)) + 1e-12
    return x * (peak / m)


def cents(c):
    return 2 ** (c / 1200)


def smooth_noise(n, rate_hz, seed=None):
    """Band-limited random control signal in [-1,1]."""
    k = max(4, int(n / SR * rate_hz) + 4)
    pts = _rng.uniform(-1, 1, k)
    xs = np.linspace(0, n, k)
    return np.interp(np.arange(n), xs, pts)


def pan(x, p):
    """Constant-power pan, p in [-1,1] (scalar or array) -> (n,2)."""
    p = np.clip(p, -1, 1)
    a = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(a), x * np.sin(a)], axis=1)


def stereo_widen(x, width=0.3, delay_ms=11):
    """Mono -> stereo via Haas + complementary filtering."""
    d = int(delay_ms * SR / 1000)
    l = x.copy()
    r = np.concatenate([np.zeros(d), x[:-d]]) if d > 0 else x.copy()
    mid = 0.5 * (l + r)
    side = 0.5 * (l - r) * width * 2
    return np.stack([mid + side, mid - side], axis=1)


def note_hz(name):
    names = {'C': -9, 'C#': -8, 'Db': -8, 'D': -7, 'D#': -6, 'Eb': -6, 'E': -5, 'F': -4,
             'F#': -3, 'Gb': -3, 'G': -2, 'G#': -1, 'Ab': -1, 'A': 0, 'A#': 1, 'Bb': 1, 'B': 2}
    p = name[:-1]
    o = int(name[-1])
    semis = names[p] + (o - 4) * 12
    return 440.0 * 2 ** (semis / 12)


# ------------------------------------------------------------------ reverb
def make_ir(rt60=2.5, dur=None, predelay=0.012, bright=9000, dark=1800, seed=3, width=1.0):
    dur = dur or rt60 * 1.2
    n = N(dur)
    r = np.random.default_rng(seed)
    t = tt(n)
    env = 10 ** (-3 * t / rt60)
    out = []
    for ch in range(2):
        x = r.standard_normal(n) * env
        # damping: cutoff slides from bright to dark across the tail
        x = tv_filter(x, lambda f, tm: lp_gain(f, bright * (dark / bright) ** np.clip(tm / (rt60 * 0.8), 0, 1), 1))
        x = hp(x, 60)
        out.append(x)
    ir = np.stack(out, axis=1)
    if width < 1:
        m = ir.mean(axis=1, keepdims=True)
        ir = m + (ir - m) * width
    # early reflections
    for k in range(10):
        d = int((0.006 + r.uniform(0, 0.05)) * SR)
        g = r.uniform(0.2, 0.6) * (1 - k / 12)
        ir[d, r.integers(0, 2)] += g
    pd = N(predelay)
    ir = np.concatenate([np.zeros((pd, 2)), ir])
    ir /= np.sqrt(np.sum(ir ** 2) / 2)
    return ir


def convolve(x_st, ir):
    out = np.zeros((x_st.shape[0] + ir.shape[0] - 1, 2))
    for ch in range(2):
        out[:, ch] = signal.fftconvolve(x_st[:, ch], ir[:, ch])
    return out[:x_st.shape[0]]
