"""Original score + sound design + mix for the UMBRAZOR edit.

Music: D minor hybrid-trailer cue at 122.55 BPM, grid starts at t=0 so that
the drop (first fight shot) is beat 28 and the title slam is beat 52.
"""
import sys
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d
import pyloudnorm as pyln
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import *
from instruments import *
from edl import build_timeline, src_to_out, seg_index, SEGMENTS, TITLE_FRAMES, FPS

TL, TOTAL = build_timeline()
BLACK_GAP = 8
DUR = (TOTAL + BLACK_GAP + TITLE_FRAMES) / FPS
NS = N(DUR)
DROP = src_to_out(TL, seg_index('leap'), 248)
BEAT = DROP / 28.0
TITLE_T = (TOTAL + BLACK_GAP) / FPS
VIDEO_END = TOTAL / FPS
print(f'dur {DUR:.3f}s  beat {BEAT:.5f}s ({60 / BEAT:.2f} BPM)  drop {DROP:.3f}  title {TITLE_T:.3f}')


def B(b):
    return b * BEAT


def S16(s):
    return s * BEAT / 4


def ev(name, f):
    return src_to_out(TL, seg_index(name), f)


# ------------------------------------------------------------------ buses
STEMS = ['drums', 'bass', 'music', 'hits', 'sfx', 'amb', 'orig']
stem = {k: np.zeros((NS, 2)) for k in STEMS}
SENDS = {}
IRS = {
    'hall': make_ir(rt60=3.4, bright=7000, dark=1400, seed=11),
    'room': make_ir(rt60=0.9, bright=8000, dark=2500, seed=5, predelay=0.006),
    'plate': make_ir(rt60=1.8, bright=10000, dark=3000, seed=9, predelay=0.02),
}


def _st(x, p):
    if x.ndim == 1:
        return pan(x, p)
    return x


def add(st, x, t, gain=1.0, p=0.0, hall=0.0, room=0.0, plate=0.0):
    x = _st(np.asarray(x, float), p) * gain
    k = int(round(t * SR))
    if k >= NS:
        return
    a = max(0, -k)
    m = min(len(x), NS - k)
    if m <= a:
        return
    stem[st][k + a:k + m] += x[a:m]
    for name, amt in (('hall', hall), ('room', room), ('plate', plate)):
        if amt > 0:
            key = (name, st)
            if key not in SENDS:
                SENDS[key] = np.zeros((NS, 2))
            SENDS[key][k + a:k + m] += x[a:m] * amt


KICKS = []      # times for bass sidechain
DUCKS = []      # (time, depth_db, release) for music ducking under big SFX


def duck(t, db=5, rel=0.35):
    DUCKS.append((t, db, rel))


# ------------------------------------------------------------------ pitches
D1, A1, Bb1, C2, D2, Eb2, F2, G2, A2, Bb2 = (note_hz(n) for n in
                                           ('D1', 'A1', 'Bb1', 'C2', 'D2', 'Eb2', 'F2', 'G2', 'A2', 'Bb2'))
G1 = note_hz('G1')


def tri(root_name):
    return root_name


# =============================================================== MUSIC
# ---------- intro (beats 0-28)
add('hits', impact(3.5, 90, 36, crack=0.3, body=0.5, tau=0.8), 0.0, 0.3, hall=0.35)
add('hits', st(braam, [D1, D2], 4.5, bright=0.35, drive=2.0, growl=0.15), 0.02, 0.2, hall=0.5)
dr = st(drone, D1, B(28) + 0.3, 140, 520)
# drone volume automation: dips under the eye close-up, swells into the drop
tdr = tt(len(dr))
eyes_t = ev('eyes', 111)
summon_t = ev('summon', 116)
g = np.ones_like(tdr) * 0.5
g *= np.clip(tdr / 1.2, 0, 1)
dip = (tdr > eyes_t) & (tdr < summon_t)
g[dip] *= 0.25
g *= 1 + 0.8 * np.clip((tdr - B(14)) / (B(27.5) - B(14)), 0, 1)
g[tdr > B(27.6)] *= np.exp(-(tdr[tdr > B(27.6)] - B(27.6)) / 0.05)
add('music', dr * g[:, None], 0.0, 0.42, hall=0.15)
add('music', st(eerie, eyes_t + 0.25), 0.0, 0.09, hall=0.45)

# ticking clock 8ths, beat 4 -> eyes
b = 4.0
while B(b) < eyes_t - 0.05:
    add('drums', tick(1.0 if (b * 2) % 2 == 0 else 0.55), B(b), 0.07, p=0.35 if (b * 2) % 2 else -0.35, room=0.3)
    b += 0.5

# monster entrance
mon_t = ev('monster', 57)
add('hits', st(braam, [D1, A1], 3.0, bright=0.7, drive=3.5, growl=0.4), mon_t, 0.38, hall=0.35)
add('drums', taiko(0.9, 1.3), mon_t, 0.65, hall=0.25)
duck(mon_t, 3)
# taiko footsteps (quarters) through the charge, building
for k, bb in enumerate((5, 6, 7, 8)):
    add('drums', taiko(1.0 + 0.04 * k, 1.0), B(bb), 0.32 + 0.07 * k, p=(-0.2, 0.2)[k % 2], hall=0.18)
    add('drums', taiko(1.5, 0.6), B(bb + 0.5), 0.16 + 0.03 * k, p=(0.3, -0.3)[k % 2], hall=0.15)
# low ostinato under the charge (8ths), Phrygian colour
ost = [D2, D2, F2, D2, Eb2, D2, F2, Eb2]
b = 4.0
i = 0
while B(b) < eyes_t - 0.05:
    add('music', spiccato(ost[i % 8] * 2, 0.9, 0.2), B(b), 0.16, p=-0.25, hall=0.2)
    add('bass', bass_note(ost[i % 8] / 2, BEAT * 0.45, 0.6), B(b), 0.22)
    b += 0.5
    i += 1
lunge_t = ev('lunge', 100)
add('drums', taiko(0.8, 1.2), lunge_t, 0.6, hall=0.3)
add('drums', tom(0.8), lunge_t + 0.06, 0.35)

# eye close-up: time freeze (heartbeat + tinnitus) then reverse swell into the summon
add('sfx', heartbeat(), eyes_t + 0.04, 0.9)
add('sfx', heartbeat(), eyes_t + 0.52, 0.75)
add('sfx', tinnitus(summon_t - eyes_t + 0.2, 5600), eyes_t, 0.012, hall=0.2)
rs = reverse_swell(summon_t - eyes_t + 0.05)
add('music', rs, summon_t - len(rs) / SR, 0.22, hall=0.2)

# summon portal
add('hits', st(braam, [D1, A1, D2], 3.5, bright=1.0, drive=3.0), summon_t, 0.42, hall=0.45)
add('hits', impact(3.0, 140, 30, crack=0.6), summon_t, 0.75, hall=0.3)
add('drums', crash(3.0, 1.0), summon_t, 0.22, p=0.2, hall=0.2)
duck(summon_t, 4)

# belt / henshin
belt_t = ev('belt', 139)
flare_t = ev('belt', 143)
add('hits', impact(3.0, 160, 32, crack=0.8), flare_t, 0.7, hall=0.35)
add('hits', st(braam, [D2, A2, D2 * 2], 3.2, bright=1.3, drive=3.0, growl=0.3), flare_t, 0.38, hall=0.5)
add('drums', crash(2.5), flare_t, 0.2, p=-0.2, hall=0.2)
duck(flare_t, 4)

# build: bass 8ths (beats 14-20) then 16ths (20-26)
def bass_line(b0, b1, step, roots, accent_pat=None):
    b = b0
    j = 0
    while b < b1 - 1e-6:
        bar_pos = b - b0
        r = roots(b)
        oct_up = (j % 4 == 2)
        f = r * (2 if oct_up else 1)
        acc = 1.0 if (j % 4 == 0) else 0.6
        if accent_pat is not None:
            acc = accent_pat[j % len(accent_pat)]
        add('bass', bass_note(f, BEAT * step * 0.82, acc), B(b), 0.5)
        b += step
        j += 1


bass_line(14, 20, 0.5, lambda b: D1)
bass_line(20, 26, 0.25, lambda b: (D1 if b < 24 else Bb1 if b < 25 else A1))

# spiccato 16ths beats 16-26 (Dm, then Bb, then A)
arps = {'Dm': ['D4', 'F4', 'A4', 'F4'], 'Bb': ['Bb3', 'D4', 'F4', 'D4'], 'A': ['A3', 'C#4', 'E4', 'C#4'],
        'Gm': ['G3', 'Bb3', 'D4', 'Bb3'], 'Dm5': ['D5', 'A4', 'F4', 'A4']}


def spic_line(b0, b1, chord_at, vel=1.0, gain=0.17, octave=1.0, p=0.3):
    b = b0
    j = 0
    while b < b1 - 1e-6:
        ch = chord_at(b)
        f = note_hz(arps[ch][j % 4]) * octave
        v = vel * (1.0 if j % 4 == 0 else 0.75)
        add('music', spiccato(f, v, 0.2), B(b), gain, p=p if j % 2 else -p, hall=0.22)
        b += 0.25
        j += 1


spic_line(16, 26, lambda b: 'Dm' if b < 20 else 'Bb' if b < 24 else 'A', gain=0.15)

# taikos in the build: quarters + pickups, intensifying
for bb in np.arange(14, 26, 1.0):
    gtk = 0.45 + 0.35 * (bb - 14) / 12
    add('drums', taiko(1.0, 1.0), B(bb), gtk, p=-0.15, hall=0.2)
    if bb >= 18:
        add('drums', taiko(1.35, 0.7), B(bb + 0.5), gtk * 0.55, p=0.25, hall=0.15)
    if bb >= 22:
        add('drums', taiko(1.6, 0.5), B(bb + 0.75), gtk * 0.4, p=0.35, hall=0.15)
    KICKS.append(B(bb))

# braams on the dive and emerge
dive_t = ev('dive', 203)
emerge_t = ev('emerge', 196)
add('hits', st(braam, [D1, D2, F2, A2], 3.5, bright=1.1, drive=3.0), dive_t, 0.4, hall=0.5)
add('hits', impact(3.0, 110, 28, crack=0.4), dive_t, 0.6, hall=0.4)
add('hits', sub_drop(2.5, 60, 25), dive_t, 0.5)
add('hits', st(braam, [Bb1, D2, F2, Bb2], 3.0, bright=1.2, drive=3.0), emerge_t, 0.4, hall=0.5)
add('hits', impact(2.5, 130, 30, crack=0.6), emerge_t, 0.65, hall=0.35)
add('drums', crash(2.5), emerge_t, 0.2, p=0.25, hall=0.2)
duck(dive_t, 4)
duck(emerge_t, 4)

# snare roll 24 -> 27.5 accelerating, crescendo
b = 24.0
while b < 27.5 - 1e-6:
    step = 0.5 if b < 25 else 0.25 if b < 26.5 else 0.125
    v = 0.25 + 0.75 * ((b - 24) / 3.5) ** 1.5
    add('drums', snare_roll_hit(v), B(b), 0.55, p=0.15, hall=0.25)
    b += step

# riser 20 -> 27.6, then reverse swell into the drop
rz = riser(B(27.6) - B(20), 180, 9000)
add('music', rz, B(20), 0.33, hall=0.2)
eyeglow_t = ev('helmet', 241)
rs = reverse_swell(DROP - eyeglow_t)
add('music', rs, DROP - len(rs) / SR, 0.3, hall=0.15)

# ---------- DROP / FIGHT (beats 28-48)
def mega_hit(t, chord, gain=1.0, title=False):
    add('hits', impact(4.5, 120, 26, crack=0.9, body=1.0, tau=1.1), t, 0.95 * gain, hall=0.35)
    add('hits', st(braam, chord, 5.0 if title else 4.0, bright=1.3, drive=3.2, growl=0.35), t, 0.55 * gain, hall=0.55)
    add('hits', sub_drop(3.0, 75, 24), t, 0.6 * gain)
    add('drums', crash(3.5, 1.2), t, 0.3 * gain, p=-0.25, hall=0.25)
    add('drums', crash(3.5, 1.0), t + 0.01, 0.25 * gain, p=0.25, hall=0.25)
    for d, pt, pp in ((0.0, 0.9, -0.3), (0.012, 1.0, 0.0), (0.025, 1.12, 0.3)):
        add('hits', taiko(pt, 1.4), t + d, 0.55 * gain, p=pp, hall=0.3)
    duck(t, 6, 0.5)


mega_hit(DROP, [D1, D2, A2])

# bar chords for the fight: 28 Dm | 32 Bb | 36 Gm | 40 A | 44 Dm (half-time)
def root_at(b):
    if b < 32: return D1
    if b < 36: return Bb1
    if b < 38: return G1
    if b < 40: return G1
    if b < 44: return A1
    return D1


def chord_at(b):
    if b < 32: return 'Dm'
    if b < 36: return 'Bb'
    if b < 40: return 'Gm'
    if b < 44: return 'A'
    return 'Dm'


kick_steps = [0, 6, 10]
taiko_steps = [0, 7, 12, 14]
bass_acc = [1.2, 0.5, 0.6, 1.0, 0.5, 0.6, 1.1, 0.5, 0.6, 0.5, 1.1, 0.5, 0.6, 1.0, 0.6, 0.7]
boom_t = ev('boom', 267)
for bar0 in (28, 32, 36, 40):
    for s in range(16):
        b = bar0 + s / 4
        t = B(b)
        # stop-time right before the explosion (beats 37.5-38) and its slow-mo (38-40)
        stop = 37.5 <= b < 40
        if not stop:
            if s in kick_steps:
                add('drums', kick(), t, 0.8)
                KICKS.append(t)
            if s == 8:
                add('drums', snare_big(), t, 0.55, p=0.05, hall=0.35)
            if s in taiko_steps:
                add('drums', taiko(1.0 if s == 0 else 1.25, 1.0), t, 0.45 if s == 0 else 0.3,
                    p=-0.2 if s % 2 else 0.2, hall=0.2)
            add('drums', hat(open_=(s % 4 == 2), vel=1.0 if s % 2 == 0 else 0.55), t, 0.16,
                p=0.4 if s % 2 else 0.25)
            add('bass', bass_note(root_at(b) * (2 if s in (3, 10, 13) else 1), BEAT / 4 * 0.85, bass_acc[s]), t, 0.5)
        # fills at the end of bars 32 and 40
        if bar0 in (28, 40) and s >= 12:
            add('drums', tom(1.3 - 0.1 * (s - 12)), t, 0.4, p=0.5 - 0.3 * (s - 12), hall=0.2)
    if bar0 != 36:
        spic_line(bar0, bar0 + 4, chord_at, vel=1.0, gain=0.26, octave=1.0, p=0.35)
        spic_line(bar0, bar0 + 4, chord_at, vel=0.8, gain=0.1, octave=2.0, p=-0.5)
    else:
        spic_line(36, 37.5, chord_at, vel=1.0, gain=0.14, p=0.35)

# brass-style chord stabs on the groove accents + sustained string bed
STAB_CH = {'Dm': ['D3', 'A3', 'D4', 'F4'], 'Bb': ['Bb2', 'F3', 'Bb3', 'D4'], 'Gm': ['G2', 'D3', 'G3', 'Bb3'],
           'A': ['A2', 'E3', 'A3', 'C#4']}
for bar0 in (28, 32, 40):
    for s_, v in ((0, 1.0), (3, 0.75), (6, 0.9), (10, 0.9), (13, 0.75)):
        b = bar0 + s_ / 4
        add('music', st(stab, [note_hz(q) for q in STAB_CH[chord_at(b)]], 0.13, v), B(b), 0.42 * v, hall=0.25)
for b in np.arange(24, 26, 0.5):
    ch = 'A' if b >= 25 else 'Bb'
    add('music', st(stab, [note_hz(q) for q in STAB_CH[ch]], 0.12, 0.6 + 0.2 * (b - 24)), B(b), 0.13, hall=0.3)
for b0, b1, ch in ((28, 32, 'Dm'), (32, 36, 'Bb'), (36, 37.5, 'Gm'), (40, 44, 'A')):
    add('music', st(pad, [note_hz(q) for q in STAB_CH[ch]], B(b1) - B(b0) + 0.3, attack=0.12, release=0.35,
                    cutoff=2600), B(b0), 0.3, hall=0.35)

# braam accents on the fight downbeats
add('hits', st(braam, [Bb1, D2, F2], 2.5, bright=1.0, drive=3.0), B(32), 0.38, hall=0.4)
mega_hit(boom_t, [G1, D2, G2, Bb2], 0.95)
add('music', st(pad, [G2, Bb2, D2 * 2], B(40) - boom_t + 0.4, attack=0.05, release=0.4, cutoff=1400), boom_t, 0.22, hall=0.4)
add('bass', bass_note(G1, B(40) - boom_t, 1.2, 2.0), boom_t, 0.45)
land_t = ev('land', 337)
add('hits', st(braam, [A1, E := note_hz('E2'), A2], 2.5, bright=1.0), B(40), 0.36, hall=0.4)

# ---------- FINALE (beats 43.5-52)
pose_t = ev('pose', 293)
mega_hit(pose_t, [D1, D2, F2, A2], 0.9)
add('music', st(choir, [note_hz('D3'), note_hz('F3'), note_hz('A3'), note_hz('D4')], VIDEO_END - pose_t + 0.6,
                   attack=0.5, release=1.0), pose_t, 0.32, hall=0.6)
add('music', st(pad, [D2, A2, D2 * 2, note_hz('F3')], VIDEO_END - pose_t + 0.6, attack=0.4, release=0.8, cutoff=1100),
    pose_t, 0.28, hall=0.5)
# half-time groove beats 44-48
for bar0 in (44,):
    for s in range(16):
        b = bar0 + s / 4
        t = B(b)
        if s in (0, 10):
            add('drums', kick(), t, 0.8)
            KICKS.append(t)
        if s == 8:
            add('drums', snare_big(), t, 0.55, hall=0.4)
        if s in (0, 6, 12):
            add('drums', taiko(1.0, 1.2), t, 0.45, hall=0.25)
        if s % 2 == 0:
            add('drums', hat(False, 0.8), t, 0.12, p=0.3)
        add('bass', bass_note(D1 * (2 if s in (6, 14) else 1), BEAT / 4 * 0.85, bass_acc[s] * 0.9), t, 0.42)
spic_line(44, 48, lambda b: 'Dm5', vel=0.9, gain=0.1, p=0.4)
wings_t = ev('wings', 395)
eyeflare_t = ev('wings', 412)
add('hits', st(braam, [D1, A1, D2], 3.0, bright=0.8, drive=2.5), B(48), 0.34, hall=0.5)
add('drums', taiko(0.85, 1.5), B(48), 0.6, hall=0.4)
# into the title: riser + reverse swell, then silence over the black frames
rz = riser(VIDEO_END - B(49), 300, 10000)
add('music', rz, B(49), 0.28, hall=0.15)
rs = reverse_swell(1.4)
add('music', rs, VIDEO_END - len(rs) / SR, 0.3)

# ---------- TITLE (beat 52)
mega_hit(TITLE_T, [D1, D2, F2, A2], 1.1, title=True)
add('music', clang(196, 4.0, 0.6), TITLE_T, 0.3, hall=0.5)
add('music', st(choir, [note_hz('D3'), note_hz('A3'), note_hz('D4'), note_hz('F4')], DUR - TITLE_T,
                   attack=0.05, release=2.6), TITLE_T, 0.35, hall=0.7)
add('music', drone(D1, DUR - TITLE_T, 400, 120) * np.linspace(1, 0, N(DUR - TITLE_T)) ** 1.5, TITLE_T, 0.35, hall=0.2)

# =============================================================== SOUND DESIGN
add('amb', rain(VIDEO_END + 0.1), 0.0, 1.0)

# intro
grip_t = ev('grip', 44)
add('sfx', creak(0.55), grip_t + 0.05, 0.35, p=0.2, room=0.3)
add('sfx', shing(1.6, 2349), grip_t + 0.25, 0.22, p=0.15, hall=0.3)
add('sfx', roar(1.5, 78), mon_t + 0.05, 0.42, p=-0.1, room=0.3, hall=0.2)
add('sfx', roar(1.1, 60), mon_t + 0.95, 0.25, p=0.1, room=0.3)
for k, tf in enumerate((B(5), B(6), B(7), B(8))):
    add('sfx', footstep_heavy(), tf + 0.01, 0.45, p=(-0.2, 0.2)[k % 2], room=0.3)
add('sfx', whoosh(0.45, 300, 2200, 0.5, 0.5, -0.6), lunge_t - 0.15, 0.4, room=0.2)
add('sfx', roar(0.7, 95), lunge_t + 0.05, 0.25, p=-0.4, room=0.3)
# summon
add('sfx', whoosh(0.5, 200, 3000, 0.75, -0.2, 0.2), summon_t - 0.35, 0.4)
add('sfx', zap(0.7, 3000, 120), summon_t, 0.32, hall=0.3)
add('sfx', crackle(1.2, 150, 0.5), summon_t, 0.3, p=0.2, room=0.3)
add('sfx', portal_hum(belt_t - summon_t + 0.1, 55), summon_t, 0.22, room=0.2)
add('sfx', whoosh(0.4, 400, 5000, 0.5, -0.9, 0.9, 0.4), ev('whip', 130) - 0.05, 0.5)
# henshin
add('sfx', powerup(flare_t - 5.85, 200, 2400), 5.85, 0.16, hall=0.2)
add('sfx', mech_click(), belt_t + 0.02, 0.45, room=0.2)
add('sfx', shing(2.0, 2637), flare_t, 0.3, hall=0.4)
add('sfx', crackle(1.0, 120, 0.4), flare_t, 0.25, p=-0.2)
portalA_t = ev('portalA', 147)
add('sfx', zap(0.6, 2500, 100), portalA_t, 0.25, hall=0.3)
add('sfx', portal_hum(ev('run', 161) - portalA_t, 49), portalA_t, 0.25)
add('sfx', whoosh(1.0, 200, 1800, 0.6, 0.6, -0.4), portalA_t + 0.1, 0.3)
run_t = ev('run', 161)
for k in range(5):
    add('sfx', splash(0.3), run_t + 0.1 + k * 0.22, 0.12, p=-0.3 + 0.15 * k, room=0.2)
add('sfx', whoosh(1.3, 150, 4000, 0.4, -0.3, 0.3), dive_t - 0.2, 0.5, hall=0.2)
add('sfx', portal_hum(emerge_t - dive_t, 41), dive_t, 0.3, hall=0.2)
add('sfx', shimmer(emerge_t - dive_t - 0.2, 55, True), dive_t + 0.3, 0.3, p=0.2, hall=0.4)
add('sfx', crackle(2.0, 90, 1.5, 3000, 10000), dive_t + 0.2, 0.18, p=-0.3, hall=0.2)
add('sfx', flame_roar(1.4), emerge_t, 0.5, room=0.3)
for d, bs in ((0.35, 520), (0.7, 610), (1.05, 470)):
    add('sfx', clang(bs, 0.6, 0.5), emerge_t + d, 0.1, p=0.3 - 0.3 * d, room=0.4)
add('sfx', shing(2.4, 3136), eyeglow_t, 0.35, hall=0.5)
add('sfx', tinnitus(DROP - eyeglow_t, 6300), eyeglow_t, 0.01)

# fight
add('sfx', whoosh(0.7, 200, 3500, 0.35, -0.6, 0.6), DROP, 0.55, room=0.2)
add('sfx', zap(0.8, 3200, 120), DROP, 0.25, hall=0.3)
slash_t = ev('slash', 281)
spark_t = ev('slash', 289)
add('sfx', energy_slash(0.7), slash_t, 0.6, hall=0.3)
add('sfx', crackle(0.8, 260, 0.3, 2500, 11000), spark_t, 0.5, p=0.4, room=0.3)
add('sfx', clang(640, 1.0, 1.0), spark_t, 0.22, p=0.3, room=0.3)
clash_t = ev('clash', 296)
add('sfx', clang(380, 1.8, 1.0), clash_t, 0.55, p=0.1, hall=0.3)
add('sfx', impact(1.5, 150, 40, crack=1.0, body=0.6, tau=0.3), clash_t, 0.6, room=0.3)
add('sfx', crackle(0.7, 250, 0.3), clash_t, 0.4, p=0.3)
duck(clash_t, 5)
add('sfx', roar(0.9, 70), ev('clash', 303), 0.3, p=0.4, room=0.3)
add('sfx', swish(0.25, 6000), ev('clash', 303) + 0.02, 0.4, p=0.3)
add('sfx', whoosh(0.35, 400, 4500, 0.5, 0.7, -0.7), ev('clash', 311) - 0.05, 0.45)
clash2_t = ev('clash', 322)
add('sfx', energy_slash(0.6), clash2_t - 0.08, 0.6, hall=0.3)
add('sfx', impact(1.2, 170, 45, crack=1.0, body=0.5, tau=0.25), clash2_t, 0.5, room=0.3)
add('sfx', clang(455, 1.2, 1.0), clash2_t, 0.3, p=-0.2, room=0.3)
duck(clash2_t, 4)
add('sfx', whoosh(0.35, 500, 6000, 0.4, -0.8, 0.8, 0.4), ev('blur', 324), 0.5)
cape_t = ev('cape', 228)
add('sfx', cloth_flap(0.9), cape_t, 0.4, p=-0.2, room=0.3)
add('sfx', whoosh(0.9, 120, 1500, 0.5, -0.4, 0.4), cape_t + 0.1, 0.4)
wave_t = ev('wave', 250)
add('sfx', whoosh(0.9, 150, 5000, 0.35, -0.7, 0.7, 0.5), wave_t - 0.15, 0.7, hall=0.2)
add('sfx', zap(0.9, 2000, 80), wave_t, 0.35, hall=0.3)
add('sfx', flame_roar(0.9), wave_t, 0.4)
add('sfx', crackle(0.8, 200, 0.5), wave_t, 0.35, p=0.3)
add('sfx', explosion(4.5, 1.6), boom_t, 0.95, hall=0.25)
add('sfx', debris(1.5, 40), boom_t + 0.25, 0.3, p=0.3, room=0.3)
duck(boom_t, 6, 0.7)
add('sfx', whoosh(0.8, 200, 2500, 0.7, 0.0, 0.0), land_t - 0.6, 0.45)
add('sfx', impact(2.5, 110, 30, crack=0.7, body=1.0, tau=0.6), land_t, 0.85, room=0.3, hall=0.2)
add('sfx', debris(1.2, 45), land_t + 0.02, 0.4, p=-0.2, room=0.3)
add('sfx', splash(0.6), land_t, 0.4, p=0.2, room=0.3)
duck(land_t, 5)
add('sfx', portal_hum(1.0, 55), ev('land', 346), 0.15)
add('sfx', zap(0.6, 1800, 60), ev('land', 348), 0.2, hall=0.3)

# finale
add('sfx', whoosh(0.6, 150, 1200, 0.85, 0.0, 0.0), pose_t - 0.5, 0.35)
add('sfx', explosion(4.0, 1.3), pose_t + 0.1, 0.8, hall=0.25)
add('sfx', flame_roar(2.0), pose_t + 0.15, 0.35, p=0.2)
duck(pose_t, 6)
add('sfx', whoosh(1.2, 120, 2000, 0.4, -0.5, 0.5), wings_t - 0.3, 0.45, hall=0.2)
add('sfx', cloth_flap(1.1), wings_t - 0.1, 0.35, p=0.3, room=0.3)
add('sfx', flame_roar(1.6), ev('wings', 390), 0.3, p=-0.3)
add('sfx', shing(2.5, 2794), eyeflare_t, 0.32, hall=0.5)
add('sfx', powerup(1.0, 300, 3000), eyeflare_t - 0.95, 0.08, hall=0.2)

# =============================================================== ORIGINAL CLIP AUDIO (synced texture)
def load_orig(path):
    x, sr = sf.read(path, always_2d=True)
    x = signal.resample_poly(x, SR, sr, axis=0)
    return x


ORIG = {'A': load_orig('A_orig_st.wav'), 'B': load_orig('B_orig_st.wav')}
orig_gain = {'walk': 0.55, 'face': 0.5, 'grip': 0.5, 'monster': 0.55, 'crouch': 0.5, 'lunge': 0.5,
             'eyes': 0.2, 'summon': 0.4, 'whip': 0.35, 'belt': 0.3, 'portalA': 0.35, 'run': 0.4,
             'dive': 0.3, 'emerge': 0.35, 'helmet': 0.0}
for i, start, pos in TL:
    seg = SEGMENTS[i]
    gseg = orig_gain.get(seg['name'], 0.32)
    if gseg <= 0:
        continue
    src = ORIG[seg['src']]
    n_out = len(pos)
    t0 = start / FPS
    ks = N(n_out / FPS)
    out_t = np.arange(ks) / SR
    # source time for each output sample (frame mapping is piecewise linear)
    fr_out = np.arange(n_out + 1) / FPS
    last_step = pos[-1] - pos[-2] if len(pos) > 1 else 1.0
    fr_src = np.concatenate([pos, [pos[-1] + last_step]]) / FPS
    s_t = np.interp(out_t, fr_out, fr_src)
    idx = s_t * SR
    seg_audio = np.stack([np.interp(idx, np.arange(len(src)), src[:, c]) for c in range(2)], axis=1)
    f = N(0.006)
    seg_audio[:f] *= np.linspace(0, 1, f)[:, None]
    seg_audio[-f:] *= np.linspace(1, 0, f)[:, None]
    add('orig', seg_audio, t0, gseg)

# =============================================================== MIX
print('reverbs...')
for (name, st), buf in SENDS.items():
    stem[st] += convolve(buf, IRS[name])

t_all = tt(NS)


def env_from(times, depth_db, rel, attack=0.004):
    g = np.zeros(NS)
    for t in times:
        k = int(t * SR)
        if k >= NS:
            continue
        m = min(NS - k, N(rel * 5))
        tr = tt(m)
        g[k:k + m] = np.maximum(g[k:k + m], np.exp(-tr / rel) * (1 - np.exp(-tr / attack)))
    return 10 ** (-depth_db * g / 20)


# bass sidechain from kicks/taikos
stem['bass'] *= env_from(KICKS, 6, 0.09)[:, None]
# duck music+drums under the big sound-design hits
gd = np.ones(NS)
for t, db, rel in DUCKS:
    gd = np.minimum(gd, env_from([t], db * 0.6, min(rel, 0.3), 0.003))
for k in ('music', 'bass', 'amb', 'orig'):
    stem[k] *= gd[:, None]
# ambience: rain lower during the fight, gone before title
amb_g = np.interp(t_all, [0, DROP - 0.2, DROP, VIDEO_END - 1.0, VIDEO_END], [1, 1, 0.45, 0.45, 0])
stem['amb'] *= amb_g[:, None]
# original audio: tame lows (score owns the sub), tuck under
for c in range(2):
    stem['orig'][:, c] = hp(lp(stem['orig'][:, c], 9000), 90)

GAINS = {'drums': 0.85, 'bass': 0.7, 'music': 0.8, 'hits': 0.75, 'sfx': 0.85, 'amb': 0.3, 'orig': 1.1}

# dynamic arc: quieter intro, rising build, full-scale fight
arc_pts = [(0, -6), (2.05, -6), (2.15, -5), (5.3, -5), (5.45, -3.5), (6.3, -5.5), (13.6, -1.5), (13.7, 0), (DUR, 0)]
arc = 10 ** (np.interp(t_all, [p[0] for p in arc_pts], [p[1] for p in arc_pts]) / 20)
sfx_pts = [(0, -3.5), (2.1, -4.5), (6.3, -3.5), (13.0, -1.5), (13.7, 0), (DUR, 0)]
arc_sfx = 10 ** (np.interp(t_all, [p[0] for p in sfx_pts], [p[1] for p in sfx_pts]) / 20)
for k in ('drums', 'bass', 'music', 'hits'):
    stem[k] *= arc[:, None]
stem['sfx'] *= arc_sfx[:, None]


def _rbj(kind, f0, gain_db, q=0.707):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    cw, sw = np.cos(w0), np.sin(w0)
    if kind == 'peak':
        al = sw / (2 * q)
        b = [1 + al * A, -2 * cw, 1 - al * A]
        a = [1 + al / A, -2 * cw, 1 - al / A]
    else:
        al = sw / 2 * np.sqrt(2)
        sA = 2 * np.sqrt(A) * al
        if kind == 'low':
            b = [A * ((A + 1) - (A - 1) * cw + sA), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sA)]
            a = [(A + 1) + (A - 1) * cw + sA, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sA]
        else:
            b = [A * ((A + 1) + (A - 1) * cw + sA), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sA)]
            a = [(A + 1) - (A - 1) * cw + sA, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sA]
    return np.array(b) / a[0], np.array(a) / a[0]


def master_eq(x):
    y = x.copy()
    for c in range(2):
        v = hp(y[:, c], 32, 4)
        for kind, f0, gdb, q in (('low', 75, -3.5, 0.7), ('peak', 2800, 2.5, 0.9), ('high', 9000, 1.5, 0.7)):
            b, a = _rbj(kind, f0, gdb, q)
            v = signal.lfilter(b, a, v)
        y[:, c] = v
    return y


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


for k in STEMS:
    print(f'stem {k:6s} rms {rms_db(stem[k] * GAINS[k]):6.1f} dB  peak {20 * np.log10(np.max(np.abs(stem[k] * GAINS[k])) + 1e-12):6.1f}')
    sf.write(f'stem_{k}.wav', (stem[k] * GAINS[k] * 0.25).astype(np.float32), SR)

mix = master_eq(sum(stem[k] * GAINS[k] for k in STEMS))


# glue compressor (RMS, 2:1 above threshold)
def compress(x, thr_db=-16, ratio=2.2, tau=0.08):
    lvl = np.sqrt(signal.lfilter([1 - np.exp(-1 / (tau * SR))], [1, -np.exp(-1 / (tau * SR))], np.mean(x ** 2, axis=1)) + 1e-12)
    ldb = 20 * np.log10(lvl)
    over = np.maximum(ldb - thr_db, 0)
    gr = over * (1 - 1 / ratio)
    return x * (10 ** (-gr / 20))[:, None]


def limit(x, ceiling_db=-1.5, look=0.003, rel=0.06):
    c = 10 ** (ceiling_db / 20)
    peak = np.max(np.abs(x), axis=1)
    g = np.minimum(1.0, c / np.maximum(peak, 1e-9))
    w = N(look)
    g = minimum_filter1d(g, size=2 * w + 1)
    g = uniform_filter1d(g, size=w)
    # release smoothing (only let gain recover slowly)
    a = np.exp(-1 / (rel * SR))
    g = np.minimum(g, signal.lfilter([1 - a], [1, -a], g, zi=[g[0] * a])[0])
    y = x * g[:, None]
    return np.clip(y, -c, c)


mix = compress(mix)
meter = pyln.Meter(SR)
pre = meter.integrated_loudness(mix)
target = -12.5
mix *= 10 ** ((target - pre) / 20)
for _ in range(3):
    lim = limit(mix * 1.0)
    l = meter.integrated_loudness(lim)
    if abs(l - target) < 0.2:
        break
    mix *= 10 ** ((target - l) / 20)
out = limit(mix)
pk = np.max(np.abs(mix), axis=1)
gr = 20 * np.log10(np.maximum(pk, 1e-9) / np.maximum(np.max(np.abs(out), axis=1), 1e-9))
print(f'limiter GR: max {gr.max():.1f} dB, >3dB {np.mean(gr > 3) * 100:.1f}% of time, >6dB {np.mean(gr > 6) * 100:.1f}%')
# tiny fades at both ends
f0, f1 = N(0.01), N(0.3)
out[:f0] *= np.linspace(0, 1, f0)[:, None]
out[-f1:] *= np.linspace(1, 0, f1)[:, None]
print('integrated LUFS', round(meter.integrated_loudness(out), 2), 'peak dBFS', round(20 * np.log10(np.max(np.abs(out))), 2))
sf.write('score_mix.wav', out.astype(np.float32), SR, subtype='FLOAT')
print('wrote score_mix.wav', out.shape[0] / SR, 's')
