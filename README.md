# UMBRAZOR — Peak Fight Edit

**`UMBRAZOR_Peak_Fight.mp4`** is the finished edit. It is 28.96 s long, 1920×822 (2.33:1), 24 fps, H.264 video with 320 kbps AAC stereo audio, mastered to −12.5 LUFS.

It cuts the two Seedance takes (the "Daus" take **A** and the "UMBRAZOR" take **B**) into one sequence. The music and every sound effect are original and were synthesized in code. No third-party audio is used.

## Structure

| Time | Beat (122.55 BPM) | Section |
|---|---|---|
| 0:00 – 0:05.4 | 0 – 11 | The hunter: walk, katana grip, monster charges, eye close-up with a heartbeat as time freezes |
| 0:05.4 – 0:13.7 | 11 – 28 | Henshin: portal summon, belt ignites, dive through the portal, crystal armor, helmet eyes glow |
| 0:13.7 – 0:21.3 | 28 – 43.5 | **Drop / fight:** leap, slash with sparks, spear clash, cape energy wave, explosion, superhero landing |
| 0:21.3 – 0:25.1 | 43.5 – 51 | Finale: explosion behind the hero, wings spread, eyes flare |
| 0:25.5 – end | 52 | `UMBRAZOR` title slam |

The tempo was chosen so that the edit's impact frames fall on the beat. The drop is on beat 28, the clash on 32, the explosion on 38, the landing on 40¼, the pose explosion on 43½ and the title on 52.

## Picture
- **Speed ramps:** slow motion uses optical-flow frame interpolation (OpenCV DIS). Speed-ups get shutter-style motion blur.
- **Hit effects:** each hit gets flash frames, camera shake, a zoom punch and chromatic aberration. Slow push-ins run across the shots.
- **Grade:** a filmic S-curve, cool shadows and warm highlights so the violet energy stands out, plus bloom, vignette and film grain. Frames are upscaled with Lanczos.
- **Title card:** chrome lettering with a violet glow, RGB-split slam-in, an anamorphic light streak and rising embers. The font is Source Sans Pro Black (OFL, license included).

## Sound
- **Score:** a hybrid trailer cue in D minor. It has braams, taikos, a distorted bass ostinato, string-style spiccato, brass-style stabs, a choir pad, risers and reverse swells.
- **SFX:** monster roar, footsteps and splashes, katana grip and blade ring, portal hum, belt power-up, crystal shimmer, energy slashes, metal clangs, sparks, explosions, ground slam and rain.
- **Original clip audio:** the clips' own audio is kept at a low level as a synced texture layer.

## Rebuild
```bash
pip install numpy scipy soundfile pyloudnorm opencv-python-headless pillow
pipeline/make.sh path/to/clipA.mp4 path/to/clipB.mp4 build/
```
`pipeline/edl.py` holds the cut list and retiming. Every visual hit and every sound effect is placed from the same frame map, so picture and sound stay in sync if you change the edit.
