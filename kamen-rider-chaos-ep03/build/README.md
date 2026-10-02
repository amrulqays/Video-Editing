# Build: Kamen Rider Chaos, Episode 3 key art

These scripts build both pieces from the source images in `input/`:

- `../banner-21x9.png`: the 21:9 key art, 3360×1440
- `../poster.png`: the portrait one-sheet, 1800×2546

## 21:9 key art

| Step | Script | What it does |
|---|---|---|
| 1 | `prep_banner.py` | Upscales the Chaos helmet (from `episode1-poster.webp`) and the Umbrazor close-up 4× with Real-ESRGAN. Cuts out the logo, the rider with his machine, the helmet and Umbrazor with BiRefNet. |
| 2 | `banner.py` | Composites the void, the HUD-crest hologram, the graded tunnel, the two giants with rim light, the clash mist, embers, eye flares, bloom, the logo and the type. Writes `../banner-21x9.png`. |

**Copy:** the tagline, episode title, `DATE`, `TIME` and billing line are constants at the top of `banner.py`.

**Layout:** the placement constants (`CH_*`, `UM_*`, `TN_*`, `LOGO_W`) sit just below the copy constants.

## Portrait one-sheet

| Step | Script | What it does |
|---|---|---|
| 1 | `prep.py` | Upscales source crops 4× with Real-ESRGAN. Cuts out Umbrazor, the Umbrazor Blade and the rider with BiRefNet. Isolates the glowing chest-emblem **3**. |
| 2 | `poster.py` | Builds the artwork: smoke, thrown ink, lightning, the two riders, the blade and the impact flash, then bloom and grading. Writes `work/art.npy`. |
| 3 | `compose.py` | Typesets the vertical slogans, the chrome カオス title (`logo.py`), the episode line, the neon **3**, the tagline and the fine print. Writes `../poster.png`. |

**Copy:** the slogans, episode title and tagline are constants at the top of `compose.py`.

**Layout:** the impact point `C`, the rider placements and the raised-arm re-pose (`ARM_*`) are in `poster.py`.

## Shared tools

- `sr.py` runs Real-ESRGAN x4 tiled on the CPU through ncnn.
- `repair.py` re-runs any tile that ncnn returned corrupted.
- `seg1.py` produces one BiRefNet cut-out per process.

## Setup

```bash
pip install pillow numpy scipy opencv-python-headless rembg onnxruntime ncnn
```

**`models/`** needs `realesrgan-x4plus.param` and `realesrgan-x4plus.bin`. Both are in `realesrgan-ncnn-vulkan-20220424-ubuntu.zip` from the Real-ESRGAN v0.2.5.0 release. rembg downloads BiRefNet on first use, and each cut-out needs about 6 GB of free memory.

**`fonts/`** needs these files from [google/fonts](https://github.com/google/fonts) (all OFL):

- `KaiseiTokumin-ExtraBold.ttf`
- `ZenOldMincho-Black.ttf`
- `ShipporiMinchoB1-Bold.ttf`
- `Cinzel[wght].ttf`
- `CormorantGaramond[wght].ttf`
- `Oswald[wght].ttf`
- `WorkSans-Regular.ttf`

## Run

```bash
python prep_banner.py && python banner.py                    # 21:9 key art
python prep.py && python poster.py && python compose.py      # portrait one-sheet
```
