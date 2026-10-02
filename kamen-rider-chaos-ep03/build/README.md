# Build: Kamen Rider Chaos, Episode 3 poster

This pipeline builds `../poster.png` (1800×2546) from the two character sheets in `input/`.

| Step | Script | What it does |
|---|---|---|
| 1 | `prep.py` | Upscales source crops 4× with Real-ESRGAN (`sr.py`, then `repair.py` re-runs any corrupted tile). Cuts out Umbrazor, the Umbrazor Blade and the rider with BiRefNet (`seg1.py`). Isolates the glowing chest-emblem **3**. |
| 2 | `poster.py` | Builds the artwork: smoke, thrown ink, lightning, the two riders, the blade and the impact flash, then bloom and grading. Writes `work/art.npy`. |
| 3 | `compose.py` | Typesets the vertical slogans, the chrome カオス title (`logo.py`), the episode line, the neon **3**, the tagline and the fine print. Writes `../poster.png`. |

## Setup

```bash
pip install pillow numpy scipy opencv-python-headless rembg onnxruntime ncnn
```

- **`models/`** needs `realesrgan-x4plus.param` and `realesrgan-x4plus.bin`. Both are in `realesrgan-ncnn-vulkan-20220424-ubuntu.zip` from the Real-ESRGAN v0.2.5.0 release. BiRefNet is downloaded by rembg on first use.
- **`fonts/`** needs these files from [google/fonts](https://github.com/google/fonts) (all OFL):
  - `KaiseiTokumin-ExtraBold.ttf`
  - `ZenOldMincho-Black.ttf`
  - `ShipporiMinchoB1-Bold.ttf`
  - `Cinzel[wght].ttf`
  - `WorkSans-Regular.ttf`

## Run

```bash
python prep.py && python poster.py && python compose.py
```

## Where to change things

- **Copy:** the slogans, episode title and tagline are constants at the top of `compose.py`.
- **Layout:** the impact point `C`, the rider placements and the raised-arm re-pose (`ARM_*`) are in `poster.py`.
