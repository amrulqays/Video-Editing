"""Scan an upscaled image tile by tile against its source; re-run any tile that came out corrupted."""
import sys
import numpy as np
import cv2
from PIL import Image
import sr

src, box, dst = sys.argv[1], sys.argv[2], sys.argv[3]
im = Image.open(src).convert('RGB')
if box != "full":
    im = im.crop(tuple(int(v) for v in box.split(',')))
a = np.asarray(im).astype(np.float32) / 255
out = np.asarray(Image.open(dst).convert('RGB')).astype(np.float32) / 255
T, P, S = sr.TILE, sr.PAD, sr.S
h, w, _ = a.shape
net = None


def err(r, ref):
    d = cv2.resize(r, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.abs(d - ref).mean())


fixed = 0
for y0 in range(0, h, T):
    for x0 in range(0, w, T):
        y1, x1 = min(y0 + T, h), min(x0 + T, w)
        cur = out[y0 * S:y1 * S, x0 * S:x1 * S]
        best = err(cur, a[y0:y1, x0:x1])
        if best < 0.04:
            continue
        if net is None:
            net = sr.make_net()
        py0, px0, py1, px1 = max(y0 - P, 0), max(x0 - P, 0), min(y1 + P, h), min(x1 + P, w)
        oy, ox = (y0 - py0) * S, (x0 - px0) * S
        for attempt in range(6):
            r = np.clip(sr.run_tile(net, a[py0:py1, px0:px1]), 0, 1)[oy:oy + (y1 - y0) * S, ox:ox + (x1 - x0) * S]
            if np.abs(r - cur).mean() < 2e-3:
                break  # reproduces the existing tile: it is just detailed, not corrupted
            e = err(r, a[y0:y1, x0:x1])
            if e < best:
                best, cur = e, r
                out[y0 * S:y1 * S, x0 * S:x1 * S] = r
                fixed += 1
            if best < 0.04:
                break
Image.fromarray((out * 255 + .5).astype(np.uint8)).save(dst)
print(dst, 'tiles repaired:', fixed)
