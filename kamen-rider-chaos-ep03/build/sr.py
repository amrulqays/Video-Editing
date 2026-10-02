"""Real-ESRGAN x4 upscaling on CPU via ncnn, tiled with overlap."""
import os
import sys
import numpy as np
import ncnn
from PIL import Image

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "realesrgan-x4plus")
TILE, PAD, S = 192, 16, 4


def make_net():
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 8
    net.load_param(MODEL + ".param")
    net.load_model(MODEL + ".bin")
    return net


def run_tile(net, arr):
    h, w, _ = arr.shape
    inp = ncnn.Mat(np.ascontiguousarray(arr.transpose(2, 0, 1)).astype(np.float32))
    ex = net.create_extractor()
    ex.input("data", inp)
    _, out = ex.extract("output")
    return np.array(out).transpose(1, 2, 0)


def upscale(img):
    net = make_net()
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    h, w, _ = a.shape
    out = np.zeros((h * S, w * S, 3), np.float32)
    run_tile(net, a[:min(h, 64), :min(w, 64)])  # warm-up: first extraction on a fresh net is garbage
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
            py0, px0 = max(y0 - PAD, 0), max(x0 - PAD, 0)
            py1, px1 = min(y1 + PAD, h), min(x1 + PAD, w)
            r = run_tile(net, a[py0:py1, px0:px1])
            oy, ox = (y0 - py0) * S, (x0 - px0) * S
            out[y0 * S:y1 * S, x0 * S:x1 * S] = r[oy:oy + (y1 - y0) * S, ox:ox + (x1 - x0) * S]
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))


if __name__ == "__main__":
    src, box, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    im = Image.open(src).convert("RGB")
    if box != "full":
        im = im.crop(tuple(int(v) for v in box.split(",")))
    upscale(im).save(dst)
    print("saved", dst)
