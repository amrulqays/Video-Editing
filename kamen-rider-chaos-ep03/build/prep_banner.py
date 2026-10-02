"""Source prep for the 21:9 key art: upscale the two close-ups, cut out every element.

Run from this directory after fetching models/ and fonts/ (see README.md).
"""
import os
import subprocess
import sys
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "input")
W = os.path.join(HERE, "work", "banner")


def run(*args):
    subprocess.run([sys.executable, *args], cwd=HERE, check=True)


def p(*a):
    return os.path.join(*a)


def main():
    os.makedirs(W, exist_ok=True)
    # Real-ESRGAN x4 on the two giant close-ups
    for src, box, out in [(p(IN, "episode1-poster.webp"), "0,0,760,941", "sr_chaos.png"),     # Chaos helmet
                          (p(IN, "umbrazor-closeup.webp"), "380,0,1800,821", "sr_umb.png")]:  # Umbrazor
        dst = p(W, out)
        if not os.path.exists(dst):
            run("sr.py", src, box, dst)
        run("repair.py", src, box, dst)

    Image.open(p(IN, "logo.webp")).convert("RGB").crop((340, 40, 1560, 720)).save(p(W, "logo_src.png"))
    Image.open(p(IN, "episode1-poster.webp")).convert("RGB").crop((0, 0, 760, 941)).save(p(W, "chaos_src.png"))
    um = Image.open(p(W, "sr_umb.png"))
    um.resize((um.width // 2, um.height // 2), Image.LANCZOS).save(p(W, "umb_half.png"))

    # BiRefNet cut-outs, one process each (they need the memory to themselves)
    for src, out in [(p(W, "logo_src.png"), "m_logo.png"),
                     (p(IN, "tunnel.webp"), "m_tunnel.png"),
                     (p(W, "chaos_src.png"), "m_chaos.png"),
                     (p(W, "umb_half.png"), "m_umb.png")]:
        if not os.path.exists(p(W, out)):
            run("seg1.py", src, p(W, out))
    print("banner prep done")


if __name__ == "__main__":
    main()
