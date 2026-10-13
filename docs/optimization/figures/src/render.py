"""Render the SVG figures to PNG with headless Chrome / Edge (fonts are baked in), then palette-compress with Pillow.
usage: python render.py fig1.svg [fig2.svg ...] [--scale 2]"""
import os
import re
import subprocess
import sys
import tempfile

from PIL import Image

BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", "google-chrome", "chromium"]


def render(svg, scale=2.0, colors=255):
    head = open(svg, encoding="utf-8").read(400)
    w, h = (int(float(v)) for v in re.search(r'width="([\d.]+)" height="([\d.]+)"', head).groups())
    png = os.path.splitext(svg)[0] + ".png"
    exe = next(b for b in BROWSERS if os.path.sep not in b or os.path.exists(b))
    prof = os.path.join(tempfile.gettempdir(), "sketch_render_profile")
    url = "file:///" + os.path.abspath(svg).replace("\\", "/")
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", f"--user-data-dir={prof}",
                    f"--force-device-scale-factor={scale}", f"--window-size={w},{h}", f"--screenshot={os.path.abspath(png)}",
                    url], check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    img = Image.open(png).convert("RGB")
    img = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    img.save(png, optimize=True)
    print(f"{png}: {img.size[0]}x{img.size[1]}, {os.path.getsize(png) / 1024:.0f} KB")
    return png


if __name__ == "__main__":
    argv, scale = sys.argv[1:], 2.0
    if "--scale" in argv:
        i = argv.index("--scale")
        scale = float(argv[i + 1])
        del argv[i:i + 2]
    for a in argv:
        render(a, scale)
