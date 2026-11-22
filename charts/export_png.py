"""Export each chart as a PNG for Substack, using headless Chrome, then trim the empty space below it.

Usage: python3 charts/export_png.py   (needs Google Chrome and Pillow)
"""
import os, subprocess
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
names = {1: 'chart-1-facts', 2: 'chart-2-nonsense', 3: 'chart-3-wrong-answers', 4: 'chart-4-documents', 5: 'chart-5-top-models'}
os.makedirs(os.path.join(here, 'png'), exist_ok=True)
for i, name in names.items():
    out = os.path.join(here, 'png', name + '.png')
    subprocess.run([chrome, '--headless=new', '--disable-gpu', '--hide-scrollbars', '--force-device-scale-factor=2',
                    '--window-size=1200,1400', '--virtual-time-budget=6000', f'--screenshot={out}',
                    f'file://{here}/charts.html?only={i}'], check=True, capture_output=True)
    im = Image.open(out).convert('RGB')
    bg = im.getpixel((2, im.height - 2))
    px = im.load()
    bottom = im.height - 1
    while bottom > 0 and all(px[x, bottom] == bg for x in range(0, im.width, 7)):
        bottom -= 1
    im.crop((0, 0, im.width, min(im.height, bottom + 64))).save(out)
    print(name, im.width, 'x', min(im.height, bottom + 64))
