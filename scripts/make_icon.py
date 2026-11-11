"""Genera assets/tessitura.ico (fumetto + forma d'onda) usato dalla finestra e dal collegamento.

Rigenerare con:  python scripts/make_icon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

S = 256
BG = (16, 23, 34)       # blu
FG = (255, 255, 255)
WAVE = (147, 197, 253)   # azzurro chiaro


def draw(size: int = S) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    k = size / 256

    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(56 * k), fill=BG)

    # A speech bubble with a five-bar waveform: conversation + transcription.
    d.rounded_rectangle([int(40*k), int(45*k), int(216*k), int(190*k)],
                        radius=int(42*k), fill=(94, 224, 202))
    d.polygon([(int(70*k), int(174*k)), (int(70*k), int(218*k)),
               (int(117*k), int(182*k))], fill=(94, 224, 202))
    for x, height in ((72, 32), (97, 65), (122, 94), (147, 55), (172, 28)):
        d.rounded_rectangle([int(x*k), int((117-height/2)*k),
                             int((x+12)*k), int((117+height/2)*k)],
                            radius=max(1, int(6*k)), fill=BG)
    return img


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "assets" / "tessitura.ico"
    out.parent.mkdir(parents=True, exist_ok=True)
    base = draw(1024).resize((256, 256), Image.LANCZOS)
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    base.save(out, format="ICO", sizes=sizes)
    base.resize((128, 128), Image.LANCZOS).save(out.with_suffix(".png"))
    print(f"icona scritta: {out} ({out.stat().st_size} byte)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
