#!/usr/bin/env python3
"""Convert SVG files to PDF. Defaults to all SVGs in docs/figures/."""

import sys
from pathlib import Path
import cairosvg

def convert(svg_path: str):
    pdf_path = str(Path(svg_path).with_suffix(".pdf"))
    cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
    print(f"{svg_path} → {pdf_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            convert(f)
    else:
        svgs = sorted(Path("docs/figures").glob("*.svg"))
        if not svgs:
            print("No SVGs found in docs/figures/")
        for svg in svgs:
            convert(str(svg))
