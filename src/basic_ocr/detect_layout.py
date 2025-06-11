#!/usr/bin/env python3
import os
import json
from PIL import Image, ImageDraw, ImageFont

# ─── CONFIG ───────────────────────────────────────────────────────────
DATA_DIR      = "/home/viyer/Documents/Infosys/landing_ai/agentic_doc/data"
IMAGE_DIR     = os.path.join(DATA_DIR, "output_images")
STRUCT_JSON   = os.path.join(DATA_DIR, "output_structure.json")
OUT_VIZ_DIR   = os.path.join(DATA_DIR, "visualizations")

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_SIZE = 12
# ─── /CONFIG ──────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_VIZ_DIR, exist_ok=True)
    # Load JSON
    with open(STRUCT_JSON, "r", encoding="utf-8") as f:
        doc = json.load(f)
    chunks = doc.get("chunks", [])

    # Prepare font
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()

    # Group chunks by page
    pages = {}
    for c in chunks:
        pages.setdefault(c["page"], []).append(c)

    # For each page, overlay boxes
    for page, chs in pages.items():
        img_path = os.path.join(IMAGE_DIR, f"page_{page}.png")
        if not os.path.exists(img_path):
            print(f"[!] Missing image for page {page}: {img_path}")
            continue

        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)

        print(f"\nPage {page}:")
        for idx, c in enumerate(chs, start=1):
            x1,y1,x2,y2 = c["bbox"]
            # draw box
            draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
            label = c["type"]
            draw.text((x1, y1 - FONT_SIZE - 2), f"{idx}. {label}", fill="red", font=font)

            # print a summary line
            summary = ""
            if "text" in c:
                summary = c["text"].replace("\n", " ")[:80]
            elif "tables" in c:
                summary = f"[{len(c['tables'])} table(s)]"
            print(f"  {idx:02d}. {label:<6} bbox={c['bbox']} → {summary}")

        # save visualization
        out_path = os.path.join(OUT_VIZ_DIR, f"page_{page}_viz.png")
        img.save(out_path)
        print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
