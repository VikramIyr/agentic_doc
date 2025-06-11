#!/usr/bin/env python3
import os
import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.cluster import AgglomerativeClustering  # ← swapped in

# ─── CONFIG ───────────────────────────────────────────────────────────

DATA_DIR       = "/home/viyer/Documents/Infosys/landing_ai/agentic_doc/data"
INPUT_PNG      = os.path.join(DATA_DIR, "test_invoice.png")
OUTPUT_PNG     = os.path.join(DATA_DIR, "test_invoice_ocr_clustered.png")

CONF_THRESHOLD = 50
TESS_CONFIG    = r"--oem 3 --psm 6"

# font setup
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
FONT_SIZE = 14

# Hierarchical clustering parameters
CLUSTER_DIST        = 300   # max allowed distance *within* a cluster
CLUSTER_LINKAGE     = "complete"  # "single", "average" or "complete"
                                         # complete = cluster diameter ≤ CLUSTER_DIST
# ─── END CONFIG ────────────────────────────────────────────────────────

# 1) load font
font = None
for p in FONT_PATHS:
    if os.path.exists(p):
        font = ImageFont.truetype(p, FONT_SIZE)
        break
if font is None:
    font = ImageFont.load_default()

# 2) load image
bgr = cv2.imread(INPUT_PNG)
if bgr is None:
    raise FileNotFoundError(INPUT_PNG)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# 3) OCR → get word boxes
ocr = pytesseract.image_to_data(
    rgb, output_type=pytesseract.Output.DICT, config=TESS_CONFIG
)

# 4) collect boxes + centroids
boxes, centroids = [], []
for i, txt in enumerate(ocr["text"]):
    txt = txt.strip()
    conf = int(ocr["conf"][i])
    if not txt or conf < CONF_THRESHOLD:
        continue
    x, y, w, h = (
        ocr["left"][i],
        ocr["top"][i],
        ocr["width"][i],
        ocr["height"][i],
    )
    boxes.append((x, y, w, h))
    centroids.append([x + w/2, y + h/2])
centroids = np.array(centroids)

# 5) Agglomerative clustering (no preset n_clusters, just distance threshold)
clusterer = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=CLUSTER_DIST,
    linkage=CLUSTER_LINKAGE
)
labels = clusterer.fit_predict(centroids)

# 6) merge boxes per cluster
cluster_rects = {}
for (x, y, w, h), lbl in zip(boxes, labels):
    x1, y1, x2, y2 = x, y, x + w, y + h
    if lbl not in cluster_rects:
        cluster_rects[lbl] = [x1, y1, x2, y2]
    else:
        rx1, ry1, rx2, ry2 = cluster_rects[lbl]
        cluster_rects[lbl] = [
            min(rx1, x1), min(ry1, y1),
            max(rx2, x2), max(ry2, y2),
        ]

# 7) draw
pil_img = Image.fromarray(rgb)
draw    = ImageDraw.Draw(pil_img)

# lightly show original words (optional)
for x, y, w, h in boxes:
    draw.rectangle([x, y, x + w, y + h], outline=(255,200,200), width=1)

# draw merged clusters
for lbl, (x1, y1, x2, y2) in cluster_rects.items():
    draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
    draw.text((x1, y1 - FONT_SIZE - 2), f"Cluster {lbl}", fill="blue", font=font)

# 8) save
pil_img.save(OUTPUT_PNG)
print(f"✅ Cluster‐annotated image saved to {OUTPUT_PNG}")
