# agentic_doc


# PDF Document Parsing & Visualization Pipeline

A lightweight, end-to-end Python toolkit for turning PDF documents into structured data and annotated visuals. Convert pages to images, detect layout regions, OCR text blocks, extract tables, and overlay bounding boxes — all under your control, without relying on paid APIs.

---

## 🚀 Features

- **PDF → Images**  
  Rasterize PDF pages at configurable DPI using `pdf2image` + Poppler.
- **Layout Detection**  
  Identify Text, Titles, Lists, Tables, and Figures with LayoutParser + Detectron2 (PubLayNet model).
- **OCR**  
  Run Tesseract on each detected region to extract text content.
- **Table Extraction**  
  Pull tabular data into pandas DataFrames via Camelot or Tabula-py.
- **Visualization**  
  Draw region bounding boxes and labels onto page images for quick inspection.
- **Modular & Extensible**  
  Swap out detectors, customize OCR config, or integrate other table-parsing backends.

---

## 📦 Requirements

- Python 3.8+  
- Poppler utilities (for `pdf2image`)  
- [pdf2image](https://github.com/Belval/pdf2image)  
- [layoutparser](https://layout-parser.readthedocs.io/) + Detectron2  
- [pytesseract](https://github.com/madmaze/pytesseract) + Tesseract-OCR  
- [camelot-py](https://camelot-py.readthedocs.io/) (optional, for tables)  
- (Optional) `opencv-python`, `pandas`, `Pillow`

Install via pip:

```bash
pip install pdf2image layoutparser[detectron2] pytesseract camelot-py[cv] opencv-python pandas pillow
