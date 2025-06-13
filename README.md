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

## 💡 Examples

### 1. MinerU: High-Fidelity Equation & Structure Extraction

```python
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
import os

# — CONFIGURE —
pdf_path       = "invoices/test_invoice.pdf"
out_images_dir = "mineru_output/images"
out_md_dir     = "mineru_output/md"
os.makedirs(out_images_dir, exist_ok=True)
os.makedirs(out_md_dir, exist_ok=True)

# prepare
reader = FileBasedDataReader("")
bytes  = reader.read(pdf_path)
ds     = PymuDocDataset(bytes)
img_w  = FileBasedDataWriter(out_images_dir)
md_w   = FileBasedDataWriter(out_md_dir)

# choose OCR or text mode
if ds.classify() == SupportedPdfParseMethod.OCR:
    res = ds.apply(doc_analyze, ocr=True)
    pipe = res.pipe_ocr_mode(img_w)
else:
    res = ds.apply(doc_analyze, ocr=False)
    pipe = res.pipe_txt_mode(img_w)

# visualize & dump
res.draw_model(f"{out_md_dir}/model.pdf")
pipe.draw_layout(f"{out_md_dir}/layout.pdf")
pipe.draw_span(f"{out_md_dir}/spans.pdf")
pipe.dump_md(md_w, "test_invoice.md", os.path.basename(out_images_dir))
pipe.dump_middle_json(md_w, "test_invoice_middle.json")
