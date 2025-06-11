from pdf2image import convert_from_path
import os

data="/home/viyer/Documents/Infosys/landing_ai/agentic_doc/data"
pdf_file_in_data = os.path.join(data, "test_invoice.pdf")

pages = convert_from_path(pdf_file_in_data, dpi=150)
for i, pil_img in enumerate(pages):
    pil_img.save(f"page_{i:03d}.png")
