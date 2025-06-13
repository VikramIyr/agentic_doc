import os

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

# ─── CONFIGURE THESE ───────────────────────────────────────────────────────────
pdf_file_name   = "/home/viyer/Documents/Infosys/landing_ai/agentic_doc/minerU/data/test2/input/test_invoice.pdf"
local_image_dir = "/home/viyer/Documents/Infosys/landing_ai/agentic_doc/minerU/data/test2/output/images"
local_md_dir    = "/home/viyer/Documents/Infosys/landing_ai/agentic_doc/minerU/data/test2/output/md_and_json"
# ───────────────────────────────────────────────────────────────────────────────

# ensure output dirs exist
os.makedirs(local_image_dir, exist_ok=True)
os.makedirs(local_md_dir, exist_ok=True)

# prepare writers and reader
image_writer = FileBasedDataWriter(local_image_dir)
md_writer    = FileBasedDataWriter(local_md_dir)
reader       = FileBasedDataReader("")
pdf_bytes    = reader.read(pdf_file_name)

# run MinerU
ds = PymuDocDataset(pdf_bytes)
if ds.classify() == SupportedPdfParseMethod.OCR:
    infer_result = ds.apply(doc_analyze, ocr=True)
    pipe_result  = infer_result.pipe_ocr_mode(image_writer)
else:
    infer_result = ds.apply(doc_analyze, ocr=False)
    pipe_result  = infer_result.pipe_txt_mode(image_writer)

# draw the model’s raw predictions
infer_result.draw_model(os.path.join(local_md_dir, f"{os.path.splitext(os.path.basename(pdf_file_name))[0]}_model.pdf"))

# dump layout- and span-annotated PDFs
pipe_result.draw_layout(os.path.join(local_md_dir, f"{os.path.splitext(os.path.basename(pdf_file_name))[0]}_layout.pdf"))
pipe_result.draw_span(os.path.join(local_md_dir, f"{os.path.splitext(os.path.basename(pdf_file_name))[0]}_spans.pdf"))

# write out the Markdown and auxiliary JSON
image_dir_name = os.path.basename(local_image_dir)
pipe_result.dump_md(md_writer, f"{os.path.splitext(os.path.basename(pdf_file_name))[0]}.md", image_dir_name)
pipe_result.dump_content_list(md_writer, f"{os.path.splitext(os.path.basename(pdf_file_name))[0]}_content_list.json", image_dir_name)
pipe_result.dump_middle_json(md_writer, f"{os.path.splitext(os.path.basename(pdf_file_name))[0]}_middle.json")
