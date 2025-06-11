from agentic_doc.parse import parse
from agentic_doc.utils import viz_parsed_document

# Define the document path and output directory
doc_path = "/home/viyer/Documents/Infosys/landing_ai/Insurancehouse.pdf"
output_dir = "/home/viyer/Documents/Infosys/landing_ai/visualizations"

# Parse the document
result = parse(doc_path)
parsed_doc = result[0]

# Create visualizations with default settings
images = viz_parsed_document(
    doc_path,
    parsed_doc,
    output_dir=output_dir
)