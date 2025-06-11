from agentic_doc.parse import parse

# Parse a local file
result = parse(["/home/viyer/Documents/Infosys/landing_ai/Insurancehouse.pdf"],grounding_save_dir="./grounding")
parsed_doc = result[0]

# Get the extracted data as markdown
(parsed_doc.markdown)

# Get the extracted data as structured chunks of content in a JSON schema
(parsed_doc.chunks)