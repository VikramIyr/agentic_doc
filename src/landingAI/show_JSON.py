from agentic_doc.parse import parse



import json

# Parse the document (no need to pass result_save_dir)
result = parse("C:\\Espace_Eliott\\Cours\\testIA2.pdf")
parsed = result[0]

# Save JSON yourself, with correct encoding
output_path = "C:\\Espace_Eliott\\Infosys_code\\JSON_res.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(parsed.model_dump(), f, ensure_ascii=False, indent=2)



# Get the extracted data as markdown
(result[0].markdown)

# Get the extracted data as structured chunks of content in a JSON schema
(result[0].chunks)  