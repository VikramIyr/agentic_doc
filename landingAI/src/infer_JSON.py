from agentic_doc.parse import parse
import os
import json

# Parse the document (no need to pass result_save_dir)


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "data", "test2", "input", "test_invoice2.pdf")
output_path = os.path.join(base_dir, "data", "test2", "output", "JSON_result.json")

result = parse(data_dir)
parsed = result[0]

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(parsed.model_dump(), f, ensure_ascii=False, indent=2)


# Get the extracted data as markdown
result[0].markdown

# Get the extracted data as structured chunks of content in a JSON schema
result[0].chunks