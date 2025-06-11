import os
from dotenv import load_dotenv

from agentic_doc.parse import parse
from agentic_doc.parse import parse_documents

from agentic_doc.utils import viz_parsed_document
from agentic_doc.config import VisualizationConfig
from agentic_doc.config import ChunkType


load_dotenv() # This loads the variables from .env into your environment

api_key = os.getenv("VISION_AGENT_API_KEY")

if api_key is None:
    print("Error: VISION_AGENT_API_KEY not found in .env file or environment variables.")
else:
    print("API Key loaded successfully!")
    # Your code that uses api_key



data_dir = "C:\\Espace_Eliott\\Infosys_code\\agentic_doc\\data\\test_invoice.pdf"
grnd_save_dir = "C:\\Espace_Eliott\\Infosys_code\\box_result"
doc_save_dir = "C:\\Espace_Eliott\\Infosys_code\\doc_result"

# Save groundings when parsing a document
results = parse_documents(
    [data_dir],
    grounding_save_dir= grnd_save_dir
)

# The grounding images will be saved to:
# path/to/save/groundings/document_TIMESTAMP/page_X/CHUNK_TYPE_CHUNK_ID_Y.png
# Where X is the page number, CHUNK_ID is the unique ID of each chunk,
# and Y is the index of the grounding within the chunk

# Each chunk's grounding in the result will have the image_path set
for chunk in results[0].chunks:
    for grounding in chunk.grounding:
        if grounding.image_path:
            print(f"Grounding saved to: {grounding.image_path}")



# Parse a document
results = parse(data_dir)
parsed_doc = results[0]

# Create visualizations with default settings
# The output images have a PIL.Image.Image type
images = viz_parsed_document(
    data_dir,
    parsed_doc,
    output_dir=doc_save_dir
)

# Or customize the visualization appearance
viz_config = VisualizationConfig(
    thickness=2,  # Thicker bounding boxes
    text_bg_opacity=0.8,  # More opaque text background
    font_scale=0.7,  # Larger text
    # Custom colors for different chunk types
    color_map={
        #ChunkType.title: (0, 0, 255),  # Red for titles
        ChunkType.text: (255, 0, 0),  # Blue for regular text
        ChunkType.marginalia: (128, 128, 128),  # Gray for marginal notes
        ChunkType.figure: (0, 255, 0),  
        ChunkType.table: (0, 0, 255),  
    }
)

images = viz_parsed_document(
    data_dir,
    parsed_doc,
    output_dir=doc_save_dir,
    viz_config=viz_config
)

# The visualization images will be saved as:
# path/to/save/visualizations/document_viz_page_X.png
# Where X is the page number