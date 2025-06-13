import os
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

from agentic_doc.parse import parse, parse_documents
from agentic_doc.utils import viz_parsed_document
from agentic_doc.config import VisualizationConfig, ChunkType


def load_api_key():
    load_dotenv()
    key = os.getenv("VISION_AGENT_API_KEY")
    if not key:
        raise EnvironmentError(
            "Error: VISION_AGENT_API_KEY not found in .env file or environment variables."
        )
    print("API Key loaded successfully!")
    return key


def parse_and_save_groundings(input_file: Path, grounding_dir: Path):
    grounding_dir.mkdir(parents=True, exist_ok=True)
    results = parse_documents([str(input_file)], grounding_save_dir=str(grounding_dir))
    for chunk in results[0].chunks:
        for grounding in chunk.grounding:
            if grounding.image_path:
                print(f"Grounding saved to: {grounding.image_path}")
    return results


def parse_and_visualize(
    input_file: Path,
    viz_dir: Path,
    use_custom_config: bool = False
):
    viz_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse(str(input_file))[0]
    # Default visualization
    viz_parsed_document(str(input_file), parsed, output_dir=str(viz_dir))
    print(f"Default visualizations saved to: {viz_dir}")

    if use_custom_config:
        config = VisualizationConfig(
            thickness=2,
            text_bg_opacity=0.8,
            font_scale=0.7,
            color_map={
                ChunkType.text: (255, 0, 0),
                ChunkType.marginalia: (128, 128, 128),
                ChunkType.figure: (0, 255, 0),
                ChunkType.table: (0, 0, 255),
            },
        )
        viz_parsed_document(
            str(input_file),
            parsed,
            output_dir=str(viz_dir),
            viz_config=config,
        )
        print(f"Custom visualizations saved to: {viz_dir}")
    return parsed


def save_json(parsed, json_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"Parsed JSON saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic AI Document Parsing Demo"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the input document (PDF or image)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base directory for all output data (groundings, visualizations, JSON)."
    )
    parser.add_argument(
        "--custom-viz",
        action="store_true",
        help="Enable custom visualization settings."
    )
    args = parser.parse_args()

    # Load API key
    load_api_key()

    # Define output subdirectories
    grounding_dir = args.data_dir / "groundings"
    viz_dir = args.data_dir / "visualizations"
    json_output = args.data_dir / "parsed_output.json"

    # Parse and save groundings
    parse_and_save_groundings(args.input, grounding_dir)

    # Parse and visualize the document
    parsed_doc = parse_and_visualize(
        args.input,
        viz_dir,
        use_custom_config=args.custom_viz,
    )

    # Save JSON dump
    save_json(parsed_doc, json_output)

    # Print markdown and chunks info
    print("\n--- Extracted Markdown Content ---")
    print(parsed_doc.markdown)

    print("\n--- Parsed Chunks ---")
    for chunk in parsed_doc.chunks:
        print(chunk)

if __name__ == "__main__":
    main()