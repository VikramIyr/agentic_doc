# extract.py

import sys
from orchestrator import AgenticOrchestrator

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract.py <document_path> [manifest.yaml]")
        sys.exit(1)

    doc_path = sys.argv[1]
    manifest = sys.argv[2] if len(sys.argv) > 2 else "default_manifest.yaml"
    orch     = AgenticOrchestrator(manifest)
    result   = orch.extract(doc_path)
    print(result)

if __name__ == "__main__":
    main()
