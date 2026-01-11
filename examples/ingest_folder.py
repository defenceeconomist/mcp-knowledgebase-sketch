import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import ingest_unstructured


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest all PDFs from a folder using Unstructured API.",
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=os.getenv("DATA_DIR", "data-raw"),
        help="Folder containing PDFs (default: DATA_DIR or data-raw)",
    )
    args = parser.parse_args()

    results = ingest_unstructured.run_from_env(data_dir_override=args.folder)
    print("Ingestion summary:")
    for item in results:
        print(
            f"- {item['file']} chunks={item['chunks']} "
            f"partition={item['partition_path']} chunk={item['chunk_path']}"
        )


if __name__ == "__main__":
    main()
