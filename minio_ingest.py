import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research.minio_ingest import main


if __name__ == "__main__":
    main()
