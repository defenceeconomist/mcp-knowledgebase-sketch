from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Sequence

COMMAND_MODULES: dict[str, str] = {
    "mcp-app": "mcp_research.mcp_app",
    "upsert-chunks": "mcp_research.upsert_chunks",
    "minio-ingest": "mcp_research.minio_ingest",
    "minio-ops": "mcp_research.minio_ops",
    "ingest-missing-minio": "mcp_research.ingest_missing_minio",
    "upload-data-to-redis": "mcp_research.upload_data_to_redis",
    "hybrid-search": "mcp_research.hybrid_search",
    "dedupe-qdrant-chunks": "mcp_research.dedupe_qdrant_chunks",
    "purge-v1-schema": "mcp_research.purge_v1_schema",
    "bibtex-autofill": "mcp_research.bibtex_autofill",
}


def _ensure_src_on_path() -> None:
    project_root = os.path.abspath(os.path.dirname(__file__))
    src_root = os.path.join(project_root, "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


def _run_module_main(module_name: str, argv: Sequence[str]) -> int:
    _ensure_src_on_path()
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if not callable(main):
        raise RuntimeError(f"Module '{module_name}' does not expose callable main()")

    original_argv = sys.argv
    sys.argv = [module_name, *list(argv)]
    try:
        main()
    finally:
        sys.argv = original_argv
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified CLI for mcp-research commands.",
    )
    parser.add_argument(
        "command",
        choices=tuple(COMMAND_MODULES.keys()),
        help="Command to run.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected command.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    ns = parser.parse_args(argv)
    return _run_module_main(COMMAND_MODULES[ns.command], ns.args)


if __name__ == "__main__":
    raise SystemExit(main())
