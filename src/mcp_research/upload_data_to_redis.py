import argparse
import logging
import os
from pathlib import Path

from mcp_research.ingest_unstructured import (
    _get_redis_client,
    upload_json_files_to_redis,
)
from mcp_research.runtime_utils import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _upload_pair(
    redis_client,
    partitions_path: Path,
    chunks_path: Path,
    redis_prefix: str,
    doc_id: str | None,
    source: str | None,
) -> None:
    """Upload one partition+chunk JSON pair into Redis."""
    upload_json_files_to_redis(
        redis_client=redis_client,
        partitions_path=partitions_path,
        chunks_path=chunks_path,
        doc_id=doc_id,
        source=source,
        prefix=redis_prefix,
    )


def _upload_directory(
    redis_client,
    partitions_dir: Path,
    chunks_dir: Path,
    redis_prefix: str,
    strict: bool,
) -> int:
    """Upload all matching partition/chunk JSON pairs from directories."""
    total = 0
    for chunk_path in sorted(chunks_dir.glob("*.json")):
        partitions_path = partitions_dir / f"{chunk_path.stem}.json"
        if not partitions_path.is_file():
            msg = f"Missing partitions for {chunk_path.name}"
            if strict:
                raise FileNotFoundError(msg)
            logger.warning(msg)
            continue
        _upload_pair(
            redis_client=redis_client,
            partitions_path=partitions_path,
            chunks_path=chunk_path,
            redis_prefix=redis_prefix,
            doc_id=None,
            source=None,
        )
        total += 1
    return total


def main() -> None:
    """CLI entry point for uploading JSON payloads into Redis."""
    load_dotenv(Path(".env"))

    parser = argparse.ArgumentParser(
        description="Upload partition + chunk JSON payloads into Redis.",
    )
    parser.add_argument(
        "--partitions-dir",
        default=os.getenv("PARTITIONS_DIR", "data/partitions"),
        help="Directory containing partition JSON files",
    )
    parser.add_argument(
        "--chunks-dir",
        default=os.getenv("CHUNKS_DIR", "data/chunks"),
        help="Directory containing chunk JSON files",
    )
    parser.add_argument(
        "--partitions-file",
        help="Single partition JSON file (requires --chunks-file)",
    )
    parser.add_argument(
        "--chunks-file",
        help="Single chunk JSON file (requires --partitions-file)",
    )
    parser.add_argument(
        "--doc-id",
        help="Override document_id (single-file mode only)",
    )
    parser.add_argument(
        "--source",
        help="Override source (single-file mode only)",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", ""),
        help="Redis URL (required)",
    )
    parser.add_argument(
        "--redis-prefix",
        default=os.getenv("REDIS_PREFIX", "unstructured"),
        help="Redis key prefix for payloads",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a chunk file is missing its partition pair",
    )
    args = parser.parse_args()

    redis_client = _get_redis_client(args.redis_url)
    if not redis_client:
        raise RuntimeError("REDIS_URL is required and redis must be installed")

    if args.partitions_file or args.chunks_file:
        if not (args.partitions_file and args.chunks_file):
            raise ValueError("Both --partitions-file and --chunks-file are required")
        _upload_pair(
            redis_client=redis_client,
            partitions_path=Path(args.partitions_file).expanduser(),
            chunks_path=Path(args.chunks_file).expanduser(),
            redis_prefix=args.redis_prefix,
            doc_id=args.doc_id,
            source=args.source,
        )
        logger.info("Uploaded %s", args.chunks_file)
        return

    partitions_dir = Path(args.partitions_dir).expanduser()
    chunks_dir = Path(args.chunks_dir).expanduser()
    if not partitions_dir.exists():
        raise FileNotFoundError(f"Partitions directory not found: {partitions_dir}")
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    total = _upload_directory(
        redis_client=redis_client,
        partitions_dir=partitions_dir,
        chunks_dir=chunks_dir,
        redis_prefix=args.redis_prefix,
        strict=args.strict,
    )
    logger.info("Uploaded %d documents to Redis", total)


if __name__ == "__main__":
    main()
