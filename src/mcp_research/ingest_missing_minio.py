import argparse
import sys
from typing import List, Tuple

from mcp_research.resolver_app import _enqueue_missing_ingests, _find_missing_minio_objects


def _print_errors(errors: List[str]) -> None:
    for error in errors:
        print(f"error: {error}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enqueue Celery ingest tasks for MinIO files missing from Redis.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List missing files without enqueuing Celery tasks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of tasks to enqueue (0 means no limit).",
    )
    args = parser.parse_args()

    missing, errors = _find_missing_minio_objects()
    if errors:
        _print_errors(errors)
    if args.limit and args.limit > 0:
        missing = missing[: args.limit]
    if args.dry_run:
        for bucket, object_name in missing:
            print(f"{bucket}/{object_name}")
        print(f"missing_count={len(missing)}")
        return
    count = _enqueue_missing_ingests(missing)
    print(f"enqueued_count={count}")


if __name__ == "__main__":
    main()
