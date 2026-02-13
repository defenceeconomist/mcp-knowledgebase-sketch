from __future__ import annotations

import argparse
import logging
import mimetypes
import os
from pathlib import Path
from typing import Iterable, Sequence

from minio import Minio
from minio.error import S3Error

from mcp_research.minio_ingest import delete_object_from_env, process_object_from_env
from mcp_research.runtime_utils import load_dotenv, load_env_bool


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_connection_args(args: argparse.Namespace) -> tuple[str, str, str, bool]:
    endpoint = (args.endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")).strip()
    access_key = (
        args.access_key or os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER") or ""
    ).strip()
    secret_key = (
        args.secret_key or os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD") or ""
    ).strip()
    secure = load_env_bool("MINIO_SECURE", False) if args.secure is None else bool(args.secure)

    if not access_key or not secret_key:
        raise RuntimeError("MINIO_ACCESS_KEY/MINIO_SECRET_KEY are required")
    return endpoint, access_key, secret_key, secure


def _build_minio_client(endpoint: str, access_key: str, secret_key: str, secure: bool) -> Minio:
    return Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )


def _apply_connection_env(endpoint: str, access_key: str, secret_key: str, secure: bool) -> None:
    """Keep env in sync so ingest helpers connect to the same MinIO target."""
    os.environ["MINIO_ENDPOINT"] = endpoint
    os.environ["MINIO_ACCESS_KEY"] = access_key
    os.environ["MINIO_SECRET_KEY"] = secret_key
    os.environ["MINIO_SECURE"] = "1" if secure else "0"


def _iter_bucket_objects(client: Minio, bucket: str) -> Iterable:
    try:
        return client.list_objects(bucket, recursive=True, include_version=True)
    except TypeError:
        return client.list_objects(bucket, recursive=True)


def _add_bucket(client: Minio, bucket: str) -> None:
    if client.bucket_exists(bucket):
        logger.info("Bucket already exists: %s", bucket)
        return
    client.make_bucket(bucket)
    logger.info("Created bucket: %s", bucket)


def _delete_bucket(client: Minio, bucket: str, force: bool) -> None:
    if not client.bucket_exists(bucket):
        raise RuntimeError(f"Bucket does not exist: {bucket}")
    if force:
        removed = 0
        for entry in _iter_bucket_objects(client, bucket):
            version_id = getattr(entry, "version_id", None)
            client.remove_object(bucket, entry.object_name, version_id=version_id)
            removed += 1
        logger.info("Removed %d object(s) from %s", removed, bucket)
    client.remove_bucket(bucket)
    logger.info("Deleted bucket: %s", bucket)


def _upload_file(
    client: Minio,
    bucket: str,
    file_path: Path,
    object_name: str | None,
    create_bucket: bool,
    content_type: str | None,
    ingest: bool,
    connection_args: tuple[str, str, str, bool],
) -> None:
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not client.bucket_exists(bucket):
        if not create_bucket:
            raise RuntimeError(f"Bucket does not exist: {bucket} (use --create-bucket)")
        _add_bucket(client, bucket)

    target_name = object_name or file_path.name
    detected_content_type = content_type
    if not detected_content_type:
        detected_content_type = mimetypes.guess_type(file_path.name)[0]

    result = client.fput_object(
        bucket_name=bucket,
        object_name=target_name,
        file_path=str(file_path),
        content_type=detected_content_type,
    )
    version_id = getattr(result, "version_id", None)
    logger.info("Uploaded %s -> s3://%s/%s", file_path, bucket, target_name)

    if ingest:
        endpoint, access_key, secret_key, secure = connection_args
        _apply_connection_env(endpoint, access_key, secret_key, secure)
        process_object_from_env(bucket=bucket, object_name=target_name, version_id=version_id)
        logger.info("Ingested s3://%s/%s via Unstructured", bucket, target_name)


def _remove_file(
    client: Minio,
    bucket: str,
    object_name: str,
    version_id: str | None,
    delete_ingested: bool,
    connection_args: tuple[str, str, str, bool],
) -> None:
    client.remove_object(bucket, object_name, version_id=version_id)
    logger.info("Removed s3://%s/%s", bucket, object_name)
    if delete_ingested:
        endpoint, access_key, secret_key, secure = connection_args
        _apply_connection_env(endpoint, access_key, secret_key, secure)
        delete_object_from_env(bucket=bucket, object_name=object_name, version_id=version_id)
        logger.info("Removed ingested vectors/metadata for s3://%s/%s", bucket, object_name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage MinIO buckets/files and trigger ingest cleanup.",
    )
    parser.add_argument("--endpoint", default=None, help="MinIO endpoint (host:port).")
    parser.add_argument("--access-key", default=None, help="MinIO access key.")
    parser.add_argument("--secret-key", default=None, help="MinIO secret key.")
    parser.add_argument(
        "--secure",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use HTTPS for MinIO connection (or --no-secure).",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    add_parser = subparsers.add_parser("add-bucket", help="Create a bucket.")
    add_parser.add_argument("bucket", help="Bucket name.")

    delete_parser = subparsers.add_parser("delete-bucket", help="Delete a bucket.")
    delete_parser.add_argument("bucket", help="Bucket name.")
    delete_parser.add_argument(
        "--force",
        action="store_true",
        help="Delete all objects in the bucket before deleting the bucket.",
    )

    upload_parser = subparsers.add_parser(
        "upload-file",
        help="Upload a file and optionally ingest it with Unstructured.",
    )
    upload_parser.add_argument("bucket", help="Target bucket.")
    upload_parser.add_argument("file", help="Local file path to upload.")
    upload_parser.add_argument(
        "--object-name",
        default=None,
        help="Target object key in MinIO (defaults to file basename).",
    )
    upload_parser.add_argument(
        "--create-bucket",
        action="store_true",
        help="Create the target bucket if it does not exist.",
    )
    upload_parser.add_argument("--content-type", default=None, help="Override MIME type.")
    upload_parser.add_argument(
        "--ingest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ingest uploaded file immediately with Unstructured (or --no-ingest).",
    )

    remove_parser = subparsers.add_parser("remove-file", help="Remove one object from a bucket.")
    remove_parser.add_argument("bucket", help="Bucket name.")
    remove_parser.add_argument("object_name", help="Object key in MinIO.")
    remove_parser.add_argument("--version-id", default=None, help="Object version id.")
    remove_parser.add_argument(
        "--delete-ingested",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete Qdrant/Redis records for the object (or --no-delete-ingested).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv(Path(".env"))
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    endpoint, access_key, secret_key, secure = _resolve_connection_args(args)
    client = _build_minio_client(endpoint, access_key, secret_key, secure)
    connection_args = (endpoint, access_key, secret_key, secure)

    try:
        if args.subcommand == "add-bucket":
            _add_bucket(client, args.bucket)
        elif args.subcommand == "delete-bucket":
            _delete_bucket(client, args.bucket, force=args.force)
        elif args.subcommand == "upload-file":
            _upload_file(
                client=client,
                bucket=args.bucket,
                file_path=Path(args.file).expanduser(),
                object_name=args.object_name,
                create_bucket=args.create_bucket,
                content_type=args.content_type,
                ingest=args.ingest,
                connection_args=connection_args,
            )
        elif args.subcommand == "remove-file":
            _remove_file(
                client=client,
                bucket=args.bucket,
                object_name=args.object_name,
                version_id=args.version_id,
                delete_ingested=args.delete_ingested,
                connection_args=connection_args,
            )
        else:  # pragma: no cover - argparse enforces valid choices
            raise ValueError(f"Unsupported subcommand: {args.subcommand}")
    except S3Error as exc:
        raise RuntimeError(f"MinIO operation failed: {exc}") from exc


if __name__ == "__main__":
    main()
