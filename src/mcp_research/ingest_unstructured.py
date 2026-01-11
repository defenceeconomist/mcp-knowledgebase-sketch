import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Sequence, Tuple

from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared
from unstructured_client.models.errors import SDKError

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_dotenv(path: Path) -> None:
    """Load a simple KEY=VALUE .env file into the process environment."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%s, using %d", key, raw, default)
        return default


def load_env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def collect_pdfs(target: Path) -> List[Path]:
    """Resolve a single PDF or all PDFs in a directory."""
    if target.is_file():
        if target.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {target}")
        return [target]
    if target.is_dir():
        return sorted(target.glob("*.pdf"))
    raise FileNotFoundError(f"PDF path not found: {target}")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get_redis_client(redis_url: str):
    if not redis_url:
        return None
    if redis is None:
        logger.warning("REDIS_URL set but redis package is missing; skipping Redis storage")
        return None
    return redis.from_url(redis_url)


def _redis_key(prefix: str, doc_id: str, suffix: str) -> str:
    return f"{prefix}:pdf:{doc_id}:{suffix}"


def _collections_key(prefix: str, doc_id: str) -> str:
    return f"{prefix}:pdf:{doc_id}:collections"


def record_collection_mapping(
    redis_client,
    doc_id: str,
    collection: str,
    prefix: str = "unstructured",
) -> str | None:
    if not redis_client or not collection:
        return None
    collections_key = _collections_key(prefix, doc_id)
    redis_client.sadd(collections_key, collection)
    redis_client.hset(
        _redis_key(prefix, doc_id, "meta"),
        mapping={"collections_key": collections_key},
    )
    return collections_key


def upload_to_redis(
    redis_client,
    doc_id: str,
    source: str,
    partitions_payload: list,
    chunks_payload: list,
    prefix: str = "unstructured",
    collection: str | None = None,
) -> dict:
    """Store partition + chunk payloads and metadata for a PDF in Redis."""
    meta_key = _redis_key(prefix, doc_id, "meta")
    partitions_key = _redis_key(prefix, doc_id, "partitions")
    chunks_key = _redis_key(prefix, doc_id, "chunks")
    collections_key = _collections_key(prefix, doc_id)
    if not redis_client:
        raise ValueError("redis_client is required to upload data to Redis")

    redis_client.set(partitions_key, json.dumps(partitions_payload, ensure_ascii=True))
    redis_client.set(chunks_key, json.dumps(chunks_payload, ensure_ascii=True))
    redis_client.hset(
        meta_key,
        mapping={
            "document_id": doc_id,
            "source": source,
            "chunks": str(len(chunks_payload)),
            "partitions_key": partitions_key,
            "chunks_key": chunks_key,
            "collections_key": collections_key,
        },
    )
    redis_client.sadd(f"{prefix}:pdf:hashes", doc_id)
    if collection:
        redis_client.sadd(collections_key, collection)
    return {
        "meta_key": meta_key,
        "partitions_key": partitions_key,
        "chunks_key": chunks_key,
        "collections_key": collections_key,
    }


def upload_json_files_to_redis(
    redis_client,
    partitions_path: Path,
    chunks_path: Path,
    doc_id: str | None = None,
    source: str | None = None,
    prefix: str = "unstructured",
    collection: str | None = None,
) -> dict:
    """Upload partition + chunk JSON files into Redis (doc_id required or inferred)."""
    if not partitions_path.is_file():
        raise FileNotFoundError(f"Partition JSON not found: {partitions_path}")
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks JSON not found: {chunks_path}")

    partitions_payload = json.loads(partitions_path.read_text(encoding="utf-8"))
    chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(chunks_payload, list):
        raise ValueError(f"Chunks JSON must be a list: {chunks_path}")

    if doc_id is None:
        if chunks_payload:
            doc_id = chunks_payload[0].get("document_id")
        if not doc_id:
            raise ValueError("doc_id is required or must exist in chunks JSON")

    if source is None:
        if chunks_payload:
            source = chunks_payload[0].get("source")
        if not source:
            source = chunks_path.stem

    return upload_to_redis(
        redis_client=redis_client,
        doc_id=doc_id,
        source=source,
        partitions_payload=partitions_payload,
        chunks_payload=chunks_payload,
        prefix=prefix,
        collection=collection,
    )


def partition_pdf(
    client: UnstructuredClient,
    pdf_path: Path,
    file_bytes: bytes | None,
    strategy: str,
    chunking_strategy: str | None,
    max_characters: int | None,
    overlap: int | None,
    languages: List[str] | None,
):
    """Call the Unstructured API to partition (and optionally chunk) a PDF."""
    if file_bytes is None:
        try:
            file_bytes = pdf_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"Failed to read {pdf_path}: {exc}") from exc

    files = shared.Files(content=file_bytes, file_name=pdf_path.name)
    params = shared.PartitionParameters(
        files=files,
        strategy=strategy,
        chunking_strategy=chunking_strategy,
        max_characters=max_characters,
        overlap=overlap,
        languages=languages,
        include_page_breaks=True,
    )

    try:
        # SDK expects an operations.PartitionRequest wrapper around PartitionParameters.
        # If an older SDK requires positional initialization, fall back accordingly.
        try:
            request_obj = operations.PartitionRequest(partition_parameters=params)
        except TypeError:
            request_obj = operations.PartitionRequest(params)
        response = client.general.partition(request=request_obj)
    except SDKError as exc:
        raise RuntimeError(f"Unstructured API error for {pdf_path.name}: {exc}") from exc

    return response.elements or []


def elements_to_chunks(elements) -> Tuple[List[str], List[List[int]]]:
    """Normalize Unstructured API elements into chunk text + page hints."""
    chunk_texts: List[str] = []
    pages: List[List[int]] = []

    for element in elements:
        data = element.to_dict() if hasattr(element, "to_dict") else element
        text = (data.get("text") or "").strip()
        if not text:
            continue

        metadata = data.get("metadata") or {}
        page_numbers: List[int] = []

        for candidate in (
            metadata.get("page_number"),
            metadata.get("page"),
        ):
            if candidate is None:
                continue
            try:
                page_numbers.append(int(candidate))
            except (TypeError, ValueError):
                continue

        page_range = metadata.get("page_range")
        if isinstance(page_range, (list, tuple)):
            for val in page_range:
                try:
                    page_numbers.append(int(val))
                except (TypeError, ValueError):
                    continue
        elif isinstance(page_range, str) and "-" in page_range:
            start, end = page_range.split("-", 1)
            try:
                start_i, end_i = int(start), int(end)
                page_numbers.extend(list(range(start_i, end_i + 1)))
            except ValueError:
                pass

        chunk_texts.append(text)
        pages.append(sorted(set(page_numbers)))

    return chunk_texts, pages


def ingest_pdfs(
    pdfs: Sequence[Path],
    unstructured_client: UnstructuredClient,
    strategy: str,
    chunking_strategy: str | None,
    max_characters: int,
    overlap: int,
    languages: List[str] | None,
    partitions_dir: Path,
    chunks_dir: Path,
    redis_client,
    redis_prefix: str,
    redis_skip_processed: bool,
    store_partitions_disk: bool,
    store_chunks_disk: bool,
) -> List[dict]:
    """Partition PDFs with Unstructured and dump partitions/chunks to disk."""
    if not pdfs:
        logger.warning("No PDFs to ingest")
        return []

    results: List[dict] = []

    for pdf_path in pdfs:
        logger.info("Processing %s via Unstructured API", pdf_path.name)

        try:
            file_bytes = pdf_path.read_bytes()
        except OSError as exc:
            logger.exception("Failed to read %s: %s", pdf_path.name, exc)
            continue

        doc_id = _hash_bytes(file_bytes)
        meta_key = _redis_key(redis_prefix, doc_id, "meta")
        chunks_key = _redis_key(redis_prefix, doc_id, "chunks")
        partitions_key = _redis_key(redis_prefix, doc_id, "partitions")

        if redis_client and redis_skip_processed:
            try:
                if redis_client.exists(meta_key):
                    logger.info("Skipping %s (already processed: %s)", pdf_path.name, doc_id)
                    results.append(
                        {
                            "file": pdf_path.name,
                            "document_id": doc_id,
                            "skipped": True,
                        }
                    )
                    continue
            except Exception as exc:
                logger.warning("Redis check failed for %s: %s", pdf_path.name, exc)

        try:
            elements = partition_pdf(
                unstructured_client,
                pdf_path=pdf_path,
                file_bytes=file_bytes,
                strategy=strategy,
                chunking_strategy=chunking_strategy,
                max_characters=max_characters,
                overlap=overlap,
                languages=languages,
            )
        except Exception as exc:
            logger.exception("Failed to partition %s: %s", pdf_path.name, exc)
            continue

        elements_payload = [
            element.to_dict() if hasattr(element, "to_dict") else element
            for element in elements
        ]
        partition_path = partitions_dir / f"{pdf_path.stem}.json"
        partition_path_value = str(partition_path) if store_partitions_disk else None
        if store_partitions_disk:
            _write_json(partition_path, elements_payload)

        chunk_payloads, page_ranges = elements_to_chunks(elements_payload)
        if not chunk_payloads:
            logger.warning("No text extracted from %s", pdf_path.name)
            continue

        chunk_items = []
        for idx, (chunk, page_list) in enumerate(zip(chunk_payloads, page_ranges)):
            chunk_items.append(
                {
                    "document_id": doc_id,
                    "source": pdf_path.name,
                    "chunk_index": idx,
                    "pages": page_list,
                    "text": chunk,
                }
            )

        chunk_path = chunks_dir / f"{pdf_path.stem}.json"
        chunk_path_value = str(chunk_path) if store_chunks_disk else None
        if store_chunks_disk:
            _write_json(chunk_path, chunk_items)

        if redis_client:
            try:
                upload_to_redis(
                    redis_client=redis_client,
                    doc_id=doc_id,
                    source=pdf_path.name,
                    partitions_payload=elements_payload,
                    chunks_payload=chunk_items,
                    prefix=redis_prefix,
                )
            except Exception as exc:
                logger.warning("Failed to write Redis data for %s: %s", pdf_path.name, exc)

        logger.info(
            "Wrote %d chunks for %s",
            len(chunk_payloads),
            pdf_path.name,
        )
        results.append(
            {
                "file": pdf_path.name,
                "document_id": doc_id,
                "chunks": len(chunk_payloads),
                "partition_path": partition_path_value,
                "chunk_path": chunk_path_value,
            }
        )

    return results


def parse_languages(raw: str | None) -> List[str] | None:
    if not raw:
        return None
    langs = [lang.strip() for lang in raw.split(",")]
    return [lang for lang in langs if lang]


def run_from_env(
    pdf_path_override: str | None = None,
    data_dir_override: str | None = None,
) -> List[dict]:
    load_dotenv(Path(".env"))
    data_dir = Path(data_dir_override or os.getenv("DATA_DIR", "data-raw")).expanduser()
    pdf_path_env = os.getenv("PDF_PATH")
    target_raw = pdf_path_override or pdf_path_env
    target = Path(target_raw).expanduser() if target_raw else data_dir

    partitions_dir = Path(os.getenv("PARTITIONS_DIR", "data/partitions")).expanduser()
    chunks_dir = Path(os.getenv("CHUNKS_DIR", "data/chunks")).expanduser()

    chunk_size = load_env_int("CHUNK_SIZE", 1200)
    chunk_overlap = load_env_int("CHUNK_OVERLAP", 200)
    chunking_strategy_raw = os.getenv("UNSTRUCTURED_CHUNKING_STRATEGY", "basic")
    chunking_strategy = (
        chunking_strategy_raw if chunking_strategy_raw.lower() != "none" else None
    )

    strategy = os.getenv("UNSTRUCTURED_STRATEGY", "hi_res")
    languages = parse_languages(os.getenv("UNSTRUCTURED_LANGUAGES"))
    unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
    unstructured_api_url = os.getenv("UNSTRUCTURED_API_URL", "https://api.unstructured.io")

    redis_url = os.getenv("REDIS_URL", "")
    redis_prefix = os.getenv("REDIS_PREFIX", "unstructured")
    redis_skip_processed = load_env_bool("REDIS_SKIP_PROCESSED", True)
    store_partitions_disk = load_env_bool("STORE_PARTITIONS_DISK", False)
    store_chunks_disk = load_env_bool("STORE_CHUNKS_DISK", False)
    redis_client = _get_redis_client(redis_url)

    if not unstructured_api_key:
        raise RuntimeError("UNSTRUCTURED_API_KEY is required to call the Unstructured API")

    pdfs = collect_pdfs(target)
    if not pdfs:
        logger.warning("No PDFs found at %s", target)
        return []

    unstructured_client = UnstructuredClient(
        api_key_auth=unstructured_api_key,
        server_url=unstructured_api_url,
    )

    return ingest_pdfs(
        pdfs=pdfs,
        unstructured_client=unstructured_client,
        strategy=strategy,
        chunking_strategy=chunking_strategy,
        max_characters=chunk_size,
        overlap=chunk_overlap,
        languages=languages,
        partitions_dir=partitions_dir,
        chunks_dir=chunks_dir,
        redis_client=redis_client,
        redis_prefix=redis_prefix,
        redis_skip_processed=redis_skip_processed,
        store_partitions_disk=store_partitions_disk,
        store_chunks_disk=store_chunks_disk,
    )


def main() -> None:
    results = run_from_env()
    summary = json.dumps(results, indent=2)
    logger.info("Ingestion complete:\n%s", summary)


if __name__ == "__main__":
    main()
