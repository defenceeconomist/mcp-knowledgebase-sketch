import json
import logging
import os
from pathlib import Path
from typing import List, Sequence, Tuple

from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared
from unstructured_client.models.errors import SDKError


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


def partition_pdf(
    client: UnstructuredClient,
    pdf_path: Path,
    strategy: str,
    chunking_strategy: str | None,
    max_characters: int | None,
    overlap: int | None,
    languages: List[str] | None,
):
    """Call the Unstructured API to partition (and optionally chunk) a PDF."""
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
) -> List[dict]:
    """Partition PDFs with Unstructured and dump partitions/chunks to disk."""
    if not pdfs:
        logger.warning("No PDFs to ingest")
        return []

    results: List[dict] = []

    for pdf_path in pdfs:
        logger.info("Processing %s via Unstructured API", pdf_path.name)

        try:
            elements = partition_pdf(
                unstructured_client,
                pdf_path=pdf_path,
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
        _write_json(partition_path, elements_payload)

        chunk_payloads, page_ranges = elements_to_chunks(elements_payload)
        if not chunk_payloads:
            logger.warning("No text extracted from %s", pdf_path.name)
            continue

        chunk_items = []
        for idx, (chunk, page_list) in enumerate(zip(chunk_payloads, page_ranges)):
            chunk_items.append(
                {
                    "source": pdf_path.name,
                    "chunk_index": idx,
                    "pages": page_list,
                    "text": chunk,
                }
            )

        chunk_path = chunks_dir / f"{pdf_path.stem}.json"
        _write_json(chunk_path, chunk_items)
        logger.info(
            "Wrote %d chunks for %s",
            len(chunk_payloads),
            pdf_path.name,
        )
        results.append(
            {
                "file": pdf_path.name,
                "chunks": len(chunk_payloads),
                "partition_path": str(partition_path),
                "chunk_path": str(chunk_path),
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
    )


def main() -> None:
    results = run_from_env()
    summary = json.dumps(results, indent=2)
    logger.info("Ingestion complete:\n%s", summary)


if __name__ == "__main__":
    main()
