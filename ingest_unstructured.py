import json
import logging
import os
import uuid
from pathlib import Path
from typing import List, Sequence, Tuple

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models
from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared
from unstructured_client.models.errors import SDKError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


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


def _to_list(x) -> List[float]:
    return x.tolist() if hasattr(x, "tolist") else list(x)


def embed_chunks(
    dense_model: TextEmbedding, sparse_model: SparseTextEmbedding, chunks: List[str]
) -> Tuple[List[List[float]], List[models.SparseVector]]:
    """Generate dense + sparse embeddings for text chunks."""
    if not chunks:
        return [], []

    dense_embs = list(dense_model.embed(chunks))
    sparse_embs = list(sparse_model.embed(chunks))

    dense_vectors: List[List[float]] = []
    sparse_vectors: List[models.SparseVector] = []

    for dense_vec, sparse_vec in zip(dense_embs, sparse_embs):
        dense_vectors.append(_to_list(dense_vec))
        sparse_vectors.append(
            models.SparseVector(
                indices=list(sparse_vec.indices),
                values=list(sparse_vec.values),
            )
        )

    return dense_vectors, sparse_vectors


def ensure_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    """Create the collection with named dense + sparse vectors if it does not exist."""
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    pdf_name: str,
    chunks: List[str],
    dense_vectors: List[List[float]],
    sparse_vectors: List[models.SparseVector],
    pages: List[List[int]],
) -> None:
    """Write chunk payloads + vectors to Qdrant."""
    points = []
    for idx, (chunk, dense_vector, sparse_vector, page_list) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors, pages)
    ):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector,
                },
                payload={
                    "source": pdf_name,
                    "chunk_index": idx,
                    "pages": page_list,
                    "text": chunk,
                },
            )
        )

    if points:
        client.upsert(collection_name=collection, points=points)


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
    client: QdrantClient,
    collection: str,
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
    unstructured_client: UnstructuredClient,
    strategy: str,
    chunking_strategy: str | None,
    max_characters: int,
    overlap: int,
    languages: List[str] | None,
) -> List[dict]:
    """Ingest PDFs into Qdrant using the Unstructured API for partitioning."""
    if not pdfs:
        logger.warning("No PDFs to ingest")
        return []

    dense_dim_probe = len(_to_list(next(iter(dense_model.embed(["dimension probe"])))))
    ensure_collection(client, collection, dense_dim_probe)
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

        chunk_payloads, page_ranges = elements_to_chunks(elements)
        if not chunk_payloads:
            logger.warning("No text extracted from %s", pdf_path.name)
            continue

        dense_vectors, sparse_vectors = embed_chunks(dense_model, sparse_model, chunk_payloads)
        upsert_chunks(
            client,
            collection=collection,
            pdf_name=pdf_path.name,
            chunks=chunk_payloads,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            pages=page_ranges,
        )
        logger.info(
            "Stored %d chunks from %s into collection '%s'",
            len(chunk_payloads),
            pdf_path.name,
            collection,
        )
        results.append(
            {
                "file": pdf_path.name,
                "chunks": len(chunk_payloads),
                "pages": page_ranges,
            }
        )

    return results


def parse_languages(raw: str | None) -> List[str] | None:
    if not raw:
        return None
    langs = [lang.strip() for lang in raw.split(",")]
    return [lang for lang in langs if lang]


def run_from_env() -> List[dict]:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    pdf_path_override = os.getenv("PDF_PATH")
    target = Path(pdf_path_override).expanduser() if pdf_path_override else data_dir

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = load_env_int("QDRANT_PORT", 6333)
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "pdf_chunks")
    dense_model_name = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    sparse_model_name = os.getenv("SPARSE_MODEL", "Qdrant/bm25")

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

    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port, prefer_grpc=False)
    dense_model = TextEmbedding(model_name=dense_model_name)
    sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
    unstructured_client = UnstructuredClient(
        api_key_auth=unstructured_api_key,
        server_url=unstructured_api_url,
    )

    return ingest_pdfs(
        pdfs=pdfs,
        client=qdrant,
        collection=qdrant_collection,
        dense_model=dense_model,
        sparse_model=sparse_model,
        unstructured_client=unstructured_client,
        strategy=strategy,
        chunking_strategy=chunking_strategy,
        max_characters=chunk_size,
        overlap=chunk_overlap,
        languages=languages,
    )


def main() -> None:
    results = run_from_env()
    summary = json.dumps(results, indent=2)
    logger.info("Ingestion complete:\n%s", summary)


if __name__ == "__main__":
    main()
