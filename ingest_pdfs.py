import json
import logging
import os
import uuid
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def extract_pdf_elements(pdf_path: Path) -> List[dict]:
    """Read a PDF with PyMuPDF and return paragraph-level elements with page numbers."""
    try:
        with fitz.open(pdf_path) as doc:
            elements: List[dict] = []
            for page_idx, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                if not paragraphs and text.strip():
                    paragraphs = [text.strip()]

                for paragraph in paragraphs:
                    elements.append(
                        {
                            "text": paragraph,
                            "metadata": {"page_number": page_idx},
                        }
                    )
        return elements
    except Exception as exc:
        raise RuntimeError(f"Failed to read {pdf_path.name}: {exc}") from exc


def chunk_elements(
    elements: Sequence[dict], chunk_size: int, chunk_overlap: int
) -> Iterable[Tuple[str, List[int]]]:
    """Chunk parsed elements into overlapping text blocks while preserving page hints."""
    buffer: List[str] = []
    buffer_len = 0
    pages: List[int] = []

    for element in elements:
        text = (element or {}).get("text") or ""
        text = text.strip()
        if not text:
            continue

        page = (element.get("metadata") or {}).get("page_number")
        projected_len = buffer_len + len(text) + 1
        if projected_len > chunk_size and buffer:
            chunk_text = " ".join(buffer).strip()
            yield chunk_text, sorted(set(pages))

            if chunk_overlap > 0 and chunk_text:
                overlap_text = chunk_text[-chunk_overlap:]
                buffer = [overlap_text]
                buffer_len = len(overlap_text)
            else:
                buffer = []
                buffer_len = 0
            pages = []

        buffer.append(text)
        buffer_len += len(text) + 1
        if page is not None:
            pages.append(int(page))

    if buffer:
        chunk_text = " ".join(buffer).strip()
        yield chunk_text, sorted(set(pages))


def embed_chunks(model: SentenceTransformer, chunks: List[str]) -> List[List[float]]:
    """Generate vector embeddings for text chunks."""
    if not chunks:
        return []
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    """Create the collection if it does not exist."""
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    pdf_name: str,
    chunks: List[str],
    vectors: List[List[float]],
    pages: List[List[int]],
) -> None:
    """Write chunk payloads + vectors to Qdrant."""
    points = []
    for idx, (chunk, vector, page_list) in enumerate(zip(chunks, vectors, pages)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
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


def collect_pdfs(target: Path) -> List[Path]:
    """Resolve a single PDF or all PDFs in a directory."""
    if target.is_file():
        if target.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {target}")
        return [target]
    if target.is_dir():
        return sorted(target.glob("*.pdf"))
    raise FileNotFoundError(f"PDF path not found: {target}")


def ingest_pdfs(
    pdfs: Sequence[Path],
    client: QdrantClient,
    collection: str,
    model: SentenceTransformer,
    chunk_size: int,
    chunk_overlap: int,
) -> List[dict]:
    """Ingest one or more PDFs into Qdrant."""
    if not pdfs:
        logger.warning("No PDFs to ingest")
        return []

    ensure_collection(client, collection, model.get_sentence_embedding_dimension())

    results: List[dict] = []

    for pdf_path in pdfs:
        logger.info("Processing %s", pdf_path.name)
        try:
            elements = extract_pdf_elements(pdf_path)
        except Exception as exc:
            logger.exception("Failed to extract %s: %s", pdf_path.name, exc)
            continue

        chunk_payloads: List[str] = []
        page_ranges: List[List[int]] = []
        for chunk_text, chunk_pages in chunk_elements(
            elements, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ):
            chunk_payloads.append(chunk_text)
            page_ranges.append(chunk_pages)

        if not chunk_payloads:
            logger.warning("No text extracted from %s", pdf_path.name)
            continue

        vectors = embed_chunks(model, chunk_payloads)
        upsert_chunks(
            client,
            collection=collection,
            pdf_name=pdf_path.name,
            chunks=chunk_payloads,
            vectors=vectors,
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


def load_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%s, using %d", key, raw, default)
        return default


def run_from_env() -> List[dict]:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    pdf_path_override = os.getenv("PDF_PATH")
    target = Path(pdf_path_override).expanduser() if pdf_path_override else data_dir
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = load_env_int("QDRANT_PORT", 6333)
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "pdf_chunks")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size = load_env_int("CHUNK_SIZE", 1200)
    chunk_overlap = load_env_int("CHUNK_OVERLAP", 200)
    pdfs = collect_pdfs(target)
    if not pdfs:
        logger.warning("No PDFs found at %s", target)
        return []

    model = SentenceTransformer(embedding_model_name)
    client = QdrantClient(host=qdrant_host, port=qdrant_port, prefer_grpc=False)

    return ingest_pdfs(
        pdfs=pdfs,
        client=client,
        collection=qdrant_collection,
        model=model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def main() -> None:
    results = run_from_env()
    summary = json.dumps(results, indent=2)
    logger.info("Ingestion complete:\n%s", summary)


if __name__ == "__main__":
    main()
