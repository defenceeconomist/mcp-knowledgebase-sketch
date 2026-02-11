from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:  # pragma: no cover - optional dependency
    Minio = None  # type: ignore[assignment]
    S3Error = Exception  # type: ignore[assignment]

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None  # type: ignore[assignment]


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
TITLE_STOP_WORDS = {"abstract", "introduction", "keywords", "contents", "references"}

MINIO_TITLE_KEYS = {
    "title",
    "dc:title",
    "dc-title",
    "pdf:title",
    "pdf-title",
    "citation-title",
}
MINIO_AUTHOR_KEYS = {
    "author",
    "authors",
    "creator",
    "dc:creator",
    "dc-creator",
    "citation-author",
}
MINIO_DOI_KEYS = {
    "doi",
    "dc:identifier",
    "dc-identifier",
    "identifier-doi",
    "citation-doi",
    "prism:doi",
    "prism-doi",
}

ALLOWED_ENTRY_TYPES = {"article", "inproceedings", "inbook", "incollection", "book", "misc", "techreport"}


def load_dotenv(path: Path) -> None:
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


def _decode_redis_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return value


def _load_env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_env_list(key: str) -> List[str]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _default_citation_key(object_name: str) -> str:
    stem = Path(object_name).stem.lower()
    citation_key = re.sub(r"[^a-z0-9]+", "", stem)
    return citation_key[:80] or "untitled"


def _default_bibtex_metadata(object_name: str) -> Dict[str, Any]:
    return {
        "citationKey": _default_citation_key(object_name),
        "entryType": "article",
        "title": "",
        "year": "",
        "authors": [],
        "journal": "",
        "booktitle": "",
        "publisher": "",
        "volume": "",
        "number": "",
        "pages": "",
        "doi": "",
        "url": "",
        "keywords": "",
        "abstract": "",
        "note": "",
    }


def _normalize_authors(value: Any) -> List[Dict[str, str]]:
    authors: List[Dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                first_name = _normalize_text(item.get("firstName"))
                last_name = _normalize_text(item.get("lastName"))
                if first_name or last_name:
                    authors.append({"firstName": first_name, "lastName": last_name})
            elif isinstance(item, str):
                parsed = _parse_author_names(item)
                for name in parsed:
                    authors.append(_author_to_bibtex(name))
        return authors
    if isinstance(value, str):
        return [_author_to_bibtex(name) for name in _parse_author_names(value)]
    return authors


def _normalize_bibtex_metadata(object_name: str, metadata: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = _default_bibtex_metadata(object_name)
    if not isinstance(metadata, dict):
        return payload

    for key in (
        "citationKey",
        "title",
        "year",
        "journal",
        "booktitle",
        "publisher",
        "volume",
        "number",
        "pages",
        "doi",
        "url",
        "keywords",
        "abstract",
        "note",
    ):
        payload[key] = _normalize_text(metadata.get(key))

    entry_type = _normalize_text(metadata.get("entryType")).lower()
    if entry_type in ALLOWED_ENTRY_TYPES:
        payload["entryType"] = entry_type

    payload["authors"] = _normalize_authors(metadata.get("authors"))
    if not payload["citationKey"]:
        payload["citationKey"] = _default_citation_key(object_name)
    return payload


def _bibtex_file_key(prefix: str, bucket: str, object_name: str) -> str:
    return f"{prefix}:file:{bucket}/{object_name}"


def _source_key(prefix: str, source: str) -> str:
    return f"{prefix}:pdf:source:{source}"


def _redis_key(prefix: str, doc_id: str, suffix: str) -> str:
    return f"{prefix}:pdf:{doc_id}:{suffix}"


def _safe_json_loads(value: Any, default: Any) -> Any:
    raw = _decode_redis_value(value)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def _parse_author_names(raw: str) -> List[str]:
    text = _normalize_text(raw)
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(";", ",")
    text = text.replace(" and ", ",")
    candidates = [entry.strip() for entry in text.split(",") if entry.strip()]
    names: List[str] = []
    for candidate in candidates:
        cleaned = re.sub(r"\d", "", candidate).strip()
        if "@" in cleaned:
            continue
        if len(cleaned.split()) > 5:
            continue
        if cleaned:
            names.append(cleaned)
    deduped: List[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def _author_to_bibtex(name: str) -> Dict[str, str]:
    parts = [part for part in name.strip().split(" ") if part]
    if not parts:
        return {"firstName": "", "lastName": ""}
    if len(parts) == 1:
        return {"firstName": "", "lastName": parts[0]}
    return {"firstName": " ".join(parts[:-1]), "lastName": parts[-1]}


def _sanitize_doi(value: str) -> str:
    doi = _normalize_text(value)
    if doi.lower().startswith("https://doi.org/"):
        doi = doi[16:]
    if doi.lower().startswith("http://doi.org/"):
        doi = doi[15:]
    match = DOI_PATTERN.search(doi)
    if not match:
        return ""
    return match.group(0).strip().rstrip(".,;)")


def _extract_doi_from_text(text: str) -> str:
    match = DOI_PATTERN.search(text or "")
    if not match:
        return ""
    return _sanitize_doi(match.group(0))


def _normalize_minio_metadata(raw_metadata: Dict[Any, Any] | None) -> Dict[str, str]:
    if not raw_metadata:
        return {}
    out: Dict[str, str] = {}
    for key, value in raw_metadata.items():
        key_text = _normalize_text(_decode_redis_value(key)).lower()
        if key_text.startswith("x-amz-meta-"):
            key_text = key_text[len("x-amz-meta-") :]
        key_text = key_text.replace("_", "-")
        out[key_text] = _normalize_text(_decode_redis_value(value))
    return out


def _looks_like_author_line(line: str) -> bool:
    lowered = line.lower()
    if any(marker in lowered for marker in ("abstract", "introduction", "doi:", "@", "http")):
        return False
    if re.search(r"\d{3,}", line):
        return False
    parts = line.replace(";", ",").split(",")
    if len(parts) >= 2:
        return True
    return " and " in lowered and len(line.split()) <= 12


def _extract_title_from_text(text: str) -> str:
    if not text:
        return ""
    lines = [re.sub(r"\s+", " ", line).strip(" -\t") for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        flattened = re.sub(r"\s+", " ", text).strip()
        return flattened[:220]

    for line in lines[:20]:
        lowered = line.lower()
        if DOI_PATTERN.search(line):
            continue
        first_word = lowered.split(" ", 1)[0] if lowered else ""
        if first_word in TITLE_STOP_WORDS:
            continue
        if len(line) < 12 or len(line) > 220:
            continue
        if _looks_like_author_line(line):
            continue
        return line
    return ""


def _extract_authors_from_text(text: str, title: str) -> List[str]:
    if not text:
        return []
    lines = [re.sub(r"\s+", " ", line).strip(" -\t") for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    start_idx = 0
    for idx, line in enumerate(lines[:12]):
        if title and line == title:
            start_idx = idx + 1
            break

    for line in lines[start_idx : start_idx + 8]:
        if not _looks_like_author_line(line):
            continue
        names = _parse_author_names(line)
        if names:
            return names
    return []


def _extract_text_from_entry(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "chunk_text", "content", "page_content", "raw_text", "chunk"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _entry_page_start(entry: Dict[str, Any]) -> int | None:
    page_start = _to_int(entry.get("page_start"))
    if page_start is not None:
        return page_start
    pages = entry.get("pages")
    if isinstance(pages, list):
        numeric = [_to_int(item) for item in pages]
        valid = [item for item in numeric if item is not None]
        if valid:
            return min(valid)
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        for key in ("page_number", "page"):
            page = _to_int(metadata.get(key))
            if page is not None:
                return page
    return None


def _first_page_text_from_entries(entries: List[Any]) -> str:
    candidates: List[Tuple[int, int, str]] = []
    fallback: List[str] = []
    for entry in entries:
        text = _extract_text_from_entry(entry)
        if not text:
            continue
        fallback.append(text)
        if not isinstance(entry, dict):
            continue
        page_start = _entry_page_start(entry)
        chunk_index = _to_int(entry.get("chunk_index")) or 0
        if page_start == 1:
            candidates.append((0, chunk_index, text))
        elif page_start is not None:
            candidates.append((page_start, chunk_index, text))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]
    return fallback[0] if fallback else ""


def _source_doc_ids(redis_client: Any, prefix: str, source: str) -> List[str]:
    source_mapping_key = _source_key(prefix, source)
    doc_ids: List[str] = []
    if hasattr(redis_client, "smembers"):
        raw_members = redis_client.smembers(source_mapping_key) or set()
        for entry in raw_members:
            decoded = _normalize_text(_decode_redis_value(entry))
            if decoded:
                doc_ids.append(decoded)
    if not doc_ids:
        raw_value = _normalize_text(_decode_redis_value(redis_client.get(source_mapping_key)))
        if raw_value:
            doc_ids.append(raw_value)
    unique = sorted(set(doc_ids))
    return [doc_id for doc_id in unique if doc_id]


def _doc_payloads(redis_client: Any, prefix: str, doc_id: str) -> Tuple[List[Any], List[Any]]:
    meta_key = _redis_key(prefix, doc_id, "meta")
    partitions_key = _redis_key(prefix, doc_id, "partitions")
    chunks_key = _redis_key(prefix, doc_id, "chunks")
    if hasattr(redis_client, "hgetall"):
        meta = redis_client.hgetall(meta_key) or {}
        mapped = {_normalize_text(_decode_redis_value(k)): _decode_redis_value(v) for k, v in meta.items()}
        partitions_key = _normalize_text(mapped.get("partitions_key")) or partitions_key
        chunks_key = _normalize_text(mapped.get("chunks_key")) or chunks_key
    partitions = _safe_json_loads(redis_client.get(partitions_key), [])
    chunks = _safe_json_loads(redis_client.get(chunks_key), [])
    if not isinstance(partitions, list):
        partitions = []
    if not isinstance(chunks, list):
        chunks = []
    return partitions, chunks


def _first_page_text_from_redis(redis_client: Any, source_prefix: str, source: str) -> Tuple[List[str], str]:
    doc_ids = _source_doc_ids(redis_client, source_prefix, source)
    for doc_id in doc_ids:
        partitions, chunks = _doc_payloads(redis_client, source_prefix, doc_id)
        text = _first_page_text_from_entries(chunks)
        if not text:
            text = _first_page_text_from_entries(partitions)
        if text:
            return doc_ids, text
    return doc_ids, ""


def _extract_candidate_signals(minio_metadata: Dict[str, str], first_page_text: str) -> Dict[str, Any]:
    title = ""
    authors: List[str] = []
    doi = ""

    for key in MINIO_DOI_KEYS:
        if key in minio_metadata:
            doi = _sanitize_doi(minio_metadata[key])
            if doi:
                break
    for key in MINIO_TITLE_KEYS:
        if key in minio_metadata:
            title = _normalize_text(minio_metadata[key])
            if title:
                break
    for key in MINIO_AUTHOR_KEYS:
        if key in minio_metadata:
            authors = _parse_author_names(minio_metadata[key])
            if authors:
                break

    if not doi:
        doi = _extract_doi_from_text(first_page_text)
    if not title:
        title = _extract_title_from_text(first_page_text)
    if not authors:
        authors = _extract_authors_from_text(first_page_text, title)

    return {
        "doi": doi,
        "title": title,
        "authors": authors,
        "first_page_text": first_page_text,
    }


def _token_set(text: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return {token for token in cleaned.split() if len(token) > 2}


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens.intersection(right_tokens))
    return intersection / float(len(left_tokens))


def _symmetric_token_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return (_token_overlap_score(left, right) + _token_overlap_score(right, left)) / 2.0


def _env_float(key: str, default: float, min_value: float, max_value: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def _author_last_names(authors: List[str]) -> set[str]:
    last_names: set[str] = set()
    for author in authors:
        parts = [part for part in author.strip().split(" ") if part]
        if parts:
            last_names.add(parts[-1].lower())
    return last_names


def _author_overlap_ratio(query_authors: List[str], candidate_authors: List[str]) -> float:
    query_last = _author_last_names(query_authors)
    if not query_last:
        return 0.5
    candidate_last = _author_last_names(candidate_authors)
    if not candidate_last:
        return 0.0
    intersection = query_last.intersection(candidate_last)
    return len(intersection) / float(len(query_last))


def _query_year(value: Any) -> int | None:
    text = _normalize_text(value)
    if not text:
        return None
    match = re.search(r"(19|20)\d{2}", text)
    if not match:
        return None
    return _to_int(match.group(0))


def _year_match_score(query_year: int | None, candidate_year: int | None) -> float:
    if query_year is None:
        return 0.5
    if candidate_year is None:
        return 0.0
    if query_year == candidate_year:
        return 1.0
    if abs(query_year - candidate_year) == 1:
        return 0.5
    return 0.0


def _crossref_raw_score(item: Dict[str, Any]) -> float:
    try:
        value = float(item.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, value)


def _candidate_quality(
    item: Dict[str, Any],
    *,
    query_title: str,
    query_authors: List[str],
    query_year: int | None,
    query_doi: str,
    max_crossref_score: float,
) -> Dict[str, Any]:
    candidate_title = _crossref_title(item)
    candidate_authors = _crossref_authors(item)
    candidate_year = _query_year(_crossref_year(item))
    candidate_doi = _sanitize_doi(_normalize_text(item.get("DOI")))
    crossref_score_raw = _crossref_raw_score(item)
    crossref_score_norm = crossref_score_raw / max_crossref_score if max_crossref_score > 0 else 0.0
    title_similarity = _symmetric_token_similarity(query_title, candidate_title)
    author_overlap = _author_overlap_ratio(query_authors, candidate_authors)
    year_match = _year_match_score(query_year, candidate_year)

    doi_conflict = False
    doi_consistency = 0.5
    clean_query_doi = _sanitize_doi(query_doi)
    if clean_query_doi:
        if candidate_doi and candidate_doi.lower() != clean_query_doi.lower():
            doi_conflict = True
            doi_consistency = 0.0
        elif candidate_doi:
            doi_consistency = 1.0

    confidence = (
        (0.45 * title_similarity)
        + (0.20 * author_overlap)
        + (0.20 * crossref_score_norm)
        + (0.10 * year_match)
        + (0.05 * doi_consistency)
    )
    if query_authors and author_overlap == 0.0:
        confidence *= 0.75
    if query_title and title_similarity < 0.30:
        confidence *= 0.60
    if doi_conflict:
        confidence = 0.0

    return {
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "title_similarity": round(title_similarity, 4),
        "author_overlap": round(author_overlap, 4),
        "year_match": round(year_match, 4),
        "crossref_score_raw": round(crossref_score_raw, 4),
        "crossref_score_norm": round(crossref_score_norm, 4),
        "doi_consistency": round(doi_consistency, 4),
        "candidate_title": candidate_title,
        "candidate_doi": candidate_doi,
        "candidate_year": str(candidate_year) if candidate_year else "",
        "candidate_authors": candidate_authors,
        "doi_conflict": doi_conflict,
    }


def _crossref_authors(message: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for author in message.get("author", []) or []:
        if not isinstance(author, dict):
            continue
        given = _normalize_text(author.get("given"))
        family = _normalize_text(author.get("family"))
        name = _normalize_text(author.get("name"))
        full_name = " ".join(part for part in (given, family) if part).strip() or name
        if full_name:
            out.append(full_name)
    return out


def _crossref_title(message: Dict[str, Any]) -> str:
    title = message.get("title")
    if isinstance(title, list) and title:
        return _normalize_text(title[0])
    if isinstance(title, str):
        return _normalize_text(title)
    return ""


def _http_get_json(url: str, timeout: int, headers: Dict[str, str]) -> Dict[str, Any]:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - controlled URLs
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    return data if isinstance(data, dict) else {}


class CrossrefClient:
    def __init__(
        self,
        *,
        api_url: str = "https://api.crossref.org",
        timeout_seconds: int = 20,
        rows: int = 5,
        mailto: str = "",
        user_agent: str = "",
        throttle_seconds: float = 0.15,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.rows = rows
        self.mailto = mailto.strip()
        self.user_agent = user_agent.strip()
        self.throttle_seconds = max(0.0, throttle_seconds)

    def _headers(self) -> Dict[str, str]:
        user_agent = self.user_agent
        if not user_agent:
            user_agent = "mcp-research-bibtex/0.1"
            if self.mailto:
                user_agent += f" (mailto:{self.mailto})"
        return {"Accept": "application/json", "User-Agent": user_agent}

    def _get(self, path: str, query: Dict[str, Any] | None = None) -> Dict[str, Any]:
        url = f"{self.api_url}{path}"
        if query:
            query_params = {k: v for k, v in query.items() if v not in (None, "")}
            if self.mailto and "mailto" not in query_params:
                query_params["mailto"] = self.mailto
            url += "?" + urlencode(query_params)
        try:
            payload = _http_get_json(url=url, timeout=self.timeout_seconds, headers=self._headers())
        except HTTPError as exc:
            if exc.code == 404:
                return {}
            raise
        if self.throttle_seconds > 0:
            time.sleep(self.throttle_seconds)
        return payload

    def lookup_by_doi(self, doi: str) -> Dict[str, Any] | None:
        clean_doi = _sanitize_doi(doi)
        if not clean_doi:
            return None
        payload = self._get(f"/works/{quote(clean_doi, safe='')}")
        message = payload.get("message")
        return message if isinstance(message, dict) else None

    def search(self, title: str, authors: List[str]) -> List[Dict[str, Any]]:
        if not title:
            return []
        first_author = authors[0] if authors else ""
        payload = self._get(
            "/works",
            query={
                "query.title": title,
                "query.author": first_author,
                "rows": self.rows,
            },
        )
        message = payload.get("message")
        if not isinstance(message, dict):
            return []
        items = message.get("items")
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    def resolve_best_match(
        self,
        *,
        doi: str,
        title: str,
        authors: List[str],
        year: int | None = None,
        confidence_threshold: float = 0.85,
        min_title_similarity: float = 0.55,
        min_author_overlap: float = 0.25,
    ) -> Dict[str, Any]:
        clean_doi = _sanitize_doi(doi)
        if clean_doi:
            by_doi = self.lookup_by_doi(clean_doi)
            if by_doi:
                candidate_doi = _sanitize_doi(_normalize_text(by_doi.get("DOI")))
                doi_conflict = bool(candidate_doi and candidate_doi.lower() != clean_doi.lower())
                diagnostics = {
                    "matched_by": "doi",
                    "confidence": 0.0 if doi_conflict else 1.0,
                    "title_similarity": 1.0 if not doi_conflict else 0.0,
                    "author_overlap": 1.0 if not doi_conflict else 0.0,
                    "year_match": 1.0 if not doi_conflict else 0.0,
                    "crossref_score_raw": _crossref_raw_score(by_doi),
                    "crossref_score_norm": 1.0 if not doi_conflict else 0.0,
                    "doi_consistency": 0.0 if doi_conflict else 1.0,
                    "candidate_doi": candidate_doi,
                    "threshold": confidence_threshold,
                }
                if doi_conflict:
                    return {
                        "status": "doi_conflict",
                        "message": None,
                        "diagnostics": diagnostics,
                        "candidates": [],
                    }
                return {
                    "status": "match",
                    "message": by_doi,
                    "diagnostics": diagnostics,
                    "candidates": [],
                }

        if not title:
            return {"status": "no_match", "message": None, "diagnostics": {}, "candidates": []}

        items = self.search(title, authors)
        if not items:
            return {"status": "no_match", "message": None, "diagnostics": {}, "candidates": []}

        max_crossref_score = max((_crossref_raw_score(item) for item in items), default=0.0)
        ranked: List[Dict[str, Any]] = []
        for item in items:
            quality = _candidate_quality(
                item,
                query_title=title,
                query_authors=authors,
                query_year=year,
                query_doi=clean_doi,
                max_crossref_score=max_crossref_score,
            )
            ranked.append({"item": item, "quality": quality})

        ranked.sort(key=lambda entry: entry["quality"]["confidence"], reverse=True)
        best = ranked[0]
        best_item = best["item"]
        best_quality = best["quality"]
        candidates = [
            {
                "title": _crossref_title(entry["item"]),
                "doi": _sanitize_doi(_normalize_text(entry["item"].get("DOI"))),
                "confidence": entry["quality"]["confidence"],
                "crossref_score_raw": entry["quality"]["crossref_score_raw"],
            }
            for entry in ranked[:3]
        ]
        diagnostics = {
            "matched_by": "search",
            **best_quality,
            "threshold": confidence_threshold,
        }

        if best_quality["doi_conflict"]:
            return {
                "status": "doi_conflict",
                "message": None,
                "diagnostics": diagnostics,
                "candidates": candidates,
            }

        author_gate_failed = bool(authors) and best_quality["author_overlap"] < min_author_overlap
        title_gate_failed = bool(title) and best_quality["title_similarity"] < min_title_similarity
        confidence_gate_failed = best_quality["confidence"] < confidence_threshold
        if author_gate_failed or title_gate_failed or confidence_gate_failed:
            return {
                "status": "low_confidence",
                "message": None,
                "diagnostics": diagnostics,
                "candidates": candidates,
            }

        return {
            "status": "match",
            "message": best_item,
            "diagnostics": diagnostics,
            "candidates": candidates,
        }

    def fetch_best_match(self, *, doi: str, title: str, authors: List[str]) -> Dict[str, Any] | None:
        resolved = self.resolve_best_match(doi=doi, title=title, authors=authors, year=None, confidence_threshold=0.0)
        message = resolved.get("message")
        return message if isinstance(message, dict) else None


def _crossref_year(message: Dict[str, Any]) -> str:
    for key in ("issued", "published-print", "published-online", "created"):
        value = message.get(key)
        if not isinstance(value, dict):
            continue
        date_parts = value.get("date-parts")
        if not isinstance(date_parts, list) or not date_parts:
            continue
        first = date_parts[0]
        if not isinstance(first, list) or not first:
            continue
        year = _to_int(first[0])
        if year:
            return str(year)
    return ""


def _entry_type_from_crossref(crossref_type: str) -> str:
    mapping = {
        "journal-article": "article",
        "article-journal": "article",
        "proceedings-article": "inproceedings",
        "proceedings": "inproceedings",
        "book": "book",
        "book-chapter": "incollection",
        "report": "techreport",
    }
    return mapping.get(crossref_type.lower(), "misc")


def _strip_html(value: str) -> str:
    if not value:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def _build_citation_key(title: str, authors: List[Dict[str, str]], year: str, fallback: str) -> str:
    family = ""
    if authors:
        family = _normalize_text(authors[0].get("lastName")).lower()
    first_title_token = ""
    for token in re.split(r"[^a-zA-Z0-9]+", title):
        if token:
            first_title_token = token.lower()
            break
    candidate = f"{family}{year}{first_title_token}"
    candidate = re.sub(r"[^a-z0-9]+", "", candidate)
    if not candidate:
        candidate = _default_citation_key(fallback)
    return candidate[:80]


def _crossref_to_bibtex(message: Dict[str, Any], object_name: str) -> Dict[str, Any]:
    title = _crossref_title(message)
    authors = [_author_to_bibtex(name) for name in _crossref_authors(message)]
    year = _crossref_year(message)
    doi = _sanitize_doi(_normalize_text(message.get("DOI")))
    url = _normalize_text(message.get("URL"))
    container_title = message.get("container-title")
    container = ""
    if isinstance(container_title, list) and container_title:
        container = _normalize_text(container_title[0])
    elif isinstance(container_title, str):
        container = _normalize_text(container_title)

    entry_type = _entry_type_from_crossref(_normalize_text(message.get("type")))
    metadata = _default_bibtex_metadata(object_name)
    metadata.update(
        {
            "entryType": entry_type,
            "title": title,
            "year": year,
            "authors": authors,
            "volume": _normalize_text(message.get("volume")),
            "number": _normalize_text(message.get("issue")),
            "pages": _normalize_text(message.get("page")),
            "publisher": _normalize_text(message.get("publisher")),
            "doi": doi,
            "url": url,
            "abstract": _strip_html(_normalize_text(message.get("abstract"))),
        }
    )
    if entry_type in {"inproceedings", "inbook", "incollection"}:
        metadata["booktitle"] = container
    else:
        metadata["journal"] = container
    metadata["citationKey"] = _build_citation_key(title, authors, year, object_name)
    return metadata


def _metadata_is_complete(metadata: Dict[str, Any]) -> bool:
    if not metadata:
        return False
    has_title = bool(_normalize_text(metadata.get("title")))
    has_doi = bool(_normalize_text(metadata.get("doi")))
    authors = metadata.get("authors")
    has_authors = isinstance(authors, list) and len(authors) > 0
    return has_title and has_doi and has_authors


def _merge_metadata(existing: Dict[str, Any], candidate: Dict[str, Any], overwrite: bool) -> Dict[str, Any]:
    merged = dict(existing)
    for key, value in candidate.items():
        if key == "authors":
            current_authors = merged.get("authors")
            if overwrite or not current_authors:
                merged[key] = value
            continue
        current = _normalize_text(merged.get(key))
        incoming = _normalize_text(value) if not isinstance(value, list) else value
        if overwrite:
            merged[key] = incoming
        elif not current:
            merged[key] = incoming
    return merged


def _load_existing_file_metadata(redis_client: Any, bibtex_prefix: str, bucket: str, object_name: str) -> Dict[str, Any]:
    key = _bibtex_file_key(bibtex_prefix, bucket, object_name)
    payload = _safe_json_loads(redis_client.get(key), {})
    if not payload and hasattr(redis_client, "hgetall"):
        hash_payload = redis_client.hgetall(key) or {}
        payload = {_decode_redis_value(k): _decode_redis_value(v) for k, v in hash_payload.items()}
    if not isinstance(payload, dict):
        payload = {}
    return _normalize_bibtex_metadata(object_name, payload)


def _save_file_metadata(redis_client: Any, bibtex_prefix: str, bucket: str, object_name: str, metadata: Dict[str, Any]) -> str:
    key = _bibtex_file_key(bibtex_prefix, bucket, object_name)
    redis_client.set(key, json.dumps(metadata, ensure_ascii=True))
    redis_client.sadd(f"{bibtex_prefix}:files", f"{bucket}/{object_name}")
    return key


def _minio_object_metadata(minio_client: Any, bucket: str, object_name: str) -> Dict[str, str]:
    try:
        stat = minio_client.stat_object(bucket, object_name)
    except Exception:
        return {}
    raw_metadata = getattr(stat, "metadata", {}) or {}
    return _normalize_minio_metadata(raw_metadata)


def _list_minio_objects(minio_client: Any, bucket: str, object_prefix: str, suffix: str, limit: int) -> List[str]:
    object_names: List[str] = []
    for entry in minio_client.list_objects(bucket, prefix=object_prefix, recursive=True):
        if getattr(entry, "is_dir", False):
            continue
        object_name = _normalize_text(getattr(entry, "object_name", ""))
        if not object_name:
            continue
        if suffix and not object_name.lower().endswith(suffix.lower()):
            continue
        object_names.append(object_name)
        if len(object_names) >= limit:
            break
    return sorted(object_names)


def _resolve_buckets(minio_client: Any, requested_buckets: List[str], all_buckets: bool) -> List[str]:
    if requested_buckets:
        return sorted(set(requested_buckets))
    env_buckets = _load_env_list("BIBTEX_MINIO_BUCKETS") or _load_env_list("MINIO_BUCKETS")
    if env_buckets:
        return sorted(set(env_buckets))
    if all_buckets:
        return sorted(bucket.name for bucket in minio_client.list_buckets())
    source_bucket = os.getenv("SOURCE_BUCKET", "").strip()
    if source_bucket:
        return [source_bucket]
    return []


def _resolve_crossref_match(
    crossref_client: Any,
    *,
    doi: str,
    title: str,
    authors: List[str],
    year: int | None,
    confidence_threshold: float,
    min_title_similarity: float,
    min_author_overlap: float,
) -> Dict[str, Any]:
    if hasattr(crossref_client, "resolve_best_match"):
        try:
            resolved = crossref_client.resolve_best_match(
                doi=doi,
                title=title,
                authors=authors,
                year=year,
                confidence_threshold=confidence_threshold,
                min_title_similarity=min_title_similarity,
                min_author_overlap=min_author_overlap,
            )
        except TypeError:
            resolved = crossref_client.resolve_best_match(doi=doi, title=title, authors=authors)
        if isinstance(resolved, dict):
            return {
                "status": _normalize_text(resolved.get("status")) or "no_match",
                "message": resolved.get("message"),
                "diagnostics": resolved.get("diagnostics") if isinstance(resolved.get("diagnostics"), dict) else {},
                "candidates": resolved.get("candidates") if isinstance(resolved.get("candidates"), list) else [],
            }

    if hasattr(crossref_client, "fetch_best_match"):
        message = crossref_client.fetch_best_match(doi=doi, title=title, authors=authors)
        if isinstance(message, dict):
            return {
                "status": "match",
                "message": message,
                "diagnostics": {"matched_by": "legacy", "confidence": 1.0, "threshold": confidence_threshold},
                "candidates": [],
            }

    return {"status": "no_match", "message": None, "diagnostics": {}, "candidates": []}


def enrich_file_metadata(
    *,
    minio_client: Any,
    redis_client: Any,
    bucket: str,
    object_name: str,
    bibtex_prefix: str,
    source_prefix: str,
    overwrite: bool,
    dry_run: bool,
    skip_complete: bool,
    crossref_client: CrossrefClient,
) -> Dict[str, Any]:
    source = f"{bucket}/{object_name}"
    existing = _load_existing_file_metadata(redis_client, bibtex_prefix, bucket, object_name)
    if skip_complete and _metadata_is_complete(existing):
        return {"bucket": bucket, "object_name": object_name, "status": "skipped_existing"}

    minio_metadata = _minio_object_metadata(minio_client, bucket, object_name)
    doc_ids, first_page_text = _first_page_text_from_redis(redis_client, source_prefix, source)
    signals = _extract_candidate_signals(minio_metadata, first_page_text)
    if not signals["doi"] and not signals["title"]:
        return {
            "bucket": bucket,
            "object_name": object_name,
            "status": "no_signals",
            "doc_ids": doc_ids,
        }

    confidence_threshold = _env_float("CROSSREF_MATCH_CONFIDENCE_THRESHOLD", 0.85, 0.0, 1.0)
    min_title_similarity = _env_float("CROSSREF_MIN_TITLE_SIMILARITY", 0.55, 0.0, 1.0)
    min_author_overlap = _env_float("CROSSREF_MIN_AUTHOR_OVERLAP", 0.25, 0.0, 1.0)
    query_year = _query_year(existing.get("year"))
    match = _resolve_crossref_match(
        crossref_client,
        doi=signals["doi"],
        title=signals["title"],
        authors=signals["authors"],
        year=query_year,
        confidence_threshold=confidence_threshold,
        min_title_similarity=min_title_similarity,
        min_author_overlap=min_author_overlap,
    )
    crossref_message = match.get("message")
    if not crossref_message:
        status = _normalize_text(match.get("status")) or "no_match"
        if status not in {"no_match", "low_confidence", "doi_conflict"}:
            status = "no_match"
        return {
            "bucket": bucket,
            "object_name": object_name,
            "status": status,
            "doc_ids": doc_ids,
            "signals": {"doi": signals["doi"], "title": signals["title"], "authors": signals["authors"]},
            "match": match.get("diagnostics", {}),
            "candidates": match.get("candidates", []),
        }

    candidate = _crossref_to_bibtex(crossref_message, object_name)
    merged = _merge_metadata(existing, candidate, overwrite=overwrite)
    normalized = _normalize_bibtex_metadata(object_name, merged)
    if not dry_run:
        redis_key = _save_file_metadata(redis_client, bibtex_prefix, bucket, object_name, normalized)
    else:
        redis_key = _bibtex_file_key(bibtex_prefix, bucket, object_name)

    return {
        "bucket": bucket,
        "object_name": object_name,
        "status": "updated" if not dry_run else "dry_run_update",
        "redis_key": redis_key,
        "doc_ids": doc_ids,
        "doi": normalized.get("doi"),
        "title": normalized.get("title"),
        "match": match.get("diagnostics", {}),
        "candidates": match.get("candidates", []),
    }


def enrich_bucket_files(
    *,
    minio_client: Any,
    redis_client: Any,
    bucket: str,
    limit: int,
    batch_size: int,
    object_prefix: str,
    suffix: str,
    bibtex_prefix: str,
    source_prefix: str,
    overwrite: bool,
    dry_run: bool,
    skip_complete: bool,
    crossref_client: CrossrefClient,
) -> Dict[str, Any]:
    object_names = _list_minio_objects(
        minio_client=minio_client,
        bucket=bucket,
        object_prefix=object_prefix,
        suffix=suffix,
        limit=limit,
    )
    counts = {
        "updated": 0,
        "dry_run_update": 0,
        "skipped_existing": 0,
        "no_match": 0,
        "low_confidence": 0,
        "doi_conflict": 0,
        "no_signals": 0,
        "error": 0,
    }
    results: List[Dict[str, Any]] = []
    for start in range(0, len(object_names), max(1, batch_size)):
        batch = object_names[start : start + max(1, batch_size)]
        for object_name in batch:
            try:
                result = enrich_file_metadata(
                    minio_client=minio_client,
                    redis_client=redis_client,
                    bucket=bucket,
                    object_name=object_name,
                    bibtex_prefix=bibtex_prefix,
                    source_prefix=source_prefix,
                    overwrite=overwrite,
                    dry_run=dry_run,
                    skip_complete=skip_complete,
                    crossref_client=crossref_client,
                )
            except (HTTPError, URLError, TimeoutError) as exc:
                logger.warning("Crossref request failed for %s/%s: %s", bucket, object_name, exc)
                result = {"bucket": bucket, "object_name": object_name, "status": "error", "error": str(exc)}
            except S3Error as exc:
                result = {"bucket": bucket, "object_name": object_name, "status": "error", "error": str(exc)}
            except Exception as exc:  # pragma: no cover - defensive runtime path
                logger.exception("Unexpected error for %s/%s", bucket, object_name)
                result = {"bucket": bucket, "object_name": object_name, "status": "error", "error": str(exc)}
            status = result.get("status", "error")
            counts[status] = counts.get(status, 0) + 1
            results.append(result)
    return {"bucket": bucket, "count": len(object_names), "counts": counts, "results": results}


def _get_redis_client(redis_url: str) -> Tuple[Any | None, str | None]:
    if not redis_url:
        return None, "REDIS_URL is required"
    if redis is None:
        return None, "redis package is required"
    try:
        return redis.from_url(redis_url), None
    except Exception as exc:  # pragma: no cover - runtime connectivity
        return None, f"Failed to connect to Redis: {exc}"


def _get_minio_client() -> Tuple[Any | None, str | None]:
    if Minio is None:
        return None, "minio package is required"
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
    secure = _load_env_bool("MINIO_SECURE", False)
    if not access_key or not secret_key:
        return None, "MINIO_ACCESS_KEY/MINIO_SECRET_KEY are required"
    try:
        return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure), None
    except Exception as exc:  # pragma: no cover - runtime connectivity
        return None, f"Failed to connect to MinIO: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-enrich BibTeX metadata from MinIO + Redis signals via Crossref.",
    )
    parser.add_argument("--bucket", action="append", default=[], help="Bucket to process (repeatable).")
    parser.add_argument("--all-buckets", action="store_true", help="Process all available MinIO buckets.")
    parser.add_argument("--limit", type=int, default=int(os.getenv("BIBTEX_AUTOFILL_LIMIT", "500")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BIBTEX_AUTOFILL_BATCH_SIZE", "25")))
    parser.add_argument("--object-prefix", default=os.getenv("BIBTEX_MINIO_PREFIX", os.getenv("MINIO_PREFIX", "")))
    parser.add_argument("--suffix", default=os.getenv("BIBTEX_MINIO_SUFFIX", ".pdf"))
    parser.add_argument("--bibtex-prefix", default=os.getenv("BIBTEX_REDIS_PREFIX", "bibtex"))
    parser.add_argument(
        "--source-redis-prefix",
        default=os.getenv("BIBTEX_SOURCE_REDIS_PREFIX", os.getenv("REDIS_PREFIX", "unstructured")),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing BibTeX fields.")
    parser.add_argument("--dry-run", action="store_true", help="Compute matches without writing Redis keys.")
    parser.add_argument(
        "--no-skip-complete",
        action="store_true",
        help="Process files even when existing BibTeX has title/authors/doi.",
    )
    parser.add_argument("--crossref-api-url", default=os.getenv("CROSSREF_API_URL", "https://api.crossref.org"))
    parser.add_argument("--crossref-mailto", default=os.getenv("CROSSREF_MAILTO", ""))
    parser.add_argument("--crossref-user-agent", default=os.getenv("CROSSREF_USER_AGENT", ""))
    parser.add_argument("--crossref-rows", type=int, default=int(os.getenv("CROSSREF_ROWS", "5")))
    parser.add_argument(
        "--crossref-timeout-seconds",
        type=int,
        default=int(os.getenv("CROSSREF_TIMEOUT_SECONDS", "20")),
    )
    parser.add_argument(
        "--crossref-throttle-seconds",
        type=float,
        default=float(os.getenv("CROSSREF_THROTTLE_SECONDS", "0.15")),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(Path(".env"))
    args = parse_args()

    redis_url = os.getenv("BIBTEX_REDIS_URL") or os.getenv("REDIS_URL", "")
    redis_client, redis_error = _get_redis_client(redis_url)
    if redis_error or not redis_client:
        raise RuntimeError(redis_error or "Redis is not configured")

    minio_client, minio_error = _get_minio_client()
    if minio_error or not minio_client:
        raise RuntimeError(minio_error or "MinIO is not configured")

    buckets = _resolve_buckets(minio_client, args.bucket, args.all_buckets)
    if not buckets:
        raise RuntimeError("No buckets configured. Use --bucket, --all-buckets, or MINIO_BUCKETS.")

    crossref_client = CrossrefClient(
        api_url=args.crossref_api_url,
        timeout_seconds=args.crossref_timeout_seconds,
        rows=args.crossref_rows,
        mailto=args.crossref_mailto,
        user_agent=args.crossref_user_agent,
        throttle_seconds=args.crossref_throttle_seconds,
    )

    bucket_reports: List[Dict[str, Any]] = []
    total_counts: Dict[str, int] = {}
    for bucket in buckets:
        logger.info("Starting bibliography autofill for bucket=%s", bucket)
        report = enrich_bucket_files(
            minio_client=minio_client,
            redis_client=redis_client,
            bucket=bucket,
            limit=max(1, args.limit),
            batch_size=max(1, args.batch_size),
            object_prefix=args.object_prefix,
            suffix=args.suffix,
            bibtex_prefix=args.bibtex_prefix,
            source_prefix=args.source_redis_prefix,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            skip_complete=not args.no_skip_complete,
            crossref_client=crossref_client,
        )
        bucket_reports.append(report)
        for key, value in report["counts"].items():
            total_counts[key] = total_counts.get(key, 0) + value

    summary = {
        "buckets": buckets,
        "bucket_reports": bucket_reports,
        "counts": total_counts,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
