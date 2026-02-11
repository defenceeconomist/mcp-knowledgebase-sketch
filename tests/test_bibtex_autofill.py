import json
import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import bibtex_autofill


class _FakeRedis:
    def __init__(self):
        self.values = {}
        self.sets = {}
        self.hashes = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value

    def sadd(self, key, value):
        self.sets.setdefault(key, set()).add(value)

    def smembers(self, key):
        return self.sets.get(key, set())

    def hgetall(self, key):
        return self.hashes.get(key, {})


class _FakeObject:
    def __init__(self, object_name):
        self.object_name = object_name
        self.is_dir = False


class _FakeBucket:
    def __init__(self, name):
        self.name = name


class _FakeStat:
    def __init__(self, metadata):
        self.metadata = metadata


class _FakeMinio:
    def __init__(self, objects_by_bucket, metadata_by_object):
        self._objects_by_bucket = objects_by_bucket
        self._metadata_by_object = metadata_by_object

    def list_objects(self, bucket, prefix="", recursive=True):
        del recursive
        for object_name in self._objects_by_bucket.get(bucket, []):
            if prefix and not object_name.startswith(prefix):
                continue
            yield _FakeObject(object_name)

    def stat_object(self, bucket, object_name):
        metadata = self._metadata_by_object.get((bucket, object_name), {})
        return _FakeStat(metadata)

    def list_buckets(self):
        return [_FakeBucket(name) for name in self._objects_by_bucket.keys()]


class _FakeCrossrefClient:
    def fetch_best_match(self, *, doi, title, authors):
        del authors
        if doi == "10.1000/xyz123":
            return {
                "DOI": "10.1000/xyz123",
                "title": ["Alpha Metadata Title"],
                "type": "journal-article",
                "author": [{"given": "Ada", "family": "Lovelace"}],
                "container-title": ["Journal of Test Fixtures"],
                "issued": {"date-parts": [[2024, 2, 1]]},
                "volume": "12",
                "issue": "2",
                "page": "10-20",
                "URL": "https://doi.org/10.1000/xyz123",
            }
        if title == "Neural Widget Systems":
            return {
                "DOI": "10.2000/abc987",
                "title": ["Neural Widget Systems"],
                "type": "proceedings-article",
                "author": [
                    {"given": "Grace", "family": "Hopper"},
                    {"given": "Katherine", "family": "Johnson"},
                ],
                "container-title": ["Proceedings of WidgetConf"],
                "issued": {"date-parts": [[2025, 5, 3]]},
                "page": "55-67",
                "URL": "https://doi.org/10.2000/abc987",
            }
        return None


class _LowConfidenceClient:
    def resolve_best_match(self, **_kwargs):
        return {
            "status": "low_confidence",
            "message": None,
            "diagnostics": {"confidence": 0.41, "matched_by": "search"},
            "candidates": [{"title": "Wrong Match", "confidence": 0.41}],
        }


class BibtexAutofillTests(unittest.TestCase):
    def test_extract_candidate_signals_prefers_minio_metadata(self):
        metadata = {
            "x-amz-meta-title": "Paper From Metadata",
            "x-amz-meta-authors": "Ada Lovelace; Grace Hopper",
            "x-amz-meta-doi": "https://doi.org/10.5000/demo123",
        }
        normalized = bibtex_autofill._normalize_minio_metadata(metadata)
        signals = bibtex_autofill._extract_candidate_signals(
            normalized,
            "Fallback text with DOI 10.9999/ignore and other content",
        )
        self.assertEqual(signals["title"], "Paper From Metadata")
        self.assertEqual(signals["doi"], "10.5000/demo123")
        self.assertEqual(signals["authors"], ["Ada Lovelace", "Grace Hopper"])

    def test_enrich_bucket_files_updates_batched_items(self):
        minio_client = _FakeMinio(
            objects_by_bucket={
                "bucket-a": ["alpha.pdf", "beta.pdf", "gamma.pdf", "notes.txt"],
            },
            metadata_by_object={
                ("bucket-a", "alpha.pdf"): {
                    "x-amz-meta-title": "Alpha metadata title",
                    "x-amz-meta-authors": "Ada Lovelace",
                    "x-amz-meta-doi": "10.1000/xyz123",
                },
                ("bucket-a", "beta.pdf"): {},
                ("bucket-a", "gamma.pdf"): {},
            },
        )

        redis_client = _FakeRedis()
        redis_client.sets["unstructured:pdf:source:bucket-a/alpha.pdf"] = {"doc-alpha"}
        redis_client.values["unstructured:pdf:doc-alpha:chunks"] = json.dumps(
            [
                {"page_start": 2, "chunk_index": 0, "text": "Later page text"},
                {"page_start": 1, "chunk_index": 0, "text": "First page alpha content"},
            ]
        )
        redis_client.sets["unstructured:pdf:source:bucket-a/beta.pdf"] = {"doc-beta"}
        redis_client.values["unstructured:pdf:doc-beta:chunks"] = json.dumps(
            [
                {
                    "page_start": 1,
                    "chunk_index": 0,
                    "text": "Neural Widget Systems\nGrace Hopper, Katherine Johnson\nAbstract: ...",
                }
            ]
        )
        existing_gamma = {
            "citationKey": "gamma2024existing",
            "entryType": "article",
            "title": "Existing Gamma Title",
            "year": "2024",
            "authors": [{"firstName": "Alan", "lastName": "Turing"}],
            "doi": "10.3000/existing",
        }
        redis_client.values["bibtex:file:bucket-a/gamma.pdf"] = json.dumps(existing_gamma)

        report = bibtex_autofill.enrich_bucket_files(
            minio_client=minio_client,
            redis_client=redis_client,
            bucket="bucket-a",
            limit=10,
            batch_size=2,
            object_prefix="",
            suffix=".pdf",
            bibtex_prefix="bibtex",
            source_prefix="unstructured",
            overwrite=False,
            dry_run=False,
            skip_complete=True,
            crossref_client=_FakeCrossrefClient(),
        )

        self.assertEqual(report["count"], 3)
        self.assertEqual(report["counts"]["updated"], 2)
        self.assertEqual(report["counts"]["skipped_existing"], 1)

        alpha_saved = json.loads(redis_client.values["bibtex:file:bucket-a/alpha.pdf"])
        beta_saved = json.loads(redis_client.values["bibtex:file:bucket-a/beta.pdf"])

        self.assertEqual(alpha_saved["doi"], "10.1000/xyz123")
        self.assertEqual(alpha_saved["title"], "Alpha Metadata Title")
        self.assertEqual(alpha_saved["journal"], "Journal of Test Fixtures")

        self.assertEqual(beta_saved["title"], "Neural Widget Systems")
        self.assertEqual(beta_saved["doi"], "10.2000/abc987")
        self.assertEqual(beta_saved["booktitle"], "Proceedings of WidgetConf")
        self.assertEqual(len(beta_saved["authors"]), 2)

        self.assertIn("bucket-a/alpha.pdf", redis_client.sets.get("bibtex:files", set()))
        self.assertIn("bucket-a/beta.pdf", redis_client.sets.get("bibtex:files", set()))

    def test_crossref_resolve_best_match_rejects_low_confidence(self):
        client = bibtex_autofill.CrossrefClient(throttle_seconds=0)
        with mock.patch.object(client, "lookup_by_doi", return_value=None):
            with mock.patch.object(
                client,
                "search",
                return_value=[
                    {
                        "score": 95,
                        "title": ["Completely Different Topic"],
                        "author": [{"given": "John", "family": "Doe"}],
                        "DOI": "10.5555/different",
                        "issued": {"date-parts": [[2020]]},
                    }
                ],
            ):
                result = client.resolve_best_match(
                    doi="",
                    title="Neural Widget Systems",
                    authors=["Grace Hopper"],
                    year=2025,
                    confidence_threshold=0.85,
                    min_title_similarity=0.55,
                    min_author_overlap=0.25,
                )

        self.assertEqual(result["status"], "low_confidence")
        self.assertIsNone(result["message"])
        self.assertGreater(len(result["candidates"]), 0)
        self.assertLess(result["diagnostics"].get("confidence", 1.0), 0.85)

    def test_crossref_resolve_best_match_detects_doi_conflict(self):
        client = bibtex_autofill.CrossrefClient(throttle_seconds=0)
        with mock.patch.object(
            client,
            "lookup_by_doi",
            return_value={
                "DOI": "10.9999/conflict",
                "title": ["Neural Widget Systems"],
                "author": [{"given": "Grace", "family": "Hopper"}],
            },
        ):
            result = client.resolve_best_match(
                doi="10.2000/abc987",
                title="Neural Widget Systems",
                authors=["Grace Hopper"],
                year=2025,
                confidence_threshold=0.85,
                min_title_similarity=0.55,
                min_author_overlap=0.25,
            )

        self.assertEqual(result["status"], "doi_conflict")
        self.assertIsNone(result["message"])
        self.assertEqual(result["diagnostics"].get("doi_consistency"), 0.0)

    def test_enrich_file_metadata_skips_low_confidence_candidates(self):
        minio_client = _FakeMinio(
            objects_by_bucket={"bucket-a": ["paper.pdf"]},
            metadata_by_object={("bucket-a", "paper.pdf"): {}},
        )
        redis_client = _FakeRedis()
        redis_client.sets["unstructured:pdf:source:bucket-a/paper.pdf"] = {"doc-paper"}
        redis_client.values["unstructured:pdf:doc-paper:chunks"] = json.dumps(
            [{"page_start": 1, "chunk_index": 0, "text": "Neural Widget Systems\nGrace Hopper"}]
        )

        result = bibtex_autofill.enrich_file_metadata(
            minio_client=minio_client,
            redis_client=redis_client,
            bucket="bucket-a",
            object_name="paper.pdf",
            bibtex_prefix="bibtex",
            source_prefix="unstructured",
            overwrite=False,
            dry_run=False,
            skip_complete=False,
            crossref_client=_LowConfidenceClient(),
        )

        self.assertEqual(result["status"], "low_confidence")
        self.assertEqual(result.get("match", {}).get("confidence"), 0.41)
        self.assertIsNone(redis_client.values.get("bibtex:file:bucket-a/paper.pdf"))

    def test_crossref_book_chapter_maps_to_incollection(self):
        message = {
            "DOI": "10.4000/inbook1",
            "title": ["A Chapter Title"],
            "type": "book-chapter",
            "author": [{"given": "Ada", "family": "Lovelace"}],
            "container-title": ["Collected Works of Computing"],
            "publisher": "Springer",
            "issued": {"date-parts": [[2022, 8, 12]]},
            "page": "101-120",
            "URL": "https://doi.org/10.4000/inbook1",
        }

        converted = bibtex_autofill._crossref_to_bibtex(message, "chapter.pdf")
        self.assertEqual(converted["entryType"], "incollection")
        self.assertEqual(converted["booktitle"], "Collected Works of Computing")
        self.assertEqual(converted["publisher"], "Springer")
        self.assertEqual(converted["journal"], "")


if __name__ == "__main__":
    unittest.main()
