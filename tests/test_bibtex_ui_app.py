import json
import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mcp_research import bibtex_ui_app

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    TestClient = None  # type: ignore[assignment]


class _FakeRedisPipeline:
    def __init__(self, client):
        self._client = client
        self._keys = []

    def get(self, key):
        self._keys.append(key)
        return self

    def execute(self):
        return [self._client.get(key) for key in self._keys]


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

    def hgetall(self, key):
        return self.hashes.get(key, {})

    def pipeline(self):
        return _FakeRedisPipeline(self)

    def smembers(self, key):
        return self.sets.get(key, set())


class _FakeBucket:
    def __init__(self, name):
        self.name = name


class _FakeObject:
    def __init__(self, object_name):
        self.object_name = object_name
        self.is_dir = False


class _FakeMinio:
    def __init__(self, buckets):
        self._buckets = buckets

    def list_buckets(self):
        return [_FakeBucket(name) for name in self._buckets.keys()]

    def list_objects(self, bucket, prefix="", recursive=True):
        del recursive
        for object_name in self._buckets.get(bucket, []):
            if prefix and not object_name.startswith(prefix):
                continue
            yield _FakeObject(object_name)


@unittest.skipIf(bibtex_ui_app.app is None or TestClient is None, "FastAPI test client unavailable")
class BibtexUiAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(bibtex_ui_app.app)

    def test_index_serves_bibtex_workspace_shell(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("BibTeX Metadata Workspace", response.text)
        self.assertIn("Buckets", response.text)
        self.assertIn("BibTeX Fields", response.text)

    def test_api_buckets_lists_minio_buckets(self):
        fake_minio = _FakeMinio({"bucket-a": [], "bucket-b": []})
        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            with mock.patch.object(
                bibtex_ui_app,
                "_resolve_minio_buckets",
                return_value=(["bucket-a", "bucket-b"], None),
            ):
                response = self.client.get("/api/buckets")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["buckets"], ["bucket-a", "bucket-b"])

    def test_api_bucket_files_loads_pdf_objects_and_bibtex_metadata(self):
        fake_minio = _FakeMinio({"bucket-a": ["folder/alpha.pdf", "beta.pdf", "ignore.txt"]})
        fake_redis = _FakeRedis()
        fake_redis.set(
            "bibtex:file:bucket-a/folder/alpha.pdf",
            json.dumps(
                {
                    "title": "Alpha Paper",
                    "year": "2024",
                    "citationKey": "alpha2024paper",
                    "entryType": "article",
                    "authors": [{"firstName": "Ada", "lastName": "Lovelace"}],
                }
            ),
        )

        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
                with mock.patch.dict(os.environ, {"BIBTEX_REDIS_PREFIX": "bibtex"}, clear=False):
                    response = self.client.get("/api/buckets/bucket-a/files")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 2)
        files = payload["files"]
        self.assertEqual([entry["objectName"] for entry in files], ["beta.pdf", "folder/alpha.pdf"])
        alpha = next(entry for entry in files if entry["objectName"] == "folder/alpha.pdf")
        self.assertEqual(alpha["title"], "Alpha Paper")
        self.assertEqual(alpha["citationKey"], "alpha2024paper")
        self.assertEqual(alpha["authors"][0]["lastName"], "Lovelace")
        self.assertIn("/r/doc?ref=", alpha["originalFileUrl"])

    def test_api_save_file_bibtex_persists_to_redis_prefix(self):
        fake_redis = _FakeRedis()
        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.dict(os.environ, {"BIBTEX_REDIS_PREFIX": "bibtex"}, clear=False):
                response = self.client.put(
                    "/api/buckets/bucket-a/files/folder%2Falpha.pdf/bibtex",
                    json={
                        "title": "Updated Alpha",
                        "year": "2025",
                        "citationKey": "alpha2025updated",
                        "entryType": "incollection",
                        "publisher": "ACM Press",
                        "authors": [
                            {"firstName": "Ada", "lastName": "Lovelace"},
                            {"firstName": "Grace", "lastName": "Hopper"},
                        ],
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["redis_key"], "bibtex:file:bucket-a/folder/alpha.pdf")
        self.assertEqual(body["file"]["title"], "Updated Alpha")
        self.assertEqual(body["file"]["entryType"], "incollection")
        self.assertEqual(body["file"]["publisher"], "ACM Press")
        self.assertIn("bucket-a/folder/alpha.pdf", fake_redis.sets.get("bibtex:files", set()))

        stored = json.loads(fake_redis.get("bibtex:file:bucket-a/folder/alpha.pdf"))
        self.assertEqual(stored["year"], "2025")
        self.assertEqual(stored["authors"][1]["lastName"], "Hopper")

    def test_api_file_redis_summary_counts_partitions_and_chunks(self):
        fake_redis = _FakeRedis()
        fake_redis.sets["unstructured:pdf:source:bucket-a/folder/alpha.pdf"] = {"doc1"}
        fake_redis.values["unstructured:pdf:doc1:partitions"] = json.dumps([{"text": "p1"}, {"text": "p2"}])
        fake_redis.values["unstructured:pdf:doc1:chunks"] = json.dumps([{"text": "c1"}, {"text": "c2"}, {"text": "c3"}])

        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.dict(os.environ, {"REDIS_PREFIX": "unstructured"}, clear=False):
                response = self.client.get("/api/buckets/bucket-a/files/folder%2Falpha.pdf/redis-summary")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["partition_count"], 2)
        self.assertEqual(payload["chunk_count"], 3)
        self.assertEqual(payload["doc_ids"], ["doc1"])

    def test_api_file_redis_data_lazy_limits_items(self):
        fake_redis = _FakeRedis()
        fake_redis.sets["unstructured:pdf:source:bucket-a/folder/alpha.pdf"] = {"doc1"}
        fake_redis.values["unstructured:pdf:doc1:chunks"] = json.dumps(
            [{"text": "chunk one"}, {"text": "chunk two"}, {"text": "chunk three"}]
        )

        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.dict(os.environ, {"REDIS_PREFIX": "unstructured"}, clear=False):
                response = self.client.get("/api/buckets/bucket-a/files/folder%2Falpha.pdf/redis-data?kind=chunks&limit=2")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["kind"], "chunks")
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["total_available"], 3)
        self.assertTrue(payload["truncated"])
        self.assertIn("chunk one", payload["items"][0]["text"])

    def test_api_bucket_autofill_missing_processes_in_batches(self):
        if bibtex_ui_app.bibtex_autofill is None:
            self.skipTest("bibtex_autofill module unavailable")
        fake_minio = _FakeMinio({"bucket-a": ["alpha.pdf", "beta.pdf", "gamma.pdf"]})
        fake_redis = _FakeRedis()
        calls = []

        def fake_enrich_file_metadata(**kwargs):
            calls.append(kwargs)
            object_name = kwargs["object_name"]
            if object_name == "alpha.pdf":
                return {"status": "updated"}
            if object_name == "beta.pdf":
                return {"status": "no_match"}
            return {"status": "skipped_existing"}

        with mock.patch.object(bibtex_ui_app, "_get_minio_client", return_value=(fake_minio, None)):
            with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
                with mock.patch.object(bibtex_ui_app.bibtex_autofill, "CrossrefClient", return_value=object()):
                    with mock.patch.object(
                        bibtex_ui_app.bibtex_autofill,
                        "enrich_file_metadata",
                        side_effect=fake_enrich_file_metadata,
                    ):
                        first = self.client.post(
                            "/api/buckets/bucket-a/autofill-missing",
                            json={
                                "objectNames": ["alpha.pdf", "beta.pdf", "gamma.pdf"],
                                "offset": 0,
                                "batchSize": 2,
                            },
                        )
                        second = self.client.post(
                            "/api/buckets/bucket-a/autofill-missing",
                            json={
                                "objectNames": ["alpha.pdf", "beta.pdf", "gamma.pdf"],
                                "offset": 2,
                                "batchSize": 2,
                            },
                        )

        self.assertEqual(first.status_code, 200)
        first_payload = first.json()
        self.assertEqual(first_payload["processed_total"], 2)
        self.assertEqual(first_payload["processed_in_batch"], 2)
        self.assertFalse(first_payload["done"])
        self.assertEqual(first_payload["next_offset"], 2)
        self.assertEqual(first_payload["counts"]["updated"], 1)
        self.assertEqual(first_payload["counts"]["no_match"], 1)

        self.assertEqual(second.status_code, 200)
        second_payload = second.json()
        self.assertEqual(second_payload["processed_total"], 3)
        self.assertEqual(second_payload["processed_in_batch"], 1)
        self.assertTrue(second_payload["done"])
        self.assertIsNone(second_payload["next_offset"])
        self.assertEqual(second_payload["counts"]["skipped_existing"], 1)

        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertFalse(call["overwrite"])
            self.assertTrue(call["skip_complete"])
            self.assertFalse(call["dry_run"])

    def test_api_file_lookup_by_doi_overwrites_metadata_from_crossref(self):
        if bibtex_ui_app.bibtex_autofill is None:
            self.skipTest("bibtex_autofill module unavailable")

        fake_redis = _FakeRedis()
        fake_redis.set(
            "bibtex:file:bucket-a/folder/alpha.pdf",
            json.dumps(
                {
                    "title": "Old Title",
                    "doi": "10.1111/old",
                    "citationKey": "oldkey",
                    "entryType": "article",
                    "authors": [{"firstName": "Old", "lastName": "Author"}],
                }
            ),
        )

        class _FakeCrossrefClient:
            def lookup_by_doi(self, doi):
                self.last_doi = doi
                return {
                    "DOI": "10.1000/testdoi",
                    "title": ["CrossRef Title"],
                    "author": [{"given": "Ada", "family": "Lovelace"}],
                    "issued": {"date-parts": [[2024]]},
                    "type": "journal-article",
                    "container-title": ["Journal of Testing"],
                    "URL": "https://doi.org/10.1000/testdoi",
                }

        fake_crossref = _FakeCrossrefClient()
        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            with mock.patch.object(bibtex_ui_app, "_crossref_client_from_env", return_value=fake_crossref):
                with mock.patch.dict(os.environ, {"BIBTEX_REDIS_PREFIX": "bibtex"}, clear=False):
                    response = self.client.post(
                        "/api/buckets/bucket-a/files/folder%2Falpha.pdf/lookup-by-doi",
                        json={"doi": "10.1000/testdoi", "overwrite": True},
                    )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["doi"], "10.1000/testdoi")
        self.assertEqual(body["file"]["title"], "CrossRef Title")
        self.assertEqual(body["file"]["year"], "2024")
        self.assertEqual(body["file"]["doi"], "10.1000/testdoi")
        self.assertEqual(body["file"]["authors"][0]["lastName"], "Lovelace")

        stored = json.loads(fake_redis.get("bibtex:file:bucket-a/folder/alpha.pdf"))
        self.assertEqual(stored["title"], "CrossRef Title")
        self.assertEqual(stored["doi"], "10.1000/testdoi")

    def test_api_file_lookup_by_doi_requires_valid_doi(self):
        if bibtex_ui_app.bibtex_autofill is None:
            self.skipTest("bibtex_autofill module unavailable")

        fake_redis = _FakeRedis()
        with mock.patch.object(bibtex_ui_app, "_get_redis_client", return_value=(fake_redis, None)):
            response = self.client.post(
                "/api/buckets/bucket-a/files/folder%2Falpha.pdf/lookup-by-doi",
                json={"doi": ""},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("valid DOI", response.json().get("detail", ""))


if __name__ == "__main__":
    unittest.main()
